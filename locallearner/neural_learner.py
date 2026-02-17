#!/usr/bin/env python3
"""NeuralLearner: learn a PCFG using neural context models and Rényi divergences.

Pipeline:
  1. Load corpus, compute basic statistics
  2. Ney-Essen clustering + NMF to get overshot candidate kernels
  3. Train single-word and pair cloze neural models
  4. Select true anchors from candidates using Rényi divergence asymmetry
  5. Estimate bottom-up (xi) parameters:
     - Lexical: xi(A->b) = E(b) * exp(-D_alpha(anchor_A || b))
     - Binary:  xi(A->BC) = E(bc)/(E(b)*E(c)) * exp(-D_alpha(anchor_A || bc))
     - Start:   xi(S->BC) from length-2 sentence counts
  6. Assemble into WCFG, convert to PCFG
"""

import os
import math
import time
import numpy as np
import torch
from collections import Counter

import wcfg
import neyessen
import ngram_counts
import nmf as nmf_module
from neural_features import (
    ClozeModel, PairClozeModel, ClozeDataset, PairClozeDataset,
    PositionalClozeModel, PositionalPairClozeModel,
    RNNClozeModel, train_rnn_cloze_model,
    build_vocab, train_cloze_model, train_pair_cloze_model,
    train_positional_cloze_model, train_positional_pair_cloze_model,
    save_model, load_model, load_sentences, save_kernels, load_kernels,
    BOUNDARY,
)


class NeuralLearner:
    """Learn a PCFG from an unlabeled corpus using neural context models."""

    def __init__(self, corpus_path):
        """Load corpus and compute basic statistics.

        Args:
            corpus_path: path to corpus file (one sentence per line,
                         space-separated tokens)
        """
        self.corpus_path = corpus_path
        self.sentences = load_sentences(corpus_path)
        self.n_sentences = len(self.sentences)

        # Basic corpus statistics
        self.word_counts = Counter()
        self.bigram_counts = Counter()
        self.length1_counts = Counter()  # words that are complete sentences
        self.length2_counts = Counter()  # bigrams that are complete sentences
        self.total_tokens = 0
        for sent in self.sentences:
            for w in sent:
                self.word_counts[w] += 1
                self.total_tokens += 1
            for i in range(len(sent) - 1):
                self.bigram_counts[(sent[i], sent[i + 1])] += 1
            if len(sent) == 1:
                self.length1_counts[sent[0]] += 1
            if len(sent) == 2:
                self.length2_counts[(sent[0], sent[1])] += 1

        self.vocab = sorted(self.word_counts.keys())
        self.vocab_size = len(self.vocab)

        # Per-sentence expectations
        self.E_sent = {w: self.word_counts[w] / self.n_sentences
                       for w in self.word_counts}
        self.E_bigram = {bg: self.bigram_counts[bg] / self.n_sentences
                         for bg in self.bigram_counts}
        self.E_length1 = {w: self.length1_counts[w] / self.n_sentences
                          for w in self.length1_counts}
        self.E_length2 = {bg: self.length2_counts[bg] / self.n_sentences
                          for bg in self.length2_counts}

        # Hyperparameters (sensible defaults)
        self.k = 2                    # context window size
        self.seed = 42                # random seed
        self.number_clusters = 10     # Ney-Essen clusters
        self.min_count_nmf = None     # min word freq for NMF (auto if None)
        self.n_candidates = 20        # overshot kernel count
        self.alpha = 2.0              # Rényi divergence alpha
        self.min_context_count = 5    # min context frequency
        self.embedding_dim = 64       # neural model embedding dim
        self.hidden_dim = 128         # neural model hidden dim
        self.n_epochs = 10            # training epochs
        self.batch_size = 4096        # training batch size
        self.ssf = 1.0                # small sample factor for NMF
        self.model_type = 'bow'       # 'bow' (bag-of-words) or 'positional'

        # Cache paths (set to enable caching)
        self.cluster_file = None
        self.bigram_file = None
        self.single_model_file = None
        self.pair_model_file = None
        self.gap_model_file = None    # RNN gap model cache path
        self.kernels_file = None

        # State (populated during learning)
        self.clusters = None          # Ney-Essen word -> cluster mapping
        self.candidate_kernels = None # overshot kernel list from NMF
        self.single_model = None      # trained ClozeModel
        self.pair_model = None        # trained PairClozeModel
        self.gap_model = None         # RNN gap model (gap=1) for pair prediction
        self.w2i = None               # word -> index for neural models
        self.i2w = None               # index -> word for neural models
        self.anchors = None           # list of selected anchor words (excl S)
        self.nonterminals = None      # list of NT labels
        self.anchor2nt = None         # anchor word -> NT label mapping

    # ==========================================================
    # Step 1: NMF for overshot candidate kernels
    # ==========================================================

    def find_candidate_kernels(self, verbose=True):
        """Run Ney-Essen clustering + NMF to get overshot candidate kernels.

        Uses a fixed count (self.n_candidates) to force the NMF to overshoot,
        producing more kernels than the true number of nonterminals.
        """
        if self.kernels_file and os.path.exists(self.kernels_file):
            self.candidate_kernels, meta = load_kernels(self.kernels_file)
            if verbose:
                print(f"Loaded {len(self.candidate_kernels)} kernels "
                      f"from {self.kernels_file}")
            return self.candidate_kernels

        # Ney-Essen clustering
        if verbose:
            print("Ney-Essen clustering...")
        self._do_clustering(verbose=verbose)

        # Compute features for NMF
        if verbose:
            print("Computing NMF features...")
        n_clusters = self.number_clusters
        stride = n_clusters + 1
        n_features = 2 * stride  # width=1: left and right context

        # Build idx2word and word2idx for words above min_count
        if self.min_count_nmf is None:
            self.min_count_nmf = max(5, self.n_sentences // 1000)

        fidx2word = []
        fword2idx = {}
        for w in self.vocab:
            if self.word_counts[w] >= self.min_count_nmf:
                fword2idx[w] = len(fidx2word)
                fidx2word.append(w)

        n_words = len(fidx2word)
        features = np.zeros((n_words + 1, n_features))  # +1 for start symbol

        # Build features from bigram counts
        BDY = ngram_counts.BOUNDARY
        corpus_bigrams = ngram_counts.count_bigrams(self.sentences)

        for (w1, w2), count in corpus_bigrams.items():
            lc = 0 if w1 == BDY else self.clusters.get(w1, 0)
            rc = 0 if w2 == BDY else self.clusters.get(w2, 0)
            if w2 != BDY and w2 in fword2idx:
                features[fword2idx[w2], lc] += count
            if w1 != BDY and w1 in fword2idx:
                features[fword2idx[w1], stride + rc] += count

        # Start symbol vector
        start_vector = np.zeros(n_features)
        start_vector[0] = 1          # left context = boundary
        start_vector[stride] = 1     # right context = boundary
        features[n_words, :] = start_vector

        fwords = fidx2word + ['S']

        # Run NMF
        if verbose:
            print(f"NMF: {n_words} words, {n_features} features, "
                  f"target {self.n_candidates} kernels...")

        my_nmf = nmf_module.NMF(features, fwords, ssf=self.ssf)
        my_nmf.use_gram_schmidt = True

        # Fix start symbol
        start_idx = n_words
        start_norm = start_vector / np.sum(start_vector)
        my_nmf.data[start_idx, :] = start_norm
        my_nmf.raw_data[start_idx, :] = start_norm
        my_nmf.counts[start_idx] = 1e8
        my_nmf.start(start_idx)
        my_nmf.excluded.add(start_idx)

        kernels = ['S']
        for _ in range(self.n_candidates - 1):
            a, ai, d = my_nmf.find_but_dont_add()
            if a is None:
                break
            my_nmf.add_basis(ai)
            kernels.append(a)
            if verbose:
                print(f"  kernel {len(kernels)}: {a} (d={d:.6f})")

        self.candidate_kernels = kernels
        if verbose:
            print(f"Found {len(kernels)} candidate kernels")

        if self.kernels_file:
            save_kernels(kernels, self.kernels_file, metadata={
                'n_candidates': str(len(kernels)),
                'mode': 'overshot',
            })

        return self.candidate_kernels

    def _do_clustering(self, verbose=False):
        """Run or load Ney-Essen clustering."""
        if self.cluster_file and os.path.exists(self.cluster_file):
            self.clusters, meta = neyessen.load_cluster_dict(
                self.cluster_file)
            if verbose:
                print(f"  Loaded clusters from {self.cluster_file}")
            return

        myc = neyessen.Clustering()
        myc.clusters = self.number_clusters
        self.clusters = myc.cluster(
            self.sentences, seed=self.seed, verbose=verbose)

        if self.cluster_file:
            neyessen.save_cluster_dict(
                self.clusters, self.cluster_file,
                n_clusters=self.number_clusters, seed=self.seed)

    # ==========================================================
    # Step 2: Train neural models
    # ==========================================================

    def train_single_model(self, verbose=True):
        """Train single-word cloze model P(w | context)."""
        if self.single_model_file and os.path.exists(self.single_model_file):
            result = load_model(self.single_model_file, device='cpu')
            self.single_model, self.w2i, self.i2w = result[0], result[1], result[2]
            if verbose:
                print(f"Loaded single model from {self.single_model_file}")
            return

        if verbose:
            mt = self.model_type
            k_str = 'variable' if mt == 'rnn' else str(self.k)
            print(f"Training single cloze model ({mt}, k={k_str}, "
                  f"{self.n_epochs} epochs)...")

        if self.model_type == 'rnn':
            model, w2i, i2w, history = train_rnn_cloze_model(
                self.sentences,
                embedding_dim=self.embedding_dim, hidden_dim=self.hidden_dim,
                gru_dim=self.hidden_dim,
                n_epochs=self.n_epochs, batch_size=self.batch_size,
                lr=1e-3, min_count=1, seed=self.seed, verbose=verbose,
            )
        else:
            train_fn = (train_positional_cloze_model
                        if self.model_type == 'positional'
                        else train_cloze_model)
            model, w2i, i2w, history = train_fn(
                self.sentences, k=self.k,
                embedding_dim=self.embedding_dim, hidden_dim=self.hidden_dim,
                n_epochs=self.n_epochs, batch_size=self.batch_size,
                lr=1e-3, min_count=1, seed=self.seed, verbose=verbose,
            )
        # Move to CPU for inference (training may use MPS/CUDA)
        model = model.cpu()
        self.single_model = model
        self.w2i = w2i
        self.i2w = i2w

        if self.single_model_file:
            save_model(model, w2i, i2w, self.single_model_file,
                       k=self.k, history=history)

    def train_pair_model(self, verbose=True):
        """Train pair cloze model P(w1, w2 | context).

        For RNN model_type, trains a gap model (gap=1) instead of a
        separate pair architecture. The gap model is an RNNClozeModel
        with the same architecture as the normal model but trained on
        contexts where the right side skips one word.
        """
        if self.model_type == 'rnn':
            return self._train_gap_model(verbose=verbose)

        if self.pair_model_file and os.path.exists(self.pair_model_file):
            result = load_model(self.pair_model_file, device='cpu')
            self.pair_model = result[0]
            if verbose:
                print(f"Loaded pair model from {self.pair_model_file}")
            return

        if verbose:
            mt = self.model_type
            print(f"Training pair cloze model ({mt}, k={self.k}, "
                  f"{self.n_epochs} epochs)...")

        train_fn = (train_positional_pair_cloze_model if self.model_type == 'positional'
                    else train_pair_cloze_model)
        model, w2i, i2w, history = train_fn(
            self.sentences, k=self.k,
            embedding_dim=self.embedding_dim, hidden_dim=self.hidden_dim,
            n_epochs=self.n_epochs, batch_size=self.batch_size,
            lr=1e-3, min_count=1, seed=self.seed, verbose=verbose,
        )
        # Move to CPU for inference (training may use MPS/CUDA)
        model = model.cpu()
        self.pair_model = model

        # Ensure vocab is consistent
        if self.w2i is None:
            self.w2i = w2i
            self.i2w = i2w

        if self.pair_model_file:
            save_model(model, w2i, i2w, self.pair_model_file,
                       k=self.k, history=history)

    def _train_gap_model(self, verbose=True):
        """Train RNN gap model (gap=1) for pair prediction.

        Uses the same vocabulary as the normal RNN model.
        """
        if self.gap_model_file and os.path.exists(self.gap_model_file):
            result = load_model(self.gap_model_file, device='cpu')
            self.gap_model = result[0]
            if verbose:
                print(f"Loaded gap model from {self.gap_model_file}")
            return

        assert self.w2i is not None, "Train single model first"

        if verbose:
            print(f"Training RNN gap model (gap=1, "
                  f"{self.n_epochs} epochs)...")

        model, w2i, i2w, history = train_rnn_cloze_model(
            self.sentences,
            embedding_dim=self.embedding_dim, hidden_dim=self.hidden_dim,
            gru_dim=self.hidden_dim,
            n_epochs=self.n_epochs, batch_size=self.batch_size,
            lr=1e-3, min_count=1, seed=self.seed, verbose=verbose,
            gap=1, word2idx=self.w2i, idx2word=self.i2w,
        )
        model = model.cpu()
        self.gap_model = model

        if self.gap_model_file:
            save_model(model, w2i, i2w, self.gap_model_file,
                       k=0, history=history)

    # ==========================================================
    # Step 3: Select anchors using Rényi divergences
    # ==========================================================

    def select_anchors(self, verbose=True):
        """Select anchors from candidates using greedy furthest-point
        with Rényi divergence and asymmetry filtering.

        Uses the single model to compute D_alpha between candidate
        kernels. Iteratively selects the candidate with the highest
        minimum divergence from the current anchor set, stopping when
        the asymmetry test indicates the next candidate is spurious.
        """
        assert self.single_model is not None, "Train single model first"
        assert self.candidate_kernels is not None, "Find candidates first"

        k = self.k
        kernel_words = [w for w in self.candidate_kernels if w != 'S']

        if verbose:
            print(f"Selecting anchors from {len(kernel_words)} candidates...")

        # Collect contexts for each kernel word
        kernel_set = set(kernel_words)
        kernel_ctxs = {w: Counter() for w in kernel_words}
        for sent in self.sentences:
            padded = [BOUNDARY] * k + list(sent) + [BOUNDARY] * k
            for i in range(k, len(padded) - k):
                w = padded[i]
                if w in kernel_set:
                    ctx = tuple(padded[i - k:i] + padded[i + 1:i + k + 1])
                    kernel_ctxs[w][ctx] += 1

        # Precompute model predictions for each kernel's contexts
        kernel_log_p = {}  # word -> (log_p_all, weights) arrays
        for w in kernel_words:
            ctxs = kernel_ctxs[w]
            freq = {c: n for c, n in ctxs.items() if n >= self.min_context_count}
            if len(freq) < 20:
                freq = dict(ctxs.most_common(100))
            if len(freq) == 0:
                continue

            ctx_list = list(freq.keys())
            counts = np.array([freq[c] for c in ctx_list], dtype=np.float64)
            weights = counts / counts.sum()

            ctx_tensor = torch.zeros(len(ctx_list), 2 * k, dtype=torch.long)
            for ci, ctx in enumerate(ctx_list):
                for j, cw in enumerate(ctx):
                    ctx_tensor[ci, j] = self.w2i.get(cw, 0)

            with torch.no_grad():
                logits = self.single_model(ctx_tensor)
                log_p = torch.log_softmax(logits, dim=-1).numpy()

            kernel_log_p[w] = (log_p, weights)

        def d_renyi(u, v):
            """D_alpha(u || v) using u's contexts."""
            if u not in kernel_log_p or v not in self.w2i:
                return 0.0
            log_p, weights = kernel_log_p[u]
            u_vid = self.w2i[u]
            v_vid = self.w2i[v]
            log_E_u = math.log(self.E_sent[u])
            log_E_v = math.log(self.E_sent[v])

            log_ratio = (log_p[:, u_vid] - log_p[:, v_vid]
                         + log_E_v - log_E_u)

            alpha = self.alpha
            if alpha == float('inf'):
                return float(np.max(log_ratio))
            scaled = (alpha - 1) * log_ratio
            max_s = np.max(scaled)
            lse = max_s + np.log(np.sum(weights * np.exp(scaled - max_s)))
            return float(lse / (alpha - 1))

        # Greedy furthest-point selection
        # Start with the most frequent candidate
        candidates = [w for w in kernel_words if w in kernel_log_p]
        candidates.sort(key=lambda w: -self.word_counts[w])

        selected = [candidates[0]]
        remaining = set(candidates[1:])

        if verbose:
            print(f"  Start: {selected[0]}")

        while remaining:
            # For each remaining candidate, compute min divergence
            # from the current selected set
            best_word = None
            best_min_d = -float('inf')

            for w in remaining:
                min_d = min(d_renyi(w, s) for s in selected)
                if min_d > best_min_d:
                    best_min_d = min_d
                    best_word = w

            # Asymmetry check: is the best candidate spurious?
            # A spurious word w has some selected anchor s where
            # D(s || w) is low but D(w || s) is high
            is_spurious = False
            for s in selected:
                d_sw = d_renyi(s, best_word)  # D(s || w)
                d_ws = d_renyi(best_word, s)   # D(w || s)
                if d_sw < 0.5 and d_ws > 1.0:
                    is_spurious = True
                    if verbose:
                        print(f"  Reject {best_word}: asymmetry with {s} "
                              f"D(s||w)={d_sw:.3f}, D(w||s)={d_ws:.3f}")
                    break

            if is_spurious:
                remaining.discard(best_word)
                continue

            # Also stop if the min divergence is too low
            if best_min_d < 0.3:
                if verbose:
                    print(f"  Stop: best min_d={best_min_d:.3f} < 0.3 "
                          f"({best_word})")
                break

            selected.append(best_word)
            remaining.discard(best_word)
            if verbose:
                print(f"  Add {best_word}: min_d={best_min_d:.3f}")

        self.anchors = selected
        n_nt = len(selected)
        self.nonterminals = ['S'] + [f'NT_{w}' for w in selected]
        self.anchor2nt = {w: f'NT_{w}' for w in selected}

        if verbose:
            print(f"Selected {n_nt} anchors: {selected}")

        return self.anchors

    def select_anchors_minimal(self, tau_low=0.5, tau_high=1.5,
                               max_terminals=500, verbose=True):
        """Select anchors by finding terminals with minimal context
        distributions, using asymmetric Rényi divergences.

        Maintains an antichain of distributionally-minimal terminals.
        A terminal a is "smaller" than b if D(a||b) < tau_low and
        D(b||a) > tau_high, meaning a's context support is a subset
        of b's.

        Processes terminals in decreasing frequency order. The
        antichain is initialised with a synthetic S terminal whose
        context distribution is concentrated on the boundary-only
        context.

        Args:
            tau_low:  threshold below which divergence indicates
                      distribution inclusion.
            tau_high: threshold above which divergence indicates
                      the reverse direction is not included.
            max_terminals: only consider the top-N most frequent
                      terminals.
            verbose:  print progress trace.

        Returns:
            list of anchor words (excluding S).
        """
        assert self.single_model is not None, "Train single model first"

        k = self.k

        # --- 1. Select terminals to consider, sorted by frequency ---
        terminals = sorted(self.vocab, key=lambda w: -self.word_counts.get(w, 0))
        terminals = [w for w in terminals if w in self.w2i][:max_terminals]

        if verbose:
            print(f"Selecting minimal-distribution anchors from "
                  f"{len(terminals)} terminals (tau_low={tau_low}, "
                  f"tau_high={tau_high})...")

        # --- 2. Collect contexts for all terminals in one corpus pass ---
        is_rnn = self._is_rnn_model()
        terminal_set = set(terminals)
        terminal_ctxs = {w: Counter() for w in terminals}

        if is_rnn:
            for sent in self.sentences:
                words = list(sent)
                n = len(words)
                for i in range(n):
                    w = words[i]
                    if w in terminal_set:
                        left = tuple([BOUNDARY] + words[:i])
                        right = tuple(words[i + 1:] + [BOUNDARY])
                        terminal_ctxs[w][(left, right)] += 1
        else:
            for sent in self.sentences:
                padded = [BOUNDARY] * k + list(sent) + [BOUNDARY] * k
                for i in range(k, len(padded) - k):
                    w = padded[i]
                    if w in terminal_set:
                        ctx = tuple(padded[i - k:i] + padded[i + 1:i + k + 1])
                        terminal_ctxs[w][ctx] += 1

        # --- 3. Precompute neural log-probs per terminal ---
        terminal_log_p = {}  # word -> (log_p_matrix, weights)

        for w in terminals:
            ctxs = terminal_ctxs[w]
            freq = {c: n for c, n in ctxs.items()
                    if n >= self.min_context_count}
            if len(freq) < 20:
                freq = dict(ctxs.most_common(100))
            if len(freq) == 0:
                continue

            ctx_list = list(freq.keys())
            counts = np.array([freq[c] for c in ctx_list], dtype=np.float64)
            weights = counts / counts.sum()

            with torch.no_grad():
                if is_rnn:
                    log_p_list = []
                    for left_words, right_words in ctx_list:
                        left_t = torch.tensor(
                            [self.w2i.get(cw, 0) for cw in left_words],
                            dtype=torch.long).unsqueeze(0)
                        right_t = torch.tensor(
                            [self.w2i.get(cw, 0) for cw in right_words],
                            dtype=torch.long).unsqueeze(0)
                        logits = self.single_model.forward_unpacked(
                            left_t, right_t)
                        log_p_list.append(
                            torch.log_softmax(logits, dim=-1)[0].numpy())
                    log_p = np.stack(log_p_list)
                else:
                    ctx_tensor = torch.zeros(
                        len(ctx_list), 2 * k, dtype=torch.long)
                    for ci, ctx in enumerate(ctx_list):
                        for j, cw in enumerate(ctx):
                            ctx_tensor[ci, j] = self.w2i.get(cw, 0)
                    logits = self.single_model(ctx_tensor)
                    log_p = torch.log_softmax(logits, dim=-1).numpy()

            terminal_log_p[w] = (log_p, weights)

        # --- 4. Synthetic S terminal: boundary-only context ---
        bdy_id = self.w2i[BOUNDARY]
        with torch.no_grad():
            if is_rnn:
                bdy_left = torch.tensor([[bdy_id]], dtype=torch.long)
                bdy_right = torch.tensor([[bdy_id]], dtype=torch.long)
                s_logits = self.single_model.forward_unpacked(
                    bdy_left, bdy_right)
            else:
                s_ctx = torch.zeros(1, 2 * k, dtype=torch.long)
                s_ctx[:, :] = bdy_id
                s_logits = self.single_model(s_ctx)
            s_log_p = torch.log_softmax(s_logits, dim=-1).numpy()
        s_weights = np.array([1.0])
        terminal_log_p['<S>'] = (s_log_p, s_weights)

        # --- 5. Divergence helper ---
        def d_renyi(u, v):
            """D_alpha(u || v) using u's contexts."""
            if u not in terminal_log_p or v not in self.w2i:
                return float('inf')
            log_p, weights = terminal_log_p[u]

            if u == '<S>':
                # S doesn't have a vocab id; use its boundary log-probs
                # D(S || v): ratio of S's self-prediction to v's prediction
                # under boundary context. S doesn't correspond to a word,
                # so we skip D where u='<S>' -- it's only used as v.
                return float('inf')

            u_vid = self.w2i[u]
            v_vid = self.w2i.get(v, None)
            if v_vid is None:
                return float('inf')

            log_E_u = math.log(self.E_sent[u])
            log_E_v = math.log(self.E_sent[v]) if v != '<S>' else 0.0

            log_ratio = (log_p[:, u_vid] - log_p[:, v_vid]
                         + log_E_v - log_E_u)

            alpha = self.alpha
            if alpha == float('inf'):
                return float(np.max(log_ratio))
            scaled = (alpha - 1) * log_ratio
            max_s = np.max(scaled)
            lse = max_s + np.log(
                np.sum(weights * np.exp(scaled - max_s)))
            return float(lse / (alpha - 1))

        def d_renyi_from_s(v):
            """D(<S> || v): divergence from S to v.

            S's context distribution is a point mass on the boundary
            context.  The divergence measures how well v is predicted
            at boundary relative to its overall frequency.

            Closed form:  D(S||v) = log E(v) - log P(v|boundary)
            This is small when v is well-predicted at boundary
            (S-like) and large otherwise.
            """
            if v not in self.w2i or v not in self.E_sent:
                return float('inf')
            log_p, _ = terminal_log_p['<S>']
            v_vid = self.w2i[v]
            log_pv_bdy = float(log_p[0, v_vid])
            log_Ev = math.log(self.E_sent[v])
            return log_Ev - log_pv_bdy

        def d_renyi_to_s(a):
            """D(a || <S>): divergence from terminal a to S.

            Closed form using corpus statistics.  S generates only
            length-1 sentences, so its context is pure boundary.
            Across a's contexts, the fraction of mass at boundary
            context is f = E_length1(a) / E_sent(a).

            D(a||S) ≈ -log(f).  This is 0 when a appears only at
            boundary (a ≡ S), and large/infinite when a rarely
            appears as a length-1 sentence.
            """
            E_a = self.E_sent.get(a, 0)
            E1_a = self.E_length1.get(a, 0)
            if E_a <= 0:
                return float('inf')
            f = E1_a / E_a
            if f <= 0:
                return float('inf')
            return -math.log(f)

        # --- 6. Main loop: build the antichain ---
        # Antichain is a list of terminal names (including '<S>')
        antichain = ['<S>']

        if verbose:
            print(f"  Init: <S> (boundary)")

        # Only process terminals that have precomputed log-probs
        terminals_with_data = [w for w in terminals if w in terminal_log_p]

        for a in terminals_with_data:
            # Compare a to each element b in the antichain
            smaller_than = []  # elements b where a < b
            equal_to = []      # elements b where a == b
            larger_than = []   # elements b where a > b

            for b in antichain:
                if b == '<S>':
                    # S is an anchor with boundary-only context.
                    # Any terminal that ever appears as a length-1
                    # sentence contains S's distribution, so it is
                    # strictly larger than S.
                    E_a = self.E_sent.get(a, 0)
                    E1_a = self.E_length1.get(a, 0)
                    if E_a > 0 and E1_a / E_a > 1e-3:
                        larger_than.append(b)
                    # else: incomparable (never occurs as length-1)
                    continue
                else:
                    d_ab = d_renyi(a, b)  # D(a || b)
                    d_ba = d_renyi(b, a)  # D(b || a)

                if d_ab < tau_low and d_ba > tau_high:
                    # a's support ⊂ b's support: a is smaller
                    smaller_than.append(b)
                elif d_ab < tau_low and d_ba < tau_low:
                    # Same support: equal
                    equal_to.append(b)
                elif d_ab > tau_high and d_ba < tau_low:
                    # b's support ⊂ a's support: a is larger
                    larger_than.append(b)
                # else: incomparable

            if equal_to:
                # a is equivalent to something already in the list
                if verbose:
                    print(f"  Skip {a:>10} (freq={self.word_counts.get(a,0):>7}): "
                          f"equal to {equal_to[0]}")
                continue

            if larger_than:
                # a is larger than something in the list — not minimal
                if verbose:
                    print(f"  Skip {a:>10} (freq={self.word_counts.get(a,0):>7}): "
                          f"larger than {larger_than[0]}")
                continue

            if smaller_than:
                # a is smaller than some elements — remove them, add a
                for b in smaller_than:
                    antichain.remove(b)
                antichain.append(a)
                if verbose:
                    removed = [b for b in smaller_than if b != '<S>']
                    print(f"  Add  {a:>10} (freq={self.word_counts.get(a,0):>7}): "
                          f"replaces {removed if removed else ['<S>']}, "
                          f"antichain size={len(antichain)}")
                continue

            # Incomparable to everything — new minimal element
            antichain.append(a)
            if verbose:
                print(f"  Add  {a:>10} (freq={self.word_counts.get(a,0):>7}): "
                      f"incomparable, antichain size={len(antichain)}")

        # --- 7. Extract anchors (exclude synthetic S) ---
        selected = [w for w in antichain if w != '<S>']

        self.anchors = selected
        n_nt = len(selected)
        self.nonterminals = ['S'] + [f'NT_{w}' for w in selected]
        self.anchor2nt = {w: f'NT_{w}' for w in selected}

        if verbose:
            print(f"\nSelected {n_nt} minimal-distribution anchors: {selected}")

        return self.anchors

    # ==========================================================
    # Step 4: Estimate xi parameters
    # ==========================================================

    def _is_rnn_model(self):
        """Check if the single model is an RNN model."""
        return isinstance(self.single_model, RNNClozeModel)

    def _collect_anchor_contexts(self):
        """Collect single-word contexts for each anchor, with counts.

        For RNN models, collects variable-length (left, right) contexts.
        For fixed-width models, collects fixed-width context tuples.
        Also always collects fixed-width contexts for binary xi estimation
        (which uses the fixed-width pair model).
        """
        self._anchor_ctxs = {}
        for a in self.anchors:
            self._anchor_ctxs[a] = Counter()

        if self._is_rnn_model():
            # Variable-length contexts for RNN single model
            for sent in self.sentences:
                words = list(sent)
                n = len(words)
                for i in range(n):
                    w = words[i]
                    if w in self._anchor_ctxs:
                        left = tuple([BOUNDARY] + words[:i])
                        right = tuple(words[i + 1:] + [BOUNDARY])
                        self._anchor_ctxs[w][(left, right)] += 1
        else:
            k = self.k
            for sent in self.sentences:
                padded = [BOUNDARY] * k + list(sent) + [BOUNDARY] * k
                for i in range(k, len(padded) - k):
                    w = padded[i]
                    if w in self._anchor_ctxs:
                        ctx = tuple(padded[i - k:i] + padded[i + 1:i + k + 1])
                        self._anchor_ctxs[w][ctx] += 1

    def _get_filtered_contexts(self, anchor):
        """Get contexts for an anchor, filtered by frequency."""
        ctxs = self._anchor_ctxs[anchor]
        freq = {c: n for c, n in ctxs.items()
                if n >= self.min_context_count}
        if len(freq) < 20:
            freq = dict(ctxs.most_common(100))
        return freq

    def _collect_gap_counts(self, verbose=False):
        """Compute per-context correction for 1-gap vs 2-gap contexts.

        The single model gives P(a | l,r) where l_r is a 1-gap context,
        the pair model gives P(bc | l,r) where l__r is a 2-gap context.
        To form the ratio P(lar)/P(lbcr) we need:

            log P(lar) = log P(a|l_r)  + log P(l_r as 1-gap)
            log P(lbcr)= log P(bc|l__r)+ log P(l__r as 2-gap)

        So each context's log ratio needs a correction term:
            gap_correction(l,r) = log P(l,r 1-gap) - log P(l,r 2-gap)
                                = log(N1(l,r)/T1) - log(N2(l,r)/T2)

        where N1, N2 are counts of (l,r) appearing as 1-gap / 2-gap
        and T1, T2 are total 1-gap / 2-gap positions in the corpus.
        """
        k = self.k

        # Collect all unique anchor contexts we need corrections for
        needed = set()
        for a in self.anchors:
            needed.update(self._anchor_ctxs[a].keys())

        n1_counts = Counter()
        n2_counts = Counter()
        total_1gap = 0
        total_2gap = 0

        for sent in self.sentences:
            padded = [BOUNDARY] * k + list(sent) + [BOUNDARY] * k
            n = len(padded)

            # 1-gap: word at position i, context is k before + k after
            for i in range(k, n - k):
                total_1gap += 1
                ctx = tuple(padded[i - k:i] + padded[i + 1:i + k + 1])
                if ctx in needed:
                    n1_counts[ctx] += 1

            # 2-gap: pair at positions i, i+1, context is k before + k after
            for i in range(k, n - k - 1):
                total_2gap += 1
                ctx = tuple(padded[i - k:i] + padded[i + 2:i + k + 2])
                if ctx in needed:
                    n2_counts[ctx] += 1

        global_offset = math.log(total_2gap) - math.log(total_1gap)

        self._gap_corrections = {}
        n_missing = 0
        for ctx in needed:
            c1 = n1_counts.get(ctx, 0)
            c2 = n2_counts.get(ctx, 0)
            if c1 > 0 and c2 > 0:
                self._gap_corrections[ctx] = (
                    math.log(c1) - math.log(c2) + global_offset)
            elif c1 > 0:
                # Context never seen with 2 gaps — use smoothed value
                self._gap_corrections[ctx] = (
                    math.log(c1) - math.log(0.5) + global_offset)
                n_missing += 1
            else:
                self._gap_corrections[ctx] = 0.0

        if verbose:
            corrections = list(self._gap_corrections.values())
            print(f"  Gap corrections: {len(corrections)} contexts, "
                  f"{n_missing} missing 2-gap counts")
            if corrections:
                arr = np.array(corrections)
                print(f"  mean={arr.mean():.3f}, std={arr.std():.3f}, "
                      f"min={arr.min():.3f}, max={arr.max():.3f}")

    def _renyi_divergence(self, log_p, anchor_vid, b_vid,
                          log_E_a, log_E_b, weights):
        """Compute D_alpha(a || b) from precomputed log probs."""
        log_ratio = (log_p[:, anchor_vid] - log_p[:, b_vid]
                     + log_E_b - log_E_a)
        alpha = self.alpha
        if alpha == float('inf'):
            return float(np.max(log_ratio))
        scaled = (alpha - 1) * log_ratio
        max_s = np.max(scaled)
        lse = max_s + np.log(np.sum(weights * np.exp(scaled - max_s)))
        return float(lse / (alpha - 1))

    def _compute_full_log_probs(self, anchor):
        """Compute full (n_ctx, vocab) log-prob matrix for an anchor's contexts.

        Handles both fixed-width and RNN models.

        Returns:
            (log_p, weights): log_p is (n_ctx, vocab) numpy array,
                              weights is (n_ctx,) numpy array.
        """
        freq = self._get_filtered_contexts(anchor)
        ctx_list = list(freq.keys())
        counts = np.array([freq[c] for c in ctx_list], dtype=np.float64)
        weights = counts / counts.sum()

        with torch.no_grad():
            if self._is_rnn_model():
                log_p_list = []
                for left_words, right_words in ctx_list:
                    left_t = torch.tensor(
                        [self.w2i.get(w, 0) for w in left_words],
                        dtype=torch.long).unsqueeze(0)
                    right_t = torch.tensor(
                        [self.w2i.get(w, 0) for w in right_words],
                        dtype=torch.long).unsqueeze(0)
                    logits = self.single_model.forward_unpacked(left_t, right_t)
                    log_p_list.append(
                        torch.log_softmax(logits, dim=-1)[0].numpy())
                log_p = np.stack(log_p_list)
            else:
                k = self.k
                n_ctx = len(ctx_list)
                ctx_tensor = torch.zeros(n_ctx, 2 * k, dtype=torch.long)
                for ci, ctx in enumerate(ctx_list):
                    for j, w in enumerate(ctx):
                        ctx_tensor[ci, j] = self.w2i.get(w, 0)
                logits = self.single_model(ctx_tensor)
                log_p = torch.log_softmax(logits, dim=-1).numpy()

        return log_p, weights

    def estimate_lexical_xi(self, verbose=True):
        """Estimate xi(A -> b) for all anchors A and terminals b.

        xi(A -> b) = E(b) * exp(-D_alpha(anchor_A || b))

        Returns:
            dict mapping (nt_label, terminal) -> xi value
        """
        assert self.single_model is not None
        assert self.anchors is not None

        if verbose:
            print("Estimating lexical xi parameters...")

        self._collect_anchor_contexts()

        self.lexical_xi = {}

        for anchor in self.anchors:
            nt = self.anchor2nt[anchor]
            a_vid = self.w2i[anchor]
            log_E_a = math.log(self.E_sent[anchor])

            log_p, weights = self._compute_full_log_probs(anchor)

            for b in self.vocab:
                if b not in self.w2i:
                    continue
                b_vid = self.w2i[b]
                log_E_b = math.log(self.E_sent[b])

                d = self._renyi_divergence(
                    log_p, a_vid, b_vid, log_E_a, log_E_b, weights)
                xi = self.E_sent[b] * math.exp(-d)
                self.lexical_xi[(nt, b)] = xi

        # S lexical rules: xi(S -> b) = E_length1(b)
        # These come directly from the frequency of b as a complete sentence.
        for b, e in self.E_length1.items():
            self.lexical_xi[('S', b)] = e

        if verbose:
            n_s_lex = sum(1 for k in self.lexical_xi if k[0] == 'S')
            print(f"  Estimated {len(self.lexical_xi)} lexical xi parameters "
                  f"({n_s_lex} for S)")

        return self.lexical_xi

    def _compute_pmi(self, anchor_B, anchor_C):
        """Compute PMI(b, c) = log E(bc) - log E(b) - log E(c).

        Returns:
            (pmi, e_bc) or (None, 0) if E(bc) == 0.
        """
        e_bc = self.E_bigram.get((anchor_B, anchor_C), 0)
        if e_bc == 0:
            return None, 0
        log_E_bc = math.log(e_bc)
        pmi = (log_E_bc
               - math.log(self.E_sent[anchor_B])
               - math.log(self.E_sent[anchor_C]))
        return pmi, e_bc

    def _prepare_anchor_contexts(self, anchor):
        """Prepare context tensors, weights, and gap corrections for an anchor.

        For fixed-width models returns:
            (ctx_tensor, weights, n_ctx, gap_corrections):
            - ctx_tensor: (n_ctx, 2*k) long tensor of context word indices
            - weights: normalised context weights
            - n_ctx: number of contexts
            - gap_corrections: numpy array of per-context log(P(1-gap)/P(2-gap)),
              or None if _collect_gap_counts has not been called.

        For RNN models returns:
            (ctx_data, weights, n_ctx, gap_corrections):
            - ctx_data: list of (left_tensor, right_tensor) pairs
            - weights: normalised context weights
            - n_ctx: number of contexts
            - gap_corrections: always None (gap correction not used with RNN)
        """
        freq = self._get_filtered_contexts(anchor)
        ctx_list = list(freq.keys())
        counts = np.array([freq[c] for c in ctx_list], dtype=np.float64)
        weights = counts / counts.sum()
        n_ctx = len(ctx_list)

        if self._is_rnn_model():
            ctx_data = []
            for left_words, right_words in ctx_list:
                left_ids = torch.tensor(
                    [self.w2i.get(w, 0) for w in left_words],
                    dtype=torch.long).unsqueeze(0)
                right_ids = torch.tensor(
                    [self.w2i.get(w, 0) for w in right_words],
                    dtype=torch.long).unsqueeze(0)
                ctx_data.append((left_ids, right_ids))
            return ctx_data, weights, n_ctx, None

        k = self.k
        ctx_tensor = torch.zeros(n_ctx, 2 * k, dtype=torch.long)
        for ci, ctx in enumerate(ctx_list):
            for j, w in enumerate(ctx):
                ctx_tensor[ci, j] = self.w2i.get(w, 0)

        gap_corrections = None
        if hasattr(self, '_gap_corrections'):
            gap_corrections = np.array([
                self._gap_corrections.get(ctx, 0.0) for ctx in ctx_list
            ])

        return ctx_tensor, weights, n_ctx, gap_corrections

    def _single_model_log_probs(self, ctx_data, word_vid):
        """Compute log P(word | ctx) for each context using the single model.

        Args:
            ctx_data: for fixed-width models, a (n_ctx, 2*k) tensor;
                      for RNN, a list of (left_tensor, right_tensor) pairs.
            word_vid: vocabulary index of the target word.

        Returns:
            numpy array of shape (n_ctx,).
        """
        with torch.no_grad():
            if self._is_rnn_model():
                log_probs = []
                for left_t, right_t in ctx_data:
                    logits = self.single_model.forward_unpacked(left_t, right_t)
                    lp = torch.log_softmax(logits, dim=-1)[0, word_vid].item()
                    log_probs.append(lp)
                return np.array(log_probs)
            else:
                logits = self.single_model(ctx_data)
                log_p = torch.log_softmax(logits, dim=-1)[:, word_vid].numpy()
                return log_p

    def _pair_model_log_probs(self, ctx_tensor, b_vid, c_vid):
        """Compute log P(b, c | ctx) = log P(b|ctx) + log P(c|ctx, b)
        for each context using the pair model.

        Returns:
            numpy array of shape (n_ctx,).
        """
        n_ctx = ctx_tensor.shape[0]
        with torch.no_grad():
            logits1 = self.pair_model.forward(ctx_tensor)
            log_p_b = torch.log_softmax(
                logits1, dim=-1)[:, b_vid].numpy()
            b_tensor = torch.full((n_ctx,), b_vid, dtype=torch.long)
            _, logits2 = self.pair_model.forward(ctx_tensor, b_tensor)
            log_p_c = torch.log_softmax(
                logits2, dim=-1)[:, c_vid].numpy()
        return log_p_b + log_p_c

    def _renyi_from_log_ratio(self, log_ratio, weights):
        """Compute Rényi divergence from pre-computed log ratios and weights.

        D_alpha = (1/(alpha-1)) * log E_weights[ exp((alpha-1) * log_ratio) ]

        Returns:
            float divergence value.
        """
        alpha = self.alpha
        if alpha == float('inf'):
            return float(np.max(log_ratio))
        scaled = (alpha - 1) * log_ratio
        max_s = np.max(scaled)
        lse = max_s + np.log(np.sum(weights * np.exp(scaled - max_s)))
        return float(lse / (alpha - 1))

    def _binary_divergence(self, log_p_a, log_E_a, log_p_bc, log_E_bc,
                           weights, gap_corrections=None):
        """Compute D_alpha(a || bc) from single and pair model log probs.

        The single model gives P(a | l,r) for 1-gap contexts, the pair
        model gives P(bc | l,r) for 2-gap contexts.  To form the correct
        ratio P(lar)/P(lbcr) we need to correct for the different
        marginal probabilities of seeing context (l,r) with 1 vs 2 gaps:

        log_ratio[ctx] = (log P(a|ctx) - log E(a))
                       - (log P(bc|ctx) - log E(bc))
                       + gap_correction(ctx)

        where gap_correction = log P(ctx as 1-gap) - log P(ctx as 2-gap).

        Returns:
            float divergence value.
        """
        log_ratio = (log_p_a - log_E_a) - (log_p_bc - log_E_bc)
        if gap_corrections is not None:
            log_ratio = log_ratio + gap_corrections
        return self._renyi_from_log_ratio(log_ratio, weights)

    def _estimate_S_binary_xi(self):
        """Estimate xi(S -> BC) for all anchor pairs B, C.

        xi(S -> BC) = E_length2(bc) / (E(b) * E(c))
        These come directly from the frequency of bc as a length-2 sentence.
        """
        for anchor_B in self.anchors:
            nt_B = self.anchor2nt[anchor_B]
            for anchor_C in self.anchors:
                nt_C = self.anchor2nt[anchor_C]
                e_l2 = self.E_length2.get(
                    (anchor_B, anchor_C), 0)
                e_b = self.E_sent[anchor_B]
                e_c = self.E_sent[anchor_C]
                xi = e_l2 / (e_b * e_c) if (e_b > 0 and e_c > 0) else 0
                self.binary_xi[('S', nt_B, nt_C)] = xi

    def _estimate_nonS_binary_xi(self):
        """Estimate xi(A -> BC) for all non-S anchors A and anchor pairs B, C.

        xi(A -> BC) = exp(PMI(b,c)) * exp(-D_alpha(a || bc))
        where PMI(b,c) = log E(bc) - log E(b) - log E(c)

        For fixed-width models:
            D_alpha uses single model for P(a|ctx) and pair model for
            P(bc|ctx), with a per-context gap correction.

        For RNN models:
            P(bc|l,r) = P(b|l,r;gap_model) * P(c|l·b,r;normal_model)
            No gap correction needed.
        """
        if self._is_rnn_model():
            return self._estimate_nonS_binary_xi_rnn()

        for anchor_A in self.anchors:
            nt_A = self.anchor2nt[anchor_A]
            a_vid = self.w2i[anchor_A]
            log_E_a = math.log(self.E_sent[anchor_A])

            ctx_tensor, weights, n_ctx, gap_corrections = (
                self._prepare_anchor_contexts(anchor_A))
            log_p_a = self._single_model_log_probs(ctx_tensor, a_vid)

            for anchor_B in self.anchors:
                nt_B = self.anchor2nt[anchor_B]
                b_vid = self.w2i[anchor_B]

                for anchor_C in self.anchors:
                    nt_C = self.anchor2nt[anchor_C]
                    c_vid = self.w2i[anchor_C]

                    pmi, e_bc = self._compute_pmi(anchor_B, anchor_C)
                    if pmi is None:
                        self.binary_xi[(nt_A, nt_B, nt_C)] = 0
                        continue

                    log_E_bc = math.log(e_bc)
                    log_p_bc = self._pair_model_log_probs(
                        ctx_tensor, b_vid, c_vid)

                    d = self._binary_divergence(
                        log_p_a, log_E_a, log_p_bc, log_E_bc,
                        weights, gap_corrections)

                    xi = math.exp(pmi - d)
                    self.binary_xi[(nt_A, nt_B, nt_C)] = xi

    def _rnn_pair_log_probs(self, ctx_data, b_vid, c_vid):
        """Compute log P(b,c | l,r) for RNN models.

        P(b,c|l,r) = P(b|l,r;gap_model) * P(c|l·b,r;normal_model)

        Args:
            ctx_data: list of (left_tensor, right_tensor) pairs,
                      where left/right are variable-length.
            b_vid: vocab index of first word (b)
            c_vid: vocab index of second word (c)

        Returns:
            numpy array of shape (n_ctx,) with log P(b,c|l,r).
        """
        log_probs = []
        b_id_tensor = torch.tensor([b_vid], dtype=torch.long)
        with torch.no_grad():
            for left_t, right_t in ctx_data:
                # P(b | l, r) using gap model
                logits_b = self.gap_model.forward_unpacked(left_t, right_t)
                log_p_b = torch.log_softmax(
                    logits_b, dim=-1)[0, b_vid].item()

                # P(c | l·b, r) using normal model
                # Append b to left context
                left_with_b = torch.cat([left_t, b_id_tensor.unsqueeze(0)],
                                        dim=1)
                logits_c = self.single_model.forward_unpacked(
                    left_with_b, right_t)
                log_p_c = torch.log_softmax(
                    logits_c, dim=-1)[0, c_vid].item()

                log_probs.append(log_p_b + log_p_c)
        return np.array(log_probs)

    def _estimate_nonS_binary_xi_rnn(self):
        """RNN version of non-S binary xi estimation.

        Uses gap model for P(b|l,r) and normal model for P(c|l·b,r).
        No gap correction needed since each model was trained on its
        respective context distribution.
        """
        for anchor_A in self.anchors:
            nt_A = self.anchor2nt[anchor_A]
            a_vid = self.w2i[anchor_A]
            log_E_a = math.log(self.E_sent[anchor_A])

            # Variable-length contexts for anchor A
            ctx_data, weights, n_ctx, _ = (
                self._prepare_anchor_contexts(anchor_A))
            log_p_a = self._single_model_log_probs(ctx_data, a_vid)

            for anchor_B in self.anchors:
                nt_B = self.anchor2nt[anchor_B]
                b_vid = self.w2i[anchor_B]

                for anchor_C in self.anchors:
                    nt_C = self.anchor2nt[anchor_C]
                    c_vid = self.w2i[anchor_C]

                    pmi, e_bc = self._compute_pmi(anchor_B, anchor_C)
                    if pmi is None:
                        self.binary_xi[(nt_A, nt_B, nt_C)] = 0
                        continue

                    log_E_bc = math.log(e_bc)
                    log_p_bc = self._rnn_pair_log_probs(
                        ctx_data, b_vid, c_vid)

                    d = self._binary_divergence(
                        log_p_a, log_E_a, log_p_bc, log_E_bc,
                        weights, gap_corrections=None)

                    xi = math.exp(pmi - d)
                    self.binary_xi[(nt_A, nt_B, nt_C)] = xi

    def estimate_binary_xi(self, verbose=True):
        """Estimate xi(A -> BC) for all triples of NTs.

        For non-S parents:
          xi(A -> BC) = exp(PMI(b,c)) * exp(-D_alpha(anchor_A || bc))

        For S:
          xi(S -> BC) = E_length2(bc) / (E(b) * E(c))

        Returns:
            dict mapping (nt_A, nt_B, nt_C) -> xi value
        """
        assert self.single_model is not None
        if self._is_rnn_model():
            assert self.gap_model is not None, "Train gap model first"
        else:
            assert self.pair_model is not None
        assert self.anchors is not None

        if verbose:
            print("Estimating binary xi parameters...")

        if not hasattr(self, '_anchor_ctxs'):
            self._collect_anchor_contexts()

        if not self._is_rnn_model():
            self._collect_gap_counts(verbose=verbose)

        self.binary_xi = {}

        self._estimate_nonS_binary_xi()
        self._estimate_S_binary_xi()

        if verbose:
            n_nonzero = sum(1 for v in self.binary_xi.values() if v > 0)
            print(f"  Estimated {len(self.binary_xi)} binary xi parameters "
                  f"({n_nonzero} nonzero)")

        return self.binary_xi

    # ==========================================================
    # Step 5: Build WCFG
    # ==========================================================

    def build_wcfg(self, verbose=True):
        """Assemble xi parameters into a WCFG in bottom-up form,
        then convert to a proper PCFG.

        Returns:
            wcfg.WCFG with xi parameters
        """
        assert self.lexical_xi is not None
        assert self.binary_xi is not None

        if verbose:
            print("Building WCFG...")

        g = wcfg.WCFG()
        g.start = 'S'
        g.nonterminals = set(self.nonterminals)
        g.terminals = set(self.vocab)

        # Lexical rules
        for (nt, b), xi in self.lexical_xi.items():
            if xi > 0:
                prod = (nt, b)
                g.productions.append(prod)
                g.parameters[prod] = xi

        # Binary rules
        for (ntA, ntB, ntC), xi in self.binary_xi.items():
            if xi > 0:
                prod = (ntA, ntB, ntC)
                g.productions.append(prod)
                g.parameters[prod] = xi

        g.set_log_parameters()

        self.xi_wcfg = g
        if verbose:
            print(f"  {len(g.productions)} productions "
                  f"({g.count_lexical()} lexical, "
                  f"{g.count_binary()} binary)")

        return g

    def convert_to_pcfg(self, verbose=True):
        """Convert xi-parameterized WCFG to a proper PCFG.

        Tries convert_parameters_xi2pi first. If that fails, works
        directly with the xi WCFG. In either case, applies
        renormalise_divergent_wcfg2 (if divergent) then Chi-Zhang
        renormalisation to get a consistent PCFG that preserves the
        conditional distribution over trees.
        """
        assert self.xi_wcfg is not None

        if verbose:
            print("Converting to PCFG...")

        try:
            grammar = self.xi_wcfg.convert_parameters_xi2pi()
            if verbose:
                print("  xi2pi conversion succeeded")
        except Exception as e:
            if verbose:
                print(f"  xi2pi failed ({e}), using xi WCFG directly")
            grammar = self.xi_wcfg.copy()

        if not grammar.is_convergent():
            if verbose:
                print("  Grammar divergent, rescaling...")
            grammar = grammar.renormalise_divergent_wcfg2()

        grammar.renormalise()
        self.output_pcfg = grammar

        if verbose:
            print(f"  PCFG: {len(grammar.nonterminals)} NTs, "
                  f"{len(grammar.terminals)} terminals, "
                  f"{len(grammar.productions)} productions")

        return grammar

    # ==========================================================
    # Full pipeline
    # ==========================================================

    def learn(self, verbose=True):
        """Run the full learning pipeline.

        Returns:
            wcfg.WCFG -- the learned PCFG
        """
        t0 = time.time()

        self.find_candidate_kernels(verbose=verbose)
        self.train_single_model(verbose=verbose)
        self.train_pair_model(verbose=verbose)
        self.select_anchors(verbose=verbose)
        self.estimate_lexical_xi(verbose=verbose)
        self.estimate_binary_xi(verbose=verbose)
        self.build_wcfg(verbose=verbose)
        pcfg = self.convert_to_pcfg(verbose=verbose)

        if verbose:
            print(f"\nTotal time: {time.time() - t0:.1f}s")

        return pcfg
