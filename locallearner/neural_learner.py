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
        self.min_context_count = 5    # min context frequency (fixed-width)
        self.n_context_samples = 500  # contexts to sample per word (RNN)
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
        self.split_model_A_file = None  # split-half model A cache
        self.split_model_B_file = None  # split-half model B cache
        self.kernels_file = None

        # State (populated during learning)
        self.clusters = None          # Ney-Essen word -> cluster mapping
        self.candidate_kernels = None # overshot kernel list from NMF
        self.single_model = None      # trained ClozeModel
        self.pair_model = None        # trained PairClozeModel
        self.gap_model = None         # RNN gap model (gap=1) for pair prediction
        self.split_model_A = None     # split-half model A
        self.split_model_B = None     # split-half model B
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

    def train_split_models(self, verbose=True):
        """Train two cloze models on independent halves of the corpus.

        Used for split-half noise calibration of divergence estimates.
        The full-corpus model must be trained first to provide the
        shared vocabulary (w2i/i2w).

        Stores models as self.split_model_A and self.split_model_B,
        and split sentence lists as self._split_sentences_A/B.
        """
        assert self.w2i is not None, "Train full model first (need vocab)"
        assert self._is_rnn_model(), "Split-half currently requires RNN model"

        # Reproducible split
        rng = np.random.RandomState(self.seed + 1000)
        indices = rng.permutation(self.n_sentences)
        half = self.n_sentences // 2
        self._split_sentences_A = [self.sentences[i] for i in indices[:half]]
        self._split_sentences_B = [self.sentences[i] for i in indices[half:]]

        if verbose:
            print(f"Split-half: {len(self._split_sentences_A)} + "
                  f"{len(self._split_sentences_B)} sentences")

        train_kwargs = dict(
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            gru_dim=self.hidden_dim,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            lr=1e-3, min_count=1,
            word2idx=self.w2i, idx2word=self.i2w,
            verbose=verbose,
        )

        # Model A
        if (self.split_model_A_file
                and os.path.exists(self.split_model_A_file)):
            result = load_model(self.split_model_A_file, device='cpu')
            self.split_model_A = result[0]
            if verbose:
                print(f"  Loaded split model A from "
                      f"{self.split_model_A_file}")
        else:
            if verbose:
                print(f"  Training split model A...")
            model_A, _, _, _ = train_rnn_cloze_model(
                self._split_sentences_A,
                seed=self.seed, **train_kwargs)
            self.split_model_A = model_A.cpu()
            if self.split_model_A_file:
                save_model(self.split_model_A, self.w2i, self.i2w,
                           self.split_model_A_file)

        # Model B
        if (self.split_model_B_file
                and os.path.exists(self.split_model_B_file)):
            result = load_model(self.split_model_B_file, device='cpu')
            self.split_model_B = result[0]
            if verbose:
                print(f"  Loaded split model B from "
                      f"{self.split_model_B_file}")
        else:
            if verbose:
                print(f"  Training split model B...")
            model_B, _, _, _ = train_rnn_cloze_model(
                self._split_sentences_B,
                seed=self.seed + 1, **train_kwargs)
            self.split_model_B = model_B.cpu()
            if self.split_model_B_file:
                save_model(self.split_model_B, self.w2i, self.i2w,
                           self.split_model_B_file)

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

    # ==========================================================
    # Context collection and log-prob computation
    # ==========================================================

    def _collect_positions(self, word_set, sentences=None):
        """Collect occurrence positions (sentence_idx, word_idx) for words.

        Args:
            word_set: set of words to collect positions for.
            sentences: sentence list to search (default: self.sentences).

        Returns:
            dict mapping word -> list of (sentence_idx, word_idx).
        """
        if sentences is None:
            sentences = self.sentences
        positions = {w: [] for w in word_set}
        for si, sent in enumerate(sentences):
            for wi, w in enumerate(sent):
                if w in positions:
                    positions[w].append((si, wi))
        return positions

    def _collect_fixed_contexts(self, word_set, sentences=None):
        """Collect fixed-width context tuples with counts for words.

        Args:
            word_set: set of words to collect contexts for.
            sentences: sentence list to search (default: self.sentences).

        Returns:
            dict mapping word -> Counter of context_tuple -> count.
        """
        if sentences is None:
            sentences = self.sentences
        k = self.k
        ctxs = {w: Counter() for w in word_set}
        for sent in sentences:
            padded = [BOUNDARY] * k + list(sent) + [BOUNDARY] * k
            for i in range(k, len(padded) - k):
                w = padded[i]
                if w in ctxs:
                    ctx = tuple(padded[i - k:i] + padded[i + 1:i + k + 1])
                    ctxs[w][ctx] += 1
        return ctxs

    def _rnn_log_probs_from_positions(self, positions, n_samples=None,
                                      rng=None, model=None, sentences=None):
        """Compute (log_p, weights) from occurrence positions using RNN.

        Optionally samples n_samples positions uniformly. Runs the
        model on each position's full-sentence context.

        Args:
            positions: list of (sentence_idx, word_idx).
            n_samples: max positions to sample (None = use all).
            rng: numpy RandomState for sampling.
            model: RNN model to use (default: self.single_model).
            sentences: sentence list for context lookup
                       (default: self.sentences).

        Returns:
            (log_p, weights) where log_p is (n_ctx, vocab) and
            weights is (n_ctx,), or None if no positions.
        """
        if model is None:
            model = self.single_model
        if sentences is None:
            sentences = self.sentences
        if not positions:
            return None
        if n_samples and len(positions) > n_samples and rng is not None:
            idx = rng.choice(len(positions), n_samples, replace=False)
            positions = [positions[i] for i in idx]

        log_p_list = []
        with torch.no_grad():
            for si, wi in positions:
                sent = sentences[si]
                left = [BOUNDARY] + list(sent[:wi])
                right = list(sent[wi + 1:]) + [BOUNDARY]
                left_t = torch.tensor(
                    [self.w2i.get(w, 0) for w in left],
                    dtype=torch.long).unsqueeze(0)
                right_t = torch.tensor(
                    [self.w2i.get(w, 0) for w in right],
                    dtype=torch.long).unsqueeze(0)
                logits = model.forward_unpacked(left_t, right_t)
                log_p_list.append(
                    torch.log_softmax(logits, dim=-1)[0].numpy())

        log_p = np.stack(log_p_list)
        weights = np.ones(len(log_p_list)) / len(log_p_list)
        return log_p, weights

    def _fixed_log_probs_from_contexts(self, ctx_counter, model=None):
        """Compute (log_p, weights) from a context Counter using fixed-width model.

        Filters by min_context_count, falls back to top-100 if too few pass.

        Args:
            ctx_counter: Counter mapping context_tuple -> count.
            model: fixed-width model to use (default: self.single_model).

        Returns:
            (log_p, weights) where log_p is (n_ctx, vocab) and
            weights is (n_ctx,), or None if no contexts.
        """
        if model is None:
            model = self.single_model
        freq = {c: n for c, n in ctx_counter.items()
                if n >= self.min_context_count}
        if len(freq) < 20:
            freq = dict(ctx_counter.most_common(100))
        if not freq:
            return None

        ctx_list = list(freq.keys())
        counts = np.array([freq[c] for c in ctx_list], dtype=np.float64)
        weights = counts / counts.sum()

        k = self.k
        ctx_tensor = torch.zeros(len(ctx_list), 2 * k, dtype=torch.long)
        for ci, ctx in enumerate(ctx_list):
            for j, w in enumerate(ctx):
                ctx_tensor[ci, j] = self.w2i.get(w, 0)

        with torch.no_grad():
            logits = model(ctx_tensor)
            log_p = torch.log_softmax(logits, dim=-1).numpy()

        return log_p, weights

    def _compute_terminal_log_probs(self, words, model=None, sentences=None):
        """Compute per-context log-prob matrices for a list of words.

        For RNN models, samples n_context_samples positions per word.
        For fixed-width models, collects and filters fixed-width contexts.

        Also adds a synthetic '<S>' entry (boundary-only context).

        Args:
            words: list of terminal words.
            model: neural model to use (default: self.single_model).
            sentences: sentence list for context collection
                       (default: self.sentences).

        Returns:
            dict mapping word -> (log_p, weights) where
            log_p is (n_ctx, vocab) numpy array and
            weights is (n_ctx,) numpy array.
        """
        if model is None:
            model = self.single_model
        if sentences is None:
            sentences = self.sentences
        word_set = set(words)
        result = {}

        if self._is_rnn_model():
            positions = self._collect_positions(word_set, sentences)
            rng = np.random.RandomState(self.seed)
            for w in words:
                lp = self._rnn_log_probs_from_positions(
                    positions[w], n_samples=self.n_context_samples, rng=rng,
                    model=model, sentences=sentences)
                if lp is not None:
                    result[w] = lp
        else:
            ctxs = self._collect_fixed_contexts(word_set, sentences)
            for w in words:
                lp = self._fixed_log_probs_from_contexts(ctxs[w],
                                                          model=model)
                if lp is not None:
                    result[w] = lp

        # Synthetic S terminal: boundary-only context
        bdy_id = self.w2i[BOUNDARY]
        with torch.no_grad():
            if self._is_rnn_model():
                bdy_left = torch.tensor([[bdy_id]], dtype=torch.long)
                bdy_right = torch.tensor([[bdy_id]], dtype=torch.long)
                s_logits = model.forward_unpacked(
                    bdy_left, bdy_right)
            else:
                s_ctx = torch.zeros(1, 2 * self.k, dtype=torch.long)
                s_ctx[:, :] = bdy_id
                s_logits = model(s_ctx)
            s_log_p = torch.log_softmax(s_logits, dim=-1).numpy()
        result['<S>'] = (s_log_p, np.array([1.0]))

        return result

    # ==========================================================
    # Subset testing and antichain construction
    # ==========================================================

    def _violation_fraction(self, terminal_log_p, u, v, epsilon):
        """Fraction of u's contexts where the subset u ⊆ v is violated.

        For each context c_i of u, computes
            r_i = log P(v|c_i) - log P(u|c_i) + log E(u) - log E(v)
        If u ⊆ v, r_i >= 0 everywhere. Returns the fraction where
        r_i < -epsilon.

        Returns 1.0 if comparison is unavailable.
        """
        if u not in terminal_log_p or u == '<S>':
            return 1.0
        u_vid = self.w2i.get(u)
        v_vid = self.w2i.get(v)
        if u_vid is None or v_vid is None:
            return 1.0

        log_p, _ = terminal_log_p[u]
        log_E_u = math.log(self.E_sent[u])
        log_E_v = math.log(self.E_sent[v]) if v != '<S>' else 0.0

        r = log_p[:, v_vid] - log_p[:, u_vid] + log_E_u - log_E_v
        return float(np.mean(r < -epsilon))

    def _estimate_noise_sigma(self, terminal_log_p, candidates, epsilon,
                              n_calibration=80):
        """Estimate the noise parameter sigma from the data.

        Model: for same-NT words (a, b) with corpus frequencies f_a, f_b,
        the per-context log-ratio r_i ~ N(0, sigma^2 * (1/f_a + 1/f_b)).
        The expected violation fraction is Phi(-eps / (sigma * sqrt(...))).

        Calibrates sigma from the smallest nonzero pairwise violation
        fractions among the most frequent candidates (which are most
        likely same-NT pairs).
        """
        from scipy.stats import norm

        n = min(n_calibration, len(candidates))
        cal = candidates[:n]

        sigmas = []
        for i, a in enumerate(cal):
            for j, b in enumerate(cal):
                if i >= j:
                    continue
                vf_ab = self._violation_fraction(
                    terminal_log_p, a, b, epsilon)
                vf_ba = self._violation_fraction(
                    terminal_log_p, b, a, epsilon)
                vf = min(vf_ab, vf_ba)
                if vf <= 0 or vf >= 0.5:
                    continue
                z = norm.ppf(vf)
                if z >= -0.01:
                    continue
                fa = self.word_counts.get(a, 1)
                fb = self.word_counts.get(b, 1)
                sigma = -epsilon / (z * math.sqrt(1.0 / fa + 1.0 / fb))
                sigmas.append(sigma)

        if not sigmas:
            return 100.0  # conservative fallback
        return float(np.percentile(sigmas, 75))

    def _adaptive_vf_threshold(self, sigma, fa, fb, epsilon,
                               confidence=0.99):
        """Pair-specific violation fraction threshold under the null.

        Under the null hypothesis (same distribution), the expected
        violation fraction depends on the noise level (sigma) and word
        frequencies. Returns the upper confidence bound.

        Args:
            sigma: estimated noise parameter from _estimate_noise_sigma.
            fa, fb: corpus frequencies of the two words.
            epsilon: log-ratio tolerance.
            confidence: confidence level for the threshold.
        """
        from scipy.stats import norm

        sigma_r = sigma * math.sqrt(1.0 / fa + 1.0 / fb)
        vf_null = norm.cdf(-epsilon / sigma_r)
        n_s = getattr(self, 'n_context_samples', 500)
        se = math.sqrt(max(vf_null * (1 - vf_null) / n_s, 1e-10))
        return vf_null + norm.ppf(confidence) * se

    def _build_antichain(self, candidates, is_subset, is_larger_than_S,
                         verbose=False):
        """Build antichain of minimal elements under a subset relation.

        Processes candidates in order (typically decreasing frequency).
        Maintains an antichain: a set where no element is a subset of
        another.

        Args:
            candidates: ordered list of terminal names.
            is_subset: callable(a, b) -> bool, True if a ⊆ b.
            is_larger_than_S: callable(a) -> bool, True if a > S.
            verbose: print progress trace.

        Returns:
            list of antichain elements (excluding '<S>').
        """
        antichain = ['<S>']
        if verbose:
            print(f"  Init: <S> (boundary)")

        for a in candidates:
            smaller_than = []
            equal_to = []
            larger_than = []

            for b in antichain:
                if b == '<S>':
                    if is_larger_than_S(a):
                        larger_than.append(b)
                    continue

                a_sub_b = is_subset(a, b)
                b_sub_a = is_subset(b, a)

                if a_sub_b and not b_sub_a:
                    smaller_than.append(b)
                elif a_sub_b and b_sub_a:
                    equal_to.append(b)
                elif not a_sub_b and b_sub_a:
                    larger_than.append(b)

            freq = self.word_counts.get(a, 0)

            if equal_to:
                if verbose:
                    print(f"  Skip {a:>10} (freq={freq:>7}): "
                          f"equal to {equal_to[0]}")
                continue

            if larger_than:
                if verbose:
                    print(f"  Skip {a:>10} (freq={freq:>7}): "
                          f"larger than {larger_than[0]}")
                continue

            if smaller_than:
                for b in smaller_than:
                    antichain.remove(b)
                antichain.append(a)
                if verbose:
                    removed = [b for b in smaller_than if b != '<S>']
                    print(f"  Add  {a:>10} (freq={freq:>7}): "
                          f"replaces {removed if removed else ['<S>']}, "
                          f"antichain size={len(antichain)}")
                continue

            antichain.append(a)
            if verbose:
                print(f"  Add  {a:>10} (freq={freq:>7}): "
                      f"incomparable, antichain size={len(antichain)}")

        return [w for w in antichain if w != '<S>']

    # ==========================================================
    # Anchor selection
    # ==========================================================

    def select_anchors_minimal(self, max_terminals=500, verbose=True,
                               epsilon=1.5):
        """Select anchors by finding terminals with minimal context
        distributions using a quantile-based subset test with
        self-calibrated frequency-adaptive thresholds.

        For each pair (a, b), computes per-context log-ratios
            r_i = log P(b|c_i) - log P(a|c_i) + log E(a) - log E(b)
        and concludes a ⊆ b if the violation fraction is below a
        pair-specific threshold that accounts for estimation noise.

        The noise model assumes log P(w|c) estimation error scales as
        sigma/sqrt(f_w). The constant sigma is self-calibrated from
        the lowest pairwise violation fractions in the data.

        Args:
            max_terminals: only consider the top-N most frequent terminals.
            verbose: print progress trace.
            epsilon: tolerance for log-ratio violations.

        Returns:
            list of anchor words (excluding S).
        """
        assert self.single_model is not None, "Train single model first"

        # 1. Select candidate terminals, sorted by frequency
        terminals = sorted(self.vocab, key=lambda w: -self.word_counts.get(w, 0))
        terminals = [w for w in terminals if w in self.w2i][:max_terminals]

        if verbose:
            print(f"Selecting minimal-distribution anchors from "
                  f"{len(terminals)} terminals (eps={epsilon})...")

        # 2. Precompute log-prob matrices
        terminal_log_p = self._compute_terminal_log_probs(terminals)
        candidates = [w for w in terminals if w in terminal_log_p]

        # 3. Calibrate noise parameter
        sigma = self._estimate_noise_sigma(
            terminal_log_p, candidates, epsilon)
        if verbose:
            print(f"  Noise sigma = {sigma:.1f}")

        # 4. Define subset test and S-comparison
        vf_floor = 0.10

        def is_subset(a, b):
            fa = self.word_counts.get(a, 1)
            fb = self.word_counts.get(b, 1)
            thresh = max(
                self._adaptive_vf_threshold(sigma, fa, fb, epsilon),
                vf_floor)
            vf = self._violation_fraction(
                terminal_log_p, a, b, epsilon)
            return vf <= thresh

        def is_larger_than_S(a):
            E_a = self.E_sent.get(a, 0)
            E1_a = self.E_length1.get(a, 0)
            return E_a > 0 and E1_a / E_a > 1e-3

        # 5. Build antichain
        selected = self._build_antichain(
            candidates, is_subset, is_larger_than_S, verbose=verbose)

        # 6. Store results
        self.anchors = selected
        self.nonterminals = ['S'] + [f'NT_{w}' for w in selected]
        self.anchor2nt = {w: f'NT_{w}' for w in selected}

        if verbose:
            print(f"\nSelected {len(selected)} minimal-distribution "
                  f"anchors: {selected}")

        return self.anchors

    # ==========================================================
    # Divergence-ordered anchor selection with split-half noise
    # ==========================================================

    def _compute_pairwise_renyi(self, terminal_log_p, candidates):
        """Compute pairwise Rényi divergences D_α(a || b) for all pairs.

        Args:
            terminal_log_p: dict mapping word -> (log_p, weights).
            candidates: list of candidate terminal words.

        Returns:
            dict mapping (a, b) -> D_α(a || b).
        """
        alpha = self.alpha
        divergences = {}

        for a in candidates:
            if a not in terminal_log_p or a == '<S>':
                continue
            log_p_a, weights_a = terminal_log_p[a]
            a_vid = self.w2i.get(a)
            if a_vid is None:
                continue
            E_a = self.E_sent.get(a, 1e-10)
            log_E_a = math.log(E_a)

            for b in candidates:
                if a == b:
                    continue
                b_vid = self.w2i.get(b)
                if b_vid is None:
                    continue
                E_b = self.E_sent.get(b, 1e-10) if b != '<S>' else 1.0
                log_E_b = math.log(E_b)

                log_ratio = (log_p_a[:, a_vid] - log_p_a[:, b_vid]
                             + log_E_b - log_E_a)

                if alpha == float('inf'):
                    divergences[(a, b)] = float(np.max(log_ratio))
                else:
                    scaled = (alpha - 1) * log_ratio
                    max_s = float(np.max(scaled))
                    lse = max_s + math.log(
                        float(np.sum(weights_a * np.exp(scaled - max_s))))
                    divergences[(a, b)] = lse / (alpha - 1)

        return divergences

    def select_anchors_divergence_ordered(self, max_terminals=300,
                                          max_consecutive_rejects=20,
                                          verbose=True):
        """Select anchors using divergence-ordered greedy furthest-point
        with split-half noise calibration and asymmetry filtering.

        Instead of processing candidates in frequency order (like
        select_anchors_minimal), uses the geometrically motivated
        furthest-point algorithm:

        1. Compute pairwise Rényi divergences from full model
        2. Compute pairwise divergences from two split-half models
        3. Estimate per-pair noise from split-half disagreement
        4. Greedy furthest-point: always pick the candidate with the
           highest noise-corrected minimum divergence from the current
           anchor set
        5. Apply asymmetry filter: reject candidates that look like
           mixtures of already-selected anchors. A mixture w of anchor
           s has D(s||w) small (bounded by log(1/fraction)) while
           D(w||s) is large --- this asymmetric signature distinguishes
           mixtures from genuinely new distributions.
        6. Stop when:
           - corrected distance drops to zero (noise dominates), or
           - max_consecutive_rejects mixture rejections in a row
             (all remaining candidates are mixtures)

        Requires: train_single_model() and train_split_models() first.

        Args:
            max_terminals: consider top-N most frequent terminals.
            max_consecutive_rejects: stop after this many consecutive
                mixture rejections (default 20).
            verbose: print progress trace.

        Returns:
            list of anchor words (excluding S).
        """
        assert self.single_model is not None, "Train single model first"
        assert self.split_model_A is not None, "Train split models first"

        # 1. Select candidate terminals
        terminals = sorted(
            self.vocab, key=lambda w: -self.word_counts.get(w, 0))
        terminals = [w for w in terminals if w in self.w2i][:max_terminals]

        if verbose:
            print(f"Divergence-ordered anchor selection from "
                  f"{len(terminals)} terminals...")

        # 2. Compute log-probs from all three models
        t0 = time.time()
        if verbose:
            print("  Computing log-probs (full model)...")
        log_p_full = self._compute_terminal_log_probs(terminals)

        if verbose:
            print("  Computing log-probs (split model A)...")
        log_p_A = self._compute_terminal_log_probs(
            terminals, model=self.split_model_A,
            sentences=self._split_sentences_A)

        if verbose:
            print("  Computing log-probs (split model B)...")
        log_p_B = self._compute_terminal_log_probs(
            terminals, model=self.split_model_B,
            sentences=self._split_sentences_B)

        candidates = [w for w in terminals
                      if w in log_p_full
                      and w in log_p_A and w in log_p_B]

        if verbose:
            print(f"  {len(candidates)} candidates with all three models "
                  f"({time.time()-t0:.1f}s)")

        # 3. Compute pairwise divergences from all three models
        t0 = time.time()
        if verbose:
            print("  Computing pairwise divergences...")
        div_full = self._compute_pairwise_renyi(log_p_full, candidates)
        div_A = self._compute_pairwise_renyi(log_p_A, candidates)
        div_B = self._compute_pairwise_renyi(log_p_B, candidates)

        if verbose:
            print(f"  {len(div_full)} directed pairs ({time.time()-t0:.1f}s)")

        # 4. Estimate per-pair noise from split-half disagreement
        noise = {}
        for key in div_full:
            d_A = div_A.get(key, 0)
            d_B = div_B.get(key, 0)
            noise[key] = abs(d_A - d_B)

        noise_vals = [v for v in noise.values() if v > 0]
        if noise_vals:
            median_noise = float(np.median(noise_vals))
            p75_noise = float(np.percentile(noise_vals, 75))
        else:
            median_noise = p75_noise = 0.0

        if verbose:
            print(f"  Split-half noise: median={median_noise:.3f}, "
                  f"p75={p75_noise:.3f}")

        # 5. Corrected symmetric distance function
        # Use median noise as a floor: split-half captures data noise
        # but not model noise (random seed initialization). The median
        # noise across all pairs provides a robust floor that accounts
        # for irreducible model-level noise.
        def corrected_distance(a, b):
            """Min-direction divergence minus noise (with floor)."""
            d_ab = div_full.get((a, b), 0)
            d_ba = div_full.get((b, a), 0)
            n_ab = max(noise.get((a, b), p75_noise), median_noise)
            n_ba = max(noise.get((b, a), p75_noise), median_noise)
            return min(d_ab - n_ab, d_ba - n_ba)

        def is_mixture(w, selected):
            """Test whether w looks like a mixture of selected anchors.

            A mixture w of anchor s satisfies:
              D(s||w) << D(w||s)  (strong asymmetry)

            The asymmetry ratio D(w||s)/D(s||w) is:
              ~1 for same-NT pairs (both small, symmetric)
              ~1 for cross-NT pairs (both large, symmetric)
              3-10x for mixtures (forward small, reverse large)

            We require ratio > 3 AND the reverse divergence must be
            clearly above noise (to avoid false positives from pairs
            where both divergences are near zero and the ratio is
            dominated by noise).

            For common anchors, D(s||w) is well-estimated (uses
            anchor's many contexts), so this test is reliable even
            when w is rare.

            For very rare w, the noise on D(w||s) is large, which
            makes the noise-floor condition harder to satisfy -- this
            is conservative (avoids false rejections of genuine
            anchors).
            """
            for s in selected:
                d_sw = div_full.get((s, w), 0)  # anchor -> candidate
                d_ws = div_full.get((w, s), 0)  # candidate -> anchor
                n_sw = noise.get((s, w), p75_noise)
                n_ws = noise.get((w, s), p75_noise)

                # Asymmetry ratio: high for mixtures, ~1 otherwise
                ratio = d_ws / max(d_sw, 1e-6)

                # Noise floor: reverse divergence must be clearly
                # above noise for the ratio to be meaningful
                reverse_reliable = d_ws > max(n_ws, n_sw)

                if ratio > 3.0 and reverse_reliable:
                    return True, s, d_sw, d_ws, n_sw, n_ws
            return False, None, 0, 0, 0, 0

        # 6. Initialize: pick most frequent terminal
        # Genuine anchors have higher individual frequency than
        # mixture terminals whose probability mass is split across NTs.
        # This avoids contaminating the seed with a shared terminal.
        non_s = [w for w in candidates if w != '<S>']
        if not non_s:
            self.anchors = []
            return []

        best_start = max(non_s,
                         key=lambda w: self.word_counts.get(w, 0))

        selected = [best_start]
        remaining = set(non_s) - {best_start}
        rejected = set()
        consecutive_rejects = 0

        if verbose:
            freq = self.word_counts.get(best_start, 0)
            print(f"\n  Init: {best_start:>10} (freq={freq:>7})")

        # 7. Greedy furthest-point with asymmetry filter
        while remaining:
            best_w = None
            best_min_d = -float('inf')

            for w in remaining:
                if w in rejected:
                    continue
                min_d = min(corrected_distance(w, s) for s in selected)
                if min_d > best_min_d:
                    best_min_d = min_d
                    best_w = w

            # Distance-based stopping: noise dominates
            if best_w is None or best_min_d <= 0:
                if verbose:
                    print(f"  Stop: best corrected distance = "
                          f"{best_min_d:.3f} <= 0")
                break

            # Asymmetry filter: reject mixtures
            mix, mix_anchor, d_sw, d_ws, n_sw, n_ws = is_mixture(
                best_w, selected)

            if mix:
                rejected.add(best_w)
                remaining.discard(best_w)
                consecutive_rejects += 1

                if verbose:
                    freq = self.word_counts.get(best_w, 0)
                    ratio = d_ws / max(d_sw, 1e-6)
                    print(f"  Reject: {best_w:>10} (freq={freq:>7}, "
                          f"mixture of {mix_anchor}, "
                          f"D(s||w)={d_sw:.3f}, D(w||s)={d_ws:.3f}, "
                          f"ratio={ratio:.1f}x)")

                if consecutive_rejects >= max_consecutive_rejects:
                    if verbose:
                        print(f"  Stop: {max_consecutive_rejects} "
                              f"consecutive mixture rejections")
                    break
                continue

            # Accept this candidate as a new anchor
            selected.append(best_w)
            remaining.discard(best_w)
            consecutive_rejects = 0

            if verbose:
                freq = self.word_counts.get(best_w, 0)
                # Show raw and corrected for diagnostics
                raw_min = min(
                    min(div_full.get((best_w, s), 0),
                        div_full.get((s, best_w), 0))
                    for s in selected[:-1])
                print(f"  Add:  {best_w:>10} (freq={freq:>7}, "
                      f"raw={raw_min:.3f}, corrected={best_min_d:.3f}, "
                      f"n={len(selected)})")

        # 8. Retroactive mixture check
        # The seed was selected without any reference set for mixture
        # detection.  Now that we have the full anchor set, check
        # every anchor against all OTHERS.  This catches cases where
        # the seed (most frequent terminal) is itself a shared word.
        pruned = list(selected)
        retroactive_rejected = []
        changed = True
        while changed:
            changed = False
            for i in range(len(pruned)):
                others = pruned[:i] + pruned[i+1:]
                if not others:
                    continue
                mix, mix_anchor, d_sw, d_ws, n_sw, n_ws = is_mixture(
                    pruned[i], others)
                if mix:
                    if verbose:
                        ratio = d_ws / max(d_sw, 1e-6)
                        print(f"  Retroactive reject: {pruned[i]:>10} "
                              f"(mixture of {mix_anchor}, "
                              f"D(s||w)={d_sw:.3f}, "
                              f"D(w||s)={d_ws:.3f}, "
                              f"ratio={ratio:.1f}x)")
                    retroactive_rejected.append(pruned[i])
                    pruned.pop(i)
                    changed = True
                    break

        selected = pruned
        rejected = rejected | set(retroactive_rejected)

        # 9. Store results
        self.anchors = selected
        self.nonterminals = ['S'] + [f'NT_{w}' for w in selected]
        self.anchor2nt = {w: f'NT_{w}' for w in selected}

        if verbose:
            print(f"\nSelected {len(selected)} anchors: {selected}")
            if rejected:
                print(f"Rejected {len(rejected)} mixtures: "
                      f"{sorted(rejected)}")

        return self.anchors

    # ==========================================================
    # Step 4: Estimate xi parameters
    # ==========================================================

    def _is_rnn_model(self):
        """Check if the single model is an RNN model."""
        return isinstance(self.single_model, RNNClozeModel)

    def _collect_anchor_contexts(self):
        """Collect contexts for anchors (one corpus pass, cached).

        For RNN: stores occurrence positions per anchor.
        For fixed-width: stores context Counter per anchor.
        """
        anchor_set = set(self.anchors)
        if self._is_rnn_model():
            self._anchor_positions = self._collect_positions(anchor_set)
            self._anchor_ctxs = None
            self._rnn_sampled_cache = {}
        else:
            self._anchor_ctxs = self._collect_fixed_contexts(anchor_set)

    def _ensure_anchor_contexts(self):
        """Ensure anchor contexts have been collected (lazy init)."""
        if self._is_rnn_model():
            if not hasattr(self, '_anchor_positions'):
                self._collect_anchor_contexts()
        else:
            if not hasattr(self, '_anchor_ctxs'):
                self._collect_anchor_contexts()

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

        For RNN: samples n_context_samples positions (cached).
        For fixed-width: uses frequency-filtered contexts.

        Returns:
            (log_p, weights): log_p is (n_ctx, vocab) numpy array,
                              weights is (n_ctx,) numpy array.
        """
        if self._is_rnn_model():
            if anchor not in self._rnn_sampled_cache:
                rng = np.random.RandomState(
                    self.seed + hash(anchor) % (2**31))
                result = self._rnn_log_probs_from_positions(
                    self._anchor_positions[anchor],
                    n_samples=self.n_context_samples, rng=rng)
                if result is None:
                    raise ValueError(f"No positions for anchor {anchor}")
                self._rnn_sampled_cache[anchor] = result
            return self._rnn_sampled_cache[anchor]
        else:
            result = self._fixed_log_probs_from_contexts(
                self._anchor_ctxs[anchor])
            if result is None:
                raise ValueError(f"No contexts for anchor {anchor}")
            return result

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

        self._ensure_anchor_contexts()

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
        """Prepare context representations for binary xi estimation.

        For fixed-width models returns:
            (ctx_tensor, weights, n_ctx, gap_corrections):
            - ctx_tensor: (n_ctx, 2*k) long tensor of context word indices
            - weights: normalised context weights
            - n_ctx: number of contexts
            - gap_corrections: per-context log(P(1-gap)/P(2-gap)) or None.

        For RNN models returns:
            (ctx_data, weights, n_ctx, None):
            - ctx_data: list of (left_tensor, right_tensor) pairs
            - weights: uniform normalised weights
            - n_ctx: number of sampled contexts
        """
        if self._is_rnn_model():
            # Ensure sampled positions are cached (same as _compute_full_log_probs)
            log_p, weights = self._compute_full_log_probs(anchor)
            # Rebuild tensor pairs from cached positions
            rng = np.random.RandomState(
                self.seed + hash(anchor) % (2**31))
            positions = self._anchor_positions[anchor]
            N = self.n_context_samples
            if len(positions) > N:
                idx = rng.choice(len(positions), N, replace=False)
                positions = [positions[i] for i in idx]
            ctx_data = []
            for si, wi in positions:
                sent = self.sentences[si]
                left = [BOUNDARY] + list(sent[:wi])
                right = list(sent[wi + 1:]) + [BOUNDARY]
                left_t = torch.tensor(
                    [self.w2i.get(w, 0) for w in left],
                    dtype=torch.long).unsqueeze(0)
                right_t = torch.tensor(
                    [self.w2i.get(w, 0) for w in right],
                    dtype=torch.long).unsqueeze(0)
                ctx_data.append((left_t, right_t))
            return ctx_data, weights, len(positions), None

        # Fixed-width: build context tensor with gap corrections
        ctxs = self._anchor_ctxs[anchor]
        freq = {c: n for c, n in ctxs.items()
                if n >= self.min_context_count}
        if len(freq) < 20:
            freq = dict(ctxs.most_common(100))

        ctx_list = list(freq.keys())
        counts = np.array([freq[c] for c in ctx_list], dtype=np.float64)
        weights = counts / counts.sum()
        n_ctx = len(ctx_list)

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

        self._ensure_anchor_contexts()
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
