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
    build_vocab, train_cloze_model, train_pair_cloze_model,
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

        # Cache paths (set to enable caching)
        self.cluster_file = None
        self.bigram_file = None
        self.single_model_file = None
        self.pair_model_file = None
        self.kernels_file = None

        # State (populated during learning)
        self.clusters = None          # Ney-Essen word -> cluster mapping
        self.candidate_kernels = None # overshot kernel list from NMF
        self.single_model = None      # trained ClozeModel
        self.pair_model = None        # trained PairClozeModel
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
            print(f"Training single cloze model (k={self.k}, "
                  f"{self.n_epochs} epochs)...")

        model, w2i, i2w, history = train_cloze_model(
            self.sentences, k=self.k,
            embedding_dim=self.embedding_dim, hidden_dim=self.hidden_dim,
            n_epochs=self.n_epochs, batch_size=self.batch_size,
            lr=1e-3, min_count=1, seed=self.seed, verbose=verbose,
        )
        self.single_model = model
        self.w2i = w2i
        self.i2w = i2w

        if self.single_model_file:
            save_model(model, w2i, i2w, self.single_model_file,
                       k=self.k, history=history)

    def train_pair_model(self, verbose=True):
        """Train pair cloze model P(w1, w2 | context)."""
        if self.pair_model_file and os.path.exists(self.pair_model_file):
            result = load_model(self.pair_model_file, device='cpu')
            self.pair_model = result[0]
            if verbose:
                print(f"Loaded pair model from {self.pair_model_file}")
            return

        if verbose:
            print(f"Training pair cloze model (k={self.k}, "
                  f"{self.n_epochs} epochs)...")

        model, w2i, i2w, history = train_pair_cloze_model(
            self.sentences, k=self.k,
            embedding_dim=self.embedding_dim, hidden_dim=self.hidden_dim,
            n_epochs=self.n_epochs, batch_size=self.batch_size,
            lr=1e-3, min_count=1, seed=self.seed, verbose=verbose,
        )
        self.pair_model = model

        # Ensure vocab is consistent
        if self.w2i is None:
            self.w2i = w2i
            self.i2w = i2w

        if self.pair_model_file:
            save_model(model, w2i, i2w, self.pair_model_file,
                       k=self.k, history=history)

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
    # Step 4: Estimate xi parameters
    # ==========================================================

    def _collect_anchor_contexts(self):
        """Collect single-word contexts for each anchor, with counts."""
        k = self.k
        self._anchor_ctxs = {}
        for a in self.anchors:
            self._anchor_ctxs[a] = Counter()

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

        k = self.k
        self.lexical_xi = {}

        for anchor in self.anchors:
            nt = self.anchor2nt[anchor]
            a_vid = self.w2i[anchor]
            log_E_a = math.log(self.E_sent[anchor])

            freq = self._get_filtered_contexts(anchor)
            ctx_list = list(freq.keys())
            counts = np.array([freq[c] for c in ctx_list], dtype=np.float64)
            weights = counts / counts.sum()
            n_ctx = len(ctx_list)

            ctx_tensor = torch.zeros(n_ctx, 2 * k, dtype=torch.long)
            for ci, ctx in enumerate(ctx_list):
                for j, w in enumerate(ctx):
                    ctx_tensor[ci, j] = self.w2i.get(w, 0)

            with torch.no_grad():
                logits = self.single_model(ctx_tensor)
                log_p = torch.log_softmax(logits, dim=-1).numpy()

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

    def estimate_binary_xi(self, verbose=True):
        """Estimate xi(A -> BC) for all triples of NTs.

        For non-S parents:
          xi(A -> BC) = E(bc)/(E(b)*E(c)) * exp(-D_alpha(anchor_A || bc))
          where D uses single model for P(a|l,r) and pair model for P(bc|l,r)

        For S:
          xi(S -> BC) = E_length2(bc) / (E(b) * E(c))

        Returns:
            dict mapping (nt_A, nt_B, nt_C) -> xi value
        """
        assert self.single_model is not None
        assert self.pair_model is not None
        assert self.anchors is not None

        if verbose:
            print("Estimating binary xi parameters...")

        if not hasattr(self, '_anchor_ctxs'):
            self._collect_anchor_contexts()

        k = self.k
        self.binary_xi = {}

        # Non-S parents: use divergence
        for anchor_A in self.anchors:
            nt_A = self.anchor2nt[anchor_A]
            a_vid = self.w2i[anchor_A]
            log_E_a = math.log(self.E_sent[anchor_A])

            freq = self._get_filtered_contexts(anchor_A)
            ctx_list = list(freq.keys())
            counts = np.array([freq[c] for c in ctx_list], dtype=np.float64)
            weights = counts / counts.sum()
            n_ctx = len(ctx_list)

            ctx_tensor = torch.zeros(n_ctx, 2 * k, dtype=torch.long)
            for ci, ctx in enumerate(ctx_list):
                for j, w in enumerate(ctx):
                    ctx_tensor[ci, j] = self.w2i.get(w, 0)

            # Single model: log P(a | ctx) for all contexts
            with torch.no_grad():
                logits_s = self.single_model(ctx_tensor)
                log_p_a = torch.log_softmax(
                    logits_s, dim=-1)[:, a_vid].numpy()

            for anchor_B in self.anchors:
                nt_B = self.anchor2nt[anchor_B]
                b_vid = self.w2i[anchor_B]

                for anchor_C in self.anchors:
                    nt_C = self.anchor2nt[anchor_C]
                    c_vid = self.w2i[anchor_C]

                    e_bc = self.E_bigram.get(
                        (anchor_B, anchor_C), 0)
                    if e_bc == 0:
                        self.binary_xi[(nt_A, nt_B, nt_C)] = 0
                        continue

                    log_E_bc = math.log(e_bc)
                    pmi = (log_E_bc - math.log(self.E_sent[anchor_B])
                           - math.log(self.E_sent[anchor_C]))

                    # Pair model: log P(b, c | ctx)
                    with torch.no_grad():
                        logits1 = self.pair_model.forward(ctx_tensor)
                        log_p_b = torch.log_softmax(
                            logits1, dim=-1)[:, b_vid].numpy()
                        b_tensor = torch.full(
                            (n_ctx,), b_vid, dtype=torch.long)
                        _, logits2 = self.pair_model.forward(
                            ctx_tensor, b_tensor)
                        log_p_c = torch.log_softmax(
                            logits2, dim=-1)[:, c_vid].numpy()

                    log_p_bc = log_p_b + log_p_c
                    log_ratio = ((log_p_a - log_E_a)
                                 - (log_p_bc - log_E_bc))

                    alpha = self.alpha
                    scaled = (alpha - 1) * log_ratio
                    max_s = np.max(scaled)
                    lse = max_s + np.log(
                        np.sum(weights * np.exp(scaled - max_s)))
                    d = lse / (alpha - 1)

                    xi = math.exp(pmi - d)
                    self.binary_xi[(nt_A, nt_B, nt_C)] = xi

        # S rules: from length-2 sentence counts
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
