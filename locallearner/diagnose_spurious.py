#!/usr/bin/env python3
"""Diagnose the source of noise for spurious NMF candidates.

For each grammar, with the correct number of kernels:
  1. Compute true P(NT | terminal) from the grammar.
  2. Run NMF with correct kernel count to get sampled kernel distributions.
  3. For the next candidate (first spurious) and a few true kernels:
     - Predict its feature vector using true posteriors + sampled kernel basis.
     - Compare predicted vs actual sampled feature vector.
     - Compare FW decomposition weights vs true posteriors.

This distinguishes:
  a) Sampling noise: predicted ≈ actual, FW weights ≈ true posteriors.
  b) Model mismatch: predicted ≠ actual even with perfect posteriors.

Usage:
    python3 diagnose_spurious.py [--base DIR] [--grammars 0,1,2]
"""

import sys, os, json, argparse, time
import numpy as np
import math

sys.path.insert(0, os.path.dirname(__file__))
SYNTHETICPCFG_PATH = os.path.join(os.path.dirname(__file__),
    '..', '..', '..', 'cursor_syntheticpcfg', 'syntheticpcfg')
sys.path.insert(0, SYNTHETICPCFG_PATH)

import locallearner as ll_module
import ngram_counts
from syntheticpcfg import pcfg as spcfg
from collections import defaultdict

DEFAULT_BASE = os.path.join(os.path.dirname(__file__), '..', 'tmp', 'jm_experiment')


def compute_true_posteriors(grammar):
    """Compute P(NT | terminal) for all terminals in the grammar.

    Returns:
        dict mapping terminal -> dict mapping NT -> posterior probability.
    """
    pe = grammar.production_expectations()
    te = grammar.terminal_expectations()

    posteriors = defaultdict(dict)
    for prod, e in pe.items():
        if len(prod) == 2:
            nt, word = prod
            posteriors[word][nt] = e / te[word]
    return dict(posteriors)


def load_grammar_spcfg(grammar_path):
    """Load a grammar as a syntheticpcfg PCFG object."""
    return spcfg.load_pcfg_from_file(grammar_path)


def run_diagnosis(corpus_path, grammar_path, n_clusters=10, seed=42,
                  min_count=1000, cluster_file=None,
                  bigram_file=None, trigram_file=None):
    """Run NMF with correct kernel count and diagnose spurious candidates."""

    # Load grammar and compute true posteriors
    grammar = load_grammar_spcfg(grammar_path)
    n_nt = len(grammar.nonterminals)
    true_posteriors = compute_true_posteriors(grammar)
    nte = grammar.nonterminal_expectations()

    # Run NMF with the true number of NTs
    learner = ll_module.LocalLearner(corpus_path)
    learner.number_clusters = n_clusters
    learner.min_count_nmf = min_count
    learner.seed = seed
    learner.cluster_file = cluster_file
    learner.bigram_file = bigram_file
    learner.trigram_file = trigram_file
    learner.nonterminals = n_nt
    learner.feature_mode = 'marginal'

    kernels = learner.find_kernels(verbose=False)

    nmf_obj = learner.nmf

    # Map kernel names to grammar NTs via best posterior
    kernel_to_nt = {}
    nt_list = sorted(grammar.nonterminals)
    for k_word in kernels[1:]:  # skip S
        post = true_posteriors.get(k_word, {})
        if post:
            best_nt = max(post, key=post.get)
            kernel_to_nt[k_word] = best_nt

    # Build kernel index mapping: kernel_idx -> kernel_word
    # bases[0] is start symbol, bases[1:] are the real kernels
    bases = nmf_obj.bases
    kernel_words = [nmf_obj.index[bi] for bi in bases]

    print(f"\nKernels ({n_nt} NTs):")
    print(f"  {'Kernel':>10s}  {'Count':>8s}  {'BestNT':>8s}  {'P(NT|w)':>8s}  {'NTexpect':>10s}")
    for ki, bi in enumerate(bases):
        w = nmf_obj.index[bi]
        post = true_posteriors.get(w, {})
        best_nt = kernel_to_nt.get(w, 'S' if ki == 0 else '?')
        best_p = post.get(best_nt, 0.0) if best_nt != 'S' else 1.0
        nt_e = nte.get(best_nt, 0.0) if best_nt != 'S' else 1.0
        cnt = int(nmf_obj.counts[bi])
        print(f"  {w:>10s}  {cnt:8d}  {best_nt:>8s}  {best_p:8.4f}  {nt_e:10.4f}")

    # Now find the next candidates (spurious) using find_but_dont_add
    # We need to re-initialise FW and find the next furthest word
    nmf_obj.initialise_frank_wolfe()

    # Get distances for all words
    all_candidates = []
    for i in range(nmf_obj.n):
        if i in nmf_obj.excluded:
            continue
        y = nmf_obj.data[i, :]
        x, d = nmf_obj.estimate_frank_wolfe(y)
        w = nmf_obj.index[i]
        cnt = int(nmf_obj.counts[i])
        all_candidates.append((d, i, w, cnt, x))

    # Sort by distance descending (furthest first)
    all_candidates.sort(key=lambda t: -t[0])

    # Analyse top spurious candidates and a few true kernels
    print(f"\n{'='*90}")
    print(f"Diagnosis: predicted vs actual for top candidates")
    print(f"{'='*90}")

    # For reference: ordered list of NTs matching kernel order
    ordered_nts = ['S'] + [kernel_to_nt.get(w, '?') for w in kernels[1:]]

    header = (f"  {'Word':>10s} {'Count':>7s} {'FWdist':>8s} "
              f"{'L2err':>8s} {'KLpred':>8s} {'BestNT':>8s} "
              f"{'P(best)':>8s}")
    print(header)
    print(f"  {'-'*80}")

    results = []
    for rank, (d, idx, word, count, fw_x) in enumerate(all_candidates[:15]):
        # Actual sampled distribution (shrinkage-normalized)
        actual = nmf_obj.data[idx, :]
        # Actual raw (no shrinkage)
        actual_raw = nmf_obj.raw_data[idx, :]

        # True posteriors for this word
        post = true_posteriors.get(word, {})
        best_nt = max(post, key=post.get) if post else '?'
        best_p = post.get(best_nt, 0.0)

        # Build true posterior weight vector aligned to kernel order
        true_x = np.zeros(len(bases))
        for ki, bi in enumerate(bases):
            k_word = nmf_obj.index[bi]
            k_nt = kernel_to_nt.get(k_word, 'S' if ki == 0 else None)
            if k_nt and k_nt in post:
                true_x[ki] = post[k_nt]
        # Normalise (should already sum to ~1 if all NTs covered)
        tx_sum = true_x.sum()
        if tx_sum > 0:
            true_x = true_x / tx_sum

        # Predicted distribution = true_x @ kernel basis vectors (shrinkage)
        basis_matrix = np.array([nmf_obj.data[bi, :] for bi in bases])
        predicted = true_x @ basis_matrix
        predicted = np.maximum(predicted, 0)
        p_sum = predicted.sum()
        if p_sum > 0:
            predicted = predicted / p_sum

        # Also predicted from raw
        basis_raw = np.array([nmf_obj.raw_data[bi, :] for bi in bases])
        predicted_raw = true_x @ basis_raw
        predicted_raw = np.maximum(predicted_raw, 0)
        pr_sum = predicted_raw.sum()
        if pr_sum > 0:
            predicted_raw = predicted_raw / pr_sum

        # FW predicted distribution
        fw_predicted = fw_x @ basis_matrix
        fw_predicted = np.maximum(fw_predicted, 0)
        fw_sum = fw_predicted.sum()
        if fw_sum > 0:
            fw_predicted = fw_predicted / fw_sum

        # Errors
        l2_true = np.linalg.norm(actual - predicted)
        l2_fw = np.linalg.norm(actual - fw_predicted)

        # KL divergence: actual || predicted (only where actual > 0)
        kl_true = 0.0
        for j in range(len(actual)):
            if actual[j] > 0 and predicted[j] > 0:
                kl_true += actual[j] * math.log(actual[j] / predicted[j])

        print(f"  {word:>10s} {count:7d} {d:8.5f} "
              f"{l2_true:8.5f} {kl_true:8.5f} {best_nt:>8s} "
              f"{best_p:8.4f}")

        results.append({
            'rank': rank,
            'word': word,
            'count': count,
            'distance': d,
            'best_nt': best_nt,
            'best_posterior': best_p,
            'l2_true_pred': l2_true,
            'l2_fw_pred': l2_fw,
            'kl_true_pred': kl_true,
            'fw_weights': fw_x.tolist(),
            'true_weights': true_x.tolist(),
        })

    # Detailed comparison for top 3 candidates
    print(f"\n{'='*90}")
    print(f"Detailed weight comparison for top 3 spurious candidates")
    print(f"{'='*90}")

    for rank, (d, idx, word, count, fw_x) in enumerate(all_candidates[:3]):
        post = true_posteriors.get(word, {})
        best_nt = max(post, key=post.get) if post else '?'

        true_x = np.zeros(len(bases))
        for ki, bi in enumerate(bases):
            k_word = nmf_obj.index[bi]
            k_nt = kernel_to_nt.get(k_word, 'S' if ki == 0 else None)
            if k_nt and k_nt in post:
                true_x[ki] = post[k_nt]
        tx_sum = true_x.sum()
        if tx_sum > 0:
            true_x = true_x / tx_sum

        print(f"\n  Candidate '{word}' (rank {rank}, count={count}, "
              f"d={d:.6f}, best_nt={best_nt}):")
        print(f"  {'Kernel':>10s} {'NT':>8s} {'TrueW':>8s} {'FW_W':>8s} "
              f"{'Diff':>8s}")
        print(f"  {'-'*50}")
        for ki, bi in enumerate(bases):
            k_word = nmf_obj.index[bi]
            k_nt = kernel_to_nt.get(k_word, 'S' if ki == 0 else '?')
            tw = true_x[ki]
            fw = fw_x[ki]
            diff = fw - tw
            if tw > 0.001 or fw > 0.001:
                print(f"  {k_word:>10s} {k_nt:>8s} {tw:8.4f} {fw:8.4f} "
                      f"{diff:+8.4f}")
        l2_weights = np.linalg.norm(fw_x - true_x)
        print(f"  L2(FW - true weights) = {l2_weights:.6f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Diagnose spurious NMF candidates')
    parser.add_argument('--base', default=DEFAULT_BASE)
    parser.add_argument('--clusters', type=int, default=10)
    parser.add_argument('--grammars', default=None,
                        help='Comma-separated grammar indices (default: first 5)')
    args = parser.parse_args()

    base = os.path.abspath(args.base)
    manifest_path = os.path.join(base, 'manifest.json')
    with open(manifest_path) as f:
        manifest = json.load(f)

    grammars = [g for g in manifest['grammars'] if 'error' not in g]
    if args.grammars is not None:
        indices = set(int(x) for x in args.grammars.split(','))
        grammars = [g for g in grammars if g['index'] in indices]
    else:
        grammars = grammars[:5]

    n_sentences = manifest.get('n_sentences', '?')
    min_count = max(5, n_sentences // 1000) if isinstance(n_sentences, int) else 1000

    for ginfo in grammars:
        corpus_path = os.path.join(base, ginfo['corpus_path'])
        grammar_path = os.path.join(base, ginfo['grammar_path'])
        gdir = os.path.dirname(corpus_path)
        cluster_file = os.path.join(gdir, f'clusters_c{args.clusters}_s42.clusters')
        bigram_file = os.path.join(gdir, 'bigrams.gz')
        trigram_file = os.path.join(gdir, 'trigrams.gz')

        n_nt = ginfo['n_nonterminals']
        print(f"\n{'#'*90}")
        print(f"# g{ginfo['index']:03d}: {n_nt} NTs, N={ginfo['n_sentences']}")
        print(f"{'#'*90}")

        run_diagnosis(
            corpus_path, grammar_path,
            n_clusters=args.clusters, seed=42,
            min_count=min_count,
            cluster_file=cluster_file,
            bigram_file=bigram_file,
            trigram_file=trigram_file)


if __name__ == '__main__':
    main()
