#!/usr/bin/env python3
"""Controlled experiment: fixed binary rules, varying terminal count.

Generates a base 5-NT grammar, then creates variants with different numbers
of terminals (keeping binary rules identical).  For each variant, trains
RNN cloze models, computes pairwise divergences, and records corrected
distances labeled by same-NT vs cross-NT.

This gives us the empirical distribution of corrected distances as a
function of vocabulary size, which informs the stopping threshold.

Usage:
    python3 experiment_terminal_scaling.py output.json
    python3 experiment_terminal_scaling.py output.json --terminal_counts 50 100 300 1000
"""

import sys
import os
import json
import argparse
import tempfile
import time
import math
import random as pyrandom
from collections import Counter

import numpy as np

# Add packages to path
sys.path.insert(0, os.path.dirname(__file__))
SYNTHETICPCFG_PATH = os.path.join(os.path.dirname(__file__),
    '..', '..', '..', 'cursor_syntheticpcfg', 'syntheticpcfg')
sys.path.insert(0, SYNTHETICPCFG_PATH)

from syntheticpcfg import pcfgfactory, pcfg as spcfg, utility as sutil
import wcfg


def generate_base_grammar(n_nonterminals, seed):
    """Generate a base PCFG and return it along with its binary rules.

    Returns:
        grammar: the full PCFG
        binary_rules: dict of {(A, B, C): prob} for binary rules
        unary_totals: dict of {A: total_lexical_prob} per nonterminal
    """
    pyrandom.seed(seed)
    np.random.seed(seed)

    # Generate with a small number of terminals (will be replaced)
    factory = pcfgfactory.FullPCFGFactory(
        nonterminals=n_nonterminals,
        terminals=100  # placeholder
    )
    factory.lexical_distribution = pcfgfactory.LogNormalPrior(sigma=4.0)
    alpha = 1.0 / ((n_nonterminals - 1) ** 2 + 1)
    factory.binary_distribution = pcfgfactory.LexicalDirichlet(dirichlet=alpha)
    factory.length_distribution = pcfgfactory.LengthDistribution()
    factory.length_distribution.set_poisson(5.0, 20)

    grammar = factory.sample_uniform()

    # Extract binary rules
    binary_rules = {}
    unary_totals = {}
    for prod, prob in grammar.parameters.items():
        if len(prod) == 3:
            binary_rules[prod] = prob
        elif len(prod) == 2:
            nt = prod[0]
            unary_totals[nt] = unary_totals.get(nt, 0.0) + prob

    return grammar, binary_rules, unary_totals


def make_grammar_with_terminals(binary_rules, unary_totals, nonterminals,
                                start, n_terminals, seed):
    """Create a new PCFG with the same binary rules but new lexical rules.

    Lexical rules are sampled from a LogNormal prior with sigma=4.0,
    matching the original factory settings.
    """
    np.random.seed(seed)

    terminals = [f't{i:04d}' for i in range(n_terminals)]

    # Build the PCFG
    grammar = spcfg.PCFG()
    grammar.start = start
    grammar.nonterminals = set(nonterminals)
    grammar.terminals = set(terminals)

    # Copy binary rules
    for prod, prob in binary_rules.items():
        grammar.productions.append(prod)
        grammar.parameters[prod] = prob

    # Sample lexical rules with LogNormal prior
    prior = pcfgfactory.LogNormalPrior(sigma=4.0)
    for nt in sorted(nonterminals):
        total = unary_totals.get(nt, 0.0)
        if total <= 0:
            continue

        # Sample proportions from the prior
        raw = prior.sample(n_terminals)
        raw_sum = sum(raw)
        for i, t in enumerate(terminals):
            prod = (nt, t)
            grammar.productions.append(prod)
            grammar.parameters[prod] = total * raw[i] / raw_sum

    grammar.normalise()
    grammar.set_log_parameters()
    return grammar


def sample_corpus(grammar, n_sentences, seed):
    """Sample a corpus from a syntheticpcfg PCFG."""
    rng = np.random.RandomState(seed)
    sampler = spcfg.Sampler(grammar, random=rng)
    sentences = []
    for _ in range(n_sentences):
        try:
            tree = sampler.sample_tree()
            s = sutil.collect_yield(tree)
            sentences.append(' '.join(s))
        except ValueError:
            continue
    return sentences


def get_word_nt_map(grammar):
    """Build a map from terminal -> best nonterminal (highest P(NT|w))."""
    te = {}  # terminal expectations E[w]
    pe = {}  # production expectations E[A -> w]
    for prod, prob in grammar.parameters.items():
        if len(prod) == 2:
            nt, w = prod
            pe[(nt, w)] = pe.get((nt, w), 0.0) + prob
            te[w] = te.get(w, 0.0) + prob

    word_nt = {}
    word_posteriors = {}
    for w in grammar.terminals:
        if w not in te or te[w] == 0:
            continue
        posteriors = {}
        for nt in grammar.nonterminals:
            p = pe.get((nt, w), 0.0) / te[w]
            if p > 0:
                posteriors[nt] = p
        if posteriors:
            best_nt = max(posteriors, key=posteriors.get)
            word_nt[w] = best_nt
            word_posteriors[w] = posteriors[best_nt]

    return word_nt, word_posteriors


def run_divergence_analysis(corpus_path, grammar, max_terminals=300,
                            alpha=2.0, epochs=20, verbose=True,
                            model_dir=None):
    """Train models (or load from cache), compute divergences, classify pairs.

    Args:
        model_dir: if provided, save/load trained models to this directory.
            Models are reused if they already exist.

    Returns a dict with full pair-level data including both directions
    of the Rényi divergence.
    """
    from neural_learner import NeuralLearner

    nl = NeuralLearner(corpus_path)
    nl.alpha = alpha
    nl.model_type = 'rnn'
    nl.n_epochs = epochs

    # Set up model caching
    if model_dir is not None:
        os.makedirs(model_dir, exist_ok=True)
        nl.single_model_file = os.path.join(model_dir, 'single.pt')
        nl.split_model_A_file = os.path.join(model_dir, 'split_A.pt')
        nl.split_model_B_file = os.path.join(model_dir, 'split_B.pt')

    t0 = time.time()
    nl.train_single_model(verbose=verbose)
    nl.train_split_models(verbose=verbose)

    # Get word -> NT map from grammar
    word_nt, word_posteriors = get_word_nt_map(grammar)

    # Select candidate terminals
    terminals = sorted(nl.vocab, key=lambda w: -nl.word_counts.get(w, 0))
    terminals = [w for w in terminals if w in nl.w2i][:max_terminals]

    if verbose:
        print(f"\n  Divergence analysis: {len(terminals)} candidates")

    # Compute log-probs
    log_p_full = nl._compute_terminal_log_probs(terminals)
    log_p_A = nl._compute_terminal_log_probs(
        terminals, model=nl.split_model_A,
        sentences=nl._split_sentences_A)
    log_p_B = nl._compute_terminal_log_probs(
        terminals, model=nl.split_model_B,
        sentences=nl._split_sentences_B)

    candidates = [w for w in terminals
                  if w in log_p_full and w in log_p_A and w in log_p_B]

    # Compute pairwise divergences
    div_full = nl._compute_pairwise_renyi(log_p_full, candidates)
    div_A = nl._compute_pairwise_renyi(log_p_A, candidates)
    div_B = nl._compute_pairwise_renyi(log_p_B, candidates)

    # Noise estimates
    noise = {}
    for key in div_full:
        noise[key] = abs(div_A.get(key, 0) - div_B.get(key, 0))

    # Fit noise model
    freq_noise_data = []
    for (a, b), n_val in noise.items():
        fa = nl.word_counts.get(a, 0)
        fb = nl.word_counts.get(b, 0)
        min_f = min(fa, fb)
        if min_f > 0:
            freq_noise_data.append((min_f, n_val))

    if len(freq_noise_data) > 10:
        fn_freqs = np.array([d[0] for d in freq_noise_data], dtype=float)
        fn_noises = np.array([d[1] for d in freq_noise_data])
        inv_sqrt = 1.0 / np.sqrt(fn_freqs)
        A_mat = np.column_stack([np.ones_like(inv_sqrt), inv_sqrt])
        coeffs, _, _, _ = np.linalg.lstsq(A_mat, fn_noises, rcond=None)
        noise_floor = max(coeffs[0], 0.0)
        noise_coeff = max(coeffs[1], 0.0)
    else:
        noise_floor = float(np.median([v for v in noise.values() if v > 0]))
        noise_coeff = 0.0

    def noise_model(fa, fb):
        min_f = max(min(fa, fb), 1)
        return noise_floor + noise_coeff / math.sqrt(min_f)

    if verbose:
        print(f"  Noise model: {noise_floor:.3f} + "
              f"{noise_coeff:.3f} / sqrt(min_freq)")

    # Classify all pairs — keep BOTH directions of the divergence
    same_nt_corrected = []
    cross_nt_corrected = []
    same_nt_raw = []
    cross_nt_raw = []
    all_pairs = []  # full detail for every pair

    for i, a in enumerate(candidates):
        for j, b in enumerate(candidates):
            if i >= j:
                continue
            nt_a = word_nt.get(a)
            nt_b = word_nt.get(b)
            if nt_a is None or nt_b is None:
                continue

            # Both directions of the divergence (NOT symmetric)
            d_ab = div_full.get((a, b), 0)
            d_ba = div_full.get((b, a), 0)

            # Noise estimates for both directions
            fa = nl.word_counts.get(a, 0)
            fb = nl.word_counts.get(b, 0)
            floor = noise_model(fa, fb)
            n_ab = max(noise.get((a, b), floor), floor)
            n_ba = max(noise.get((b, a), floor), floor)

            # Corrected divergences (both directions)
            c_ab = d_ab - n_ab
            c_ba = d_ba - n_ba
            corrected_min = min(c_ab, c_ba)

            detail = {
                'a': a, 'b': b,
                'nt_a': nt_a, 'nt_b': nt_b,
                'freq_a': fa, 'freq_b': fb,
                'd_ab': float(d_ab), 'd_ba': float(d_ba),
                'n_ab': float(n_ab), 'n_ba': float(n_ba),
                'c_ab': float(c_ab), 'c_ba': float(c_ba),
                'corrected_min': float(corrected_min),
                'posterior_a': float(word_posteriors.get(a, 0)),
                'posterior_b': float(word_posteriors.get(b, 0)),
                'same_nt': nt_a == nt_b,
            }
            all_pairs.append(detail)

            if nt_a == nt_b:
                same_nt_corrected.append(corrected_min)
                same_nt_raw.append(min(d_ab, d_ba))
            else:
                cross_nt_corrected.append(corrected_min)
                cross_nt_raw.append(min(d_ab, d_ba))

    elapsed = time.time() - t0

    same_pairs = [p for p in all_pairs if p['same_nt']]
    cross_pairs = [p for p in all_pairs if not p['same_nt']]

    if verbose:
        print(f"  Same-NT pairs: {len(same_pairs)}, "
              f"Cross-NT pairs: {len(cross_pairs)}")
        if same_pairs:
            sc = np.array([p['corrected_min'] for p in same_pairs])
            print(f"  Same-NT corrected_min: "
                  f"median={np.median(sc):.3f}, "
                  f"mean={np.mean(sc):.3f}, "
                  f"std={np.std(sc):.3f}, "
                  f"max={np.max(sc):.3f}, "
                  f"p95={np.percentile(sc, 95):.3f}, "
                  f"p99={np.percentile(sc, 99):.3f}")
            # Also show both directions separately
            c_ab = np.array([p['c_ab'] for p in same_pairs])
            c_ba = np.array([p['c_ba'] for p in same_pairs])
            print(f"  Same-NT c_ab: "
                  f"median={np.median(c_ab):.3f}, p99={np.percentile(c_ab, 99):.3f}")
            print(f"  Same-NT c_ba: "
                  f"median={np.median(c_ba):.3f}, p99={np.percentile(c_ba, 99):.3f}")
        if cross_pairs:
            cc = np.array([p['corrected_min'] for p in cross_pairs])
            print(f"  Cross-NT corrected_min: "
                  f"median={np.median(cc):.3f}, "
                  f"mean={np.mean(cc):.3f}, "
                  f"min={np.min(cc):.3f}, "
                  f"p5={np.percentile(cc, 5):.3f}")

    return {
        'same_nt_corrected': [float(x) for x in same_nt_corrected],
        'cross_nt_corrected': [float(x) for x in cross_nt_corrected],
        'same_nt_raw': [float(x) for x in same_nt_raw],
        'cross_nt_raw': [float(x) for x in cross_nt_raw],
        'noise_model_params': {'floor': float(noise_floor),
                               'coeff': float(noise_coeff)},
        'n_candidates': len(candidates),
        'elapsed': round(elapsed, 1),
        'all_pairs': all_pairs,  # full detail for every pair
    }


def main():
    parser = argparse.ArgumentParser(
        description="Terminal scaling experiment: "
                    "fixed binary rules, varying terminal count")
    parser.add_argument('output', help='Output JSON file')
    parser.add_argument('--nonterminals', type=int, default=5,
                        help='Number of nonterminals (default: 5)')
    parser.add_argument('--terminal_counts', type=int, nargs='+',
                        default=[50, 100, 300, 1000],
                        help='Terminal counts to test (default: 50 100 300 1000)')
    parser.add_argument('--n_sentences', type=int, default=100000,
                        help='Corpus size (default: 100000)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='RNN training epochs (default: 20)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base seed for grammar generation (default: 42)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory for persistent data (grammars, '
                             'corpora, models). If not set, uses a temp dir.')

    args = parser.parse_args()

    # Set up persistent data directory
    if args.data_dir:
        data_dir = args.data_dir
        os.makedirs(data_dir, exist_ok=True)
    else:
        data_dir = tempfile.mkdtemp(prefix='terminal_scaling_')
    print(f"Data directory: {data_dir}")

    # Generate base grammar to get binary rules
    print("Generating base grammar...")
    grammar_base, binary_rules, unary_totals = generate_base_grammar(
        args.nonterminals, args.seed)
    nonterminals = sorted(grammar_base.nonterminals)
    start = grammar_base.start

    print(f"  {len(binary_rules)} binary rules, "
          f"{len(nonterminals)} nonterminals")
    print(f"  Unary totals: "
          + ", ".join(f"{nt}={unary_totals.get(nt, 0):.3f}"
                      for nt in nonterminals))

    results = []
    for n_t in args.terminal_counts:
        print(f"\n{'='*60}")
        print(f"Terminal count: {n_t}")
        print(f"{'='*60}")

        # Per-terminal-count subdirectory
        run_dir = os.path.join(data_dir, f'nt{n_t}')
        os.makedirs(run_dir, exist_ok=True)

        # Create grammar variant
        grammar = make_grammar_with_terminals(
            binary_rules, unary_totals, nonterminals, start,
            n_t, seed=args.seed + n_t)

        grammar_path = os.path.join(run_dir, 'grammar.pcfg')
        grammar.store(grammar_path)

        # Sample corpus (reuse if it exists)
        corpus_path = os.path.join(run_dir, 'corpus.txt')
        if os.path.exists(corpus_path):
            with open(corpus_path) as f:
                n_lines = sum(1 for _ in f)
            print(f"  Reusing corpus: {n_lines} sentences")
        else:
            sentences = sample_corpus(grammar, args.n_sentences,
                                      seed=args.seed + 2000 + n_t)
            with open(corpus_path, 'w') as f:
                for s in sentences:
                    f.write(s + '\n')
            print(f"  Corpus: {len(sentences)} sentences, "
                  f"vocab={len(set(w for s in sentences for w in s.split()))}")

        # Model cache directory
        model_dir = os.path.join(run_dir, 'models')

        # Run analysis
        analysis = run_divergence_analysis(
            corpus_path, grammar,
            max_terminals=min(n_t, 300),
            epochs=args.epochs,
            model_dir=model_dir)

        result = {
            'n_terminals': n_t,
            'n_nonterminals': args.nonterminals,
            'n_sentences': analysis.get('n_sentences',
                                        args.n_sentences),
            **analysis,
        }
        results.append(result)

        # Save incrementally
        with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Terminals':>10} {'Cands':>6} "
          f"{'SameNT_med':>11} {'SameNT_max':>11} {'SameNT_p99':>11} "
          f"{'CrossNT_min':>12} {'CrossNT_p5':>11} {'Gap':>8} "
          f"{'Time':>6}")
    print('-' * 100)
    for r in results:
        sc = np.array(r['same_nt_corrected']) if r['same_nt_corrected'] else np.array([0])
        cc = np.array(r['cross_nt_corrected']) if r['cross_nt_corrected'] else np.array([0])
        gap = np.min(cc) - np.max(sc) if len(cc) > 0 and len(sc) > 0 else 0
        print(f"{r['n_terminals']:>10} {r['n_candidates']:>6} "
              f"{np.median(sc):>11.3f} {np.max(sc):>11.3f} "
              f"{np.percentile(sc, 99):>11.3f} "
              f"{np.min(cc):>12.3f} {np.percentile(cc, 5):>11.3f} "
              f"{gap:>8.3f} "
              f"{r['elapsed']:>5.0f}s")


if __name__ == '__main__':
    main()
