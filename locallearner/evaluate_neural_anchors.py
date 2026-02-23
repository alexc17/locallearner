#!/usr/bin/env python3
"""Evaluate anchor selection methods on randomly generated PCFGs.

Supports multiple methods (neural models, gold kernels, supervised MLE,
target ceiling) on the same grammars and corpora for fair comparison.

Usage (config file, recommended):
    python3 evaluate_neural_anchors.py --config experiment.json

All settings live in the config JSON for reproducibility.  CLI flags
override config values when given.

Config file format:
    {
      "output": "results.json",
      "workdir": "./eval_output",
      "quiet": false,
      "grammars": {
        "n_nonterminals": [5],
        "n_terminals": 1000,
        "n_grammars": 5,
        "n_sentences": 100000,
        "base_seed": 1
      },
      "methods": [
        {"name": "rnn_div", "model_type": "rnn", "kernel_method": "divergence_ordered",
         "epochs": 20, "alpha": 2.0, "max_terminals": 300},
        {"name": "gold", "kernel_method": "gold"},
        {"name": "supervised", "kernel_method": "supervised", "ml_maxlength": 10},
        {"name": "target", "kernel_method": "target"}
      ]
    }

CLI flags (--output, --workdir, --quiet) override the config values.
"""

import sys
import os
import json
import argparse
import math
import tempfile
import time
import logging
import random as pyrandom
from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

# Add packages to path
sys.path.insert(0, os.path.dirname(__file__))
SYNTHETICPCFG_PATH = os.path.join(os.path.dirname(__file__),
    '..', '..', '..', 'cursor_syntheticpcfg', 'syntheticpcfg')
sys.path.insert(0, SYNTHETICPCFG_PATH)

import wcfg
import evaluation

from syntheticpcfg import pcfgfactory, pcfg as spcfg, utility as sutil


# ── Grammar generation ──────────────────────────────────────────────

def generate_grammar(n_nonterminals, n_terminals, seed):
    """Generate a random PCFG using FullPCFGFactory.

    Same settings as evaluate_autodetect.py:
    - LogNormal(sigma=4.0) lexical distribution
    - Dirichlet(alpha = 1/((n-1)^2+1)) binary distribution
    - Poisson(5.0, max_length=20) length distribution
    - Full CNF backbone
    """
    pyrandom.seed(seed)
    np.random.seed(seed)

    factory = pcfgfactory.FullPCFGFactory(
        nonterminals=n_nonterminals,
        terminals=n_terminals
    )
    factory.lexical_distribution = pcfgfactory.LogNormalPrior(sigma=4.0)
    alpha = 1.0 / ((n_nonterminals - 1) ** 2 + 1)
    factory.binary_distribution = pcfgfactory.LexicalDirichlet(dirichlet=alpha)
    factory.length_distribution = pcfgfactory.LengthDistribution()
    factory.length_distribution.set_poisson(5.0, 20)

    grammar = factory.sample_uniform()
    return grammar


def sample_corpus(grammar, n_sentences, seed):
    """Sample a corpus from a syntheticpcfg PCFG.

    Returns:
        sentences: list of yield strings ("w1 w2 w3")
        trees: list of tree strings in S-expression format
    """
    rng = np.random.RandomState(seed)
    sampler = spcfg.Sampler(grammar, random=rng)
    sentences = []
    trees = []
    for _ in range(n_sentences):
        try:
            tree = sampler.sample_tree()
            s = sutil.collect_yield(tree)
            sentences.append(' '.join(s))
            trees.append(sutil.tree_to_string(tree))
        except ValueError:
            # max depth exceeded — skip
            continue
    return sentences, trees


# ── Gold kernels (Hungarian algorithm) ──────────────────────────────

def get_gold_kernels(grammar):
    """Find optimal anchor word for each non-start NT using Hungarian algorithm."""
    te = grammar.terminal_expectations()
    pe = grammar.production_expectations()
    non_start_nts = sorted([nt for nt in grammar.nonterminals
                            if nt != grammar.start])
    terminals = sorted(grammar.terminals)
    nn = len(non_start_nts)
    nt_count = len(terminals)

    score_matrix = np.full((nn, nt_count), -1e30)
    for j, ntj in enumerate(non_start_nts):
        for i, a in enumerate(terminals):
            ea = te.get(a, 0.0)
            if ea <= 0:
                continue
            e_prod = pe.get((ntj, a), 0.0)
            if e_prod <= 0:
                continue
            posterior = e_prod / ea
            score_matrix[j, i] = 0.5 * math.log(posterior) + 0.5 * math.log(ea)

    dim = max(nn, nt_count)
    padded = np.full((dim, dim), -1e30)
    padded[:nn, :nt_count] = score_matrix
    row_ind, col_ind = linear_sum_assignment(-padded)

    assignment = {}
    for j, i in zip(row_ind, col_ind):
        if j < nn and i < nt_count:
            assignment[non_start_nts[j]] = terminals[i]

    return [assignment[ntj] for ntj in non_start_nts]


# ── Anchor evaluation ───────────────────────────────────────────────

def evaluate_anchors(target, anchors):
    """Evaluate anchor quality against the target grammar.

    Anchors are non-S terminals (S is handled as a virtual anchor).
    The correct count is n_target - 1 (excluding S).

    Returns dict with evaluation metrics including per-anchor NT posteriors.
    """
    n_target = len(target.nonterminals)
    n_non_s = n_target - 1  # anchors exclude S

    result = {
        'n_detected': len(anchors),
        'n_target': n_target,
        'n_non_s_target': n_non_s,
        'correct_count': len(anchors) == n_non_s,
        'delta': len(anchors) - n_non_s,
    }

    # Compute per-anchor NT posteriors regardless of count match
    te = target.terminal_expectations()
    pe = target.production_expectations()
    anchor_info = {}
    nt_coverage = set()
    for w in anchors:
        if w in te and te[w] > 0:
            posteriors = {}
            for nt in sorted(target.nonterminals):
                p = pe.get((nt, w), 0.0) / te[w]
                if p > 0.001:
                    posteriors[nt] = round(float(p), 4)
            best_nt = max(posteriors, key=posteriors.get)
            anchor_info[w] = {
                'best_nt': best_nt,
                'posterior': round(float(posteriors[best_nt]), 4),
                'posteriors': posteriors,
            }
            nt_coverage.add(best_nt)
        else:
            anchor_info[w] = {'best_nt': '?', 'posterior': 0.0}
    result['anchor_info'] = anchor_info
    result['n_nts_covered'] = len(nt_coverage)
    result['nts_covered'] = sorted(nt_coverage)

    # Count duplicates (multiple anchors mapping to same NT)
    nt_counts = Counter(info['best_nt'] for info in anchor_info.values())
    result['duplicate_nts'] = {nt: c for nt, c in nt_counts.items() if c > 1}

    if len(anchors) == n_non_s:
        # Full kernel list with S placeholder
        kernels = ['S'] + list(anchors)
        eval_result = evaluation.evaluate_kernels_hungarian(target, kernels)
        result.update({
            'mean_posterior': eval_result['mean_posterior'],
            'min_posterior': eval_result['min_posterior'],
            'accuracy': eval_result['accuracy'],
        })
    else:
        result.update({
            'mean_posterior': None,
            'min_posterior': None,
            'accuracy': None,
        })

    return result


# ── Method runners ──────────────────────────────────────────────────

def run_method_divergence_ordered(method, corpus_path, target, verbose):
    """Run divergence-ordered anchor selection (RNN + split-half)."""
    from neural_learner import NeuralLearner

    nl = NeuralLearner(corpus_path)
    nl.model_type = method.get('model_type', 'rnn')
    nl.alpha = method.get('alpha', 2.0)
    nl.n_epochs = method.get('epochs', 20)
    nl.embedding_dim = method.get('embedding_dim', 64)
    nl.hidden_dim = method.get('hidden_dim', 128)
    max_terminals = method.get('max_terminals', 300)

    t0 = time.time()
    nl.train_single_model(verbose=verbose)
    nl.train_split_models(verbose=verbose)
    anchors = nl.select_anchors_divergence_ordered(
        max_terminals=max_terminals, verbose=verbose)
    elapsed = time.time() - t0

    result = evaluate_anchors(target, anchors)
    result['anchors'] = anchors
    result['elapsed'] = round(elapsed, 2)
    return result


def run_method_minimal(method, corpus_path, target, verbose):
    """Run minimal (antichain) anchor selection with any model type."""
    from neural_learner import NeuralLearner

    nl = NeuralLearner(corpus_path)
    nl.model_type = method.get('model_type', 'bow')
    nl.k = method.get('k', 2)
    nl.alpha = method.get('alpha', 2.0)
    nl.n_epochs = method.get('epochs', 20)
    nl.embedding_dim = method.get('embedding_dim', 64)
    nl.hidden_dim = method.get('hidden_dim', 128)
    max_terminals = method.get('max_terminals', 500)
    epsilon = method.get('epsilon', 1.5)

    t0 = time.time()
    nl.train_single_model(verbose=verbose)
    anchors = nl.select_anchors_minimal(
        max_terminals=max_terminals, epsilon=epsilon, verbose=verbose)
    elapsed = time.time() - t0

    result = evaluate_anchors(target, anchors)
    result['anchors'] = anchors
    result['elapsed'] = round(elapsed, 2)
    return result


def run_method_gold(method, target, verbose):
    """Run gold (oracle) anchor selection via Hungarian algorithm."""
    t0 = time.time()
    anchors = get_gold_kernels(target)
    elapsed = time.time() - t0

    result = evaluate_anchors(target, anchors)
    result['anchors'] = anchors
    result['elapsed'] = round(elapsed, 2)
    return result


def run_method_supervised(method, trees_path, target, verbose):
    """Evaluate supervised MLE grammar from sampled trees."""
    ml_maxlength = method.get('ml_maxlength', 10)

    t0 = time.time()
    ml_grammar = wcfg.load_wcfg_from_treebank(trees_path, ml_maxlength, 0,
                                                pcfg=True)
    elapsed = time.time() - t0

    # Save ML grammar next to the trees file for inspection
    ml_path = os.path.join(os.path.dirname(trees_path), 'ml.pcfg')
    ml_grammar.store(ml_path)

    # KLD against target.
    # The ML grammar has the same NT names as the target (sampled from it),
    # so we don't need bijection estimation.  Fall back gracefully if
    # the ML grammar is degenerate (missing NTs from short-sentence filter).
    try:
        kld = evaluation.smoothed_kld_exact(target, ml_grammar,
                                            compute_bijection=False)
    except Exception:
        try:
            kld = evaluation.smoothed_kld_exact(target, ml_grammar,
                                                compute_bijection=True)
        except Exception:
            kld = None

    result = {
        'kernel_method': 'supervised',
        'elapsed': round(elapsed, 2),
        'kld': kld,
        'n_terminals': len(ml_grammar.terminals),
        'expected_length': ml_grammar.expected_length(),
        'ml_maxlength': ml_maxlength,
    }

    if verbose:
        kld_s = f"{kld:.4f}" if kld is not None else "n/a"
        print(f"  Supervised MLE: {len(ml_grammar.terminals)} terminals, "
              f"E[len]={ml_grammar.expected_length():.3f}, KLD={kld_s}")

    return result


def run_method_target(method, target, verbose):
    """Evaluate target grammar against itself (ceiling)."""
    t0 = time.time()
    elen = target.expected_length()
    elapsed = time.time() - t0

    result = {
        'kernel_method': 'target',
        'elapsed': round(elapsed, 2),
        'kld': 0.0,
        'n_terminals': len(target.terminals),
        'expected_length': elen,
    }

    if verbose:
        print(f"  Target: {len(target.terminals)} terminals, "
              f"E[len]={elen:.3f}, KLD=0.0")

    return result


# ── Main experiment loop ────────────────────────────────────────────

def run_experiment(grammar_config, methods, verbose=True, workdir=None):
    """Run all methods on a single grammar. Returns list of result dicts.

    Args:
        grammar_config: dict with grammar parameters
        methods: list of method dicts
        verbose: print progress
        workdir: if set, use this as the base directory and create a
            subdirectory for this grammar (e.g. workdir/nt5_s1/).
            If None, use a temporary directory.
    """
    n_nt = grammar_config['n_nonterminals']
    n_t = grammar_config['n_terminals']
    n_sentences = grammar_config['n_sentences']
    grammar_seed = grammar_config['grammar_seed']
    corpus_seed = grammar_config['corpus_seed']

    if verbose:
        print(f"\n{'='*60}")
        print(f"Grammar: {n_nt} NTs, {n_t} terminals, "
              f"seed={grammar_seed}, corpus={n_sentences} sentences")
        print(f"{'='*60}")

    grammar = generate_grammar(n_nt, n_t, grammar_seed)

    if workdir is not None:
        tmpdir = os.path.join(workdir, f'nt{n_nt}_s{grammar_seed}')
        os.makedirs(tmpdir, exist_ok=True)
    else:
        tmpdir = tempfile.mkdtemp(prefix='eval_anchors_')
    if verbose:
        print(f"  Working directory: {tmpdir}")

    try:
        # Store grammar
        grammar_path = os.path.join(tmpdir, 'grammar.pcfg')
        grammar.store(grammar_path)
        target = wcfg.load_wcfg_from_file(grammar_path)

        # Sample corpus (yields + trees, shared across methods)
        sentences, trees = sample_corpus(grammar, n_sentences, corpus_seed)
        corpus_path = os.path.join(tmpdir, 'corpus.txt')
        with open(corpus_path, 'w') as f:
            for s in sentences:
                f.write(s + '\n')

        trees_path = os.path.join(tmpdir, 'trees.txt')
        with open(trees_path, 'w') as f:
            for t in trees:
                f.write(t + '\n')

        actual_sentences = len(sentences)
        vocab_size = len(set(w for s in sentences for w in s.split()))

        if verbose:
            print(f"  Corpus: {actual_sentences} sentences, "
                  f"{vocab_size} types")

        # Run each method
        method_results = []
        for method in methods:
            name = method['name']
            kernel_method = method['kernel_method']

            if verbose:
                print(f"\n--- Method: {name} ({kernel_method}) ---")

            base = {
                'n_nonterminals': n_nt,
                'n_terminals': n_t,
                'n_sentences': n_sentences,
                'grammar_seed': grammar_seed,
                'corpus_seed': corpus_seed,
                'actual_sentences': actual_sentences,
                'vocab_size': vocab_size,
                'method': name,
                'kernel_method': kernel_method,
            }

            try:
                if kernel_method == 'divergence_ordered':
                    r = run_method_divergence_ordered(
                        method, corpus_path, target, verbose)
                elif kernel_method == 'minimal':
                    r = run_method_minimal(
                        method, corpus_path, target, verbose)
                elif kernel_method == 'gold':
                    r = run_method_gold(method, target, verbose)
                elif kernel_method == 'supervised':
                    r = run_method_supervised(
                        method, trees_path, target, verbose)
                elif kernel_method == 'target':
                    r = run_method_target(method, target, verbose)
                else:
                    raise ValueError(
                        f"Unknown kernel_method: {kernel_method}")

                base.update(r)

                # Print anchor-based results
                if verbose and 'anchors' in base:
                    anchors = base['anchors']
                    print(f"\n  Result: found {len(anchors)} anchors "
                          f"(target: {n_nt})")
                    if base.get('mean_posterior') is not None:
                        print(f"    Mean posterior: "
                              f"{base['mean_posterior']:.3f}")
                        print(f"    Min posterior:  "
                              f"{base['min_posterior']:.3f}")
                        print(f"    Accuracy:       "
                              f"{base['accuracy']:.3f}")
                    print(f"    NTs covered: "
                          f"{base.get('nts_covered', [])}")
                    if base.get('duplicate_nts'):
                        print(f"    Duplicates: "
                              f"{base['duplicate_nts']}")
                    print(f"    Time: {base.get('elapsed', 0):.1f}s")
                    for w in anchors:
                        info = base.get('anchor_info', {}).get(w, {})
                        nt = info.get('best_nt', '?')
                        post = info.get('posterior', 0)
                        print(f"      {w}: {nt} ({post:.3f})")

            except Exception as e:
                base['error'] = str(e)
                if verbose:
                    print(f"  ERROR: {e}")
                    import traceback
                    traceback.print_exc()

            method_results.append(base)

    except Exception as e:
        if verbose:
            print(f"  Setup ERROR: {e}")
            import traceback
            traceback.print_exc()
        return [{'error': str(e), 'grammar_seed': grammar_seed,
                 'n_nonterminals': n_nt}]

    return method_results


def print_summary(results):
    """Print a summary table of all results."""
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    # Separate anchor-based and grammar-based results
    anchor_results = [r for r in results
                      if r.get('kernel_method') not in
                      ('supervised', 'target')]
    grammar_results = [r for r in results
                       if r.get('kernel_method') in
                       ('supervised', 'target')]

    if anchor_results:
        print(f"\n{'Method':>15} {'NTs':>4} {'Seed':>5} {'Found':>6} "
              f"{'NonS':>5} {'Match':>6} {'Covered':>8} {'MeanPost':>9} "
              f"{'MinPost':>8} {'Time':>6}")
        print('-' * 88)
        for r in anchor_results:
            match = 'YES' if r.get('correct_count') else 'no'
            mp = (f"{r['mean_posterior']:.3f}"
                  if r.get('mean_posterior') is not None else '-')
            mnp = (f"{r['min_posterior']:.3f}"
                   if r.get('min_posterior') is not None else '-')
            elapsed = f"{r.get('elapsed', 0):.0f}s"
            covered = r.get('n_nts_covered', '?')
            non_s = r.get('n_non_s_target',
                          r['n_nonterminals'] - 1)
            print(f"{r['method']:>15} {r['n_nonterminals']:>4} "
                  f"{r['grammar_seed']:>5} "
                  f"{r.get('n_detected', '?'):>6} {non_s:>5} "
                  f"{match:>6} {covered:>8} {mp:>9} "
                  f"{mnp:>8} {elapsed:>6}")

        # Per-method accuracy summary
        method_names = sorted(set(r['method'] for r in anchor_results))
        for name in method_names:
            mr = [r for r in anchor_results if r['method'] == name]
            correct = sum(1 for r in mr if r.get('correct_count'))
            print(f"  {name}: {correct}/{len(mr)} correct count")

    if grammar_results:
        print(f"\n{'Method':>15} {'NTs':>4} {'Seed':>5} {'KLD':>10} "
              f"{'Terms':>6} {'E[len]':>8} {'Time':>6}")
        print('-' * 60)
        for r in grammar_results:
            kld = (f"{r['kld']:.4f}"
                   if r.get('kld') is not None else '-')
            elapsed = f"{r.get('elapsed', 0):.1f}s"
            elen = (f"{r['expected_length']:.3f}"
                    if r.get('expected_length') is not None else '-')
            n_terms = r.get('n_terminals', '?')
            print(f"{r['method']:>15} {r['n_nonterminals']:>4} "
                  f"{r['grammar_seed']:>5} {kld:>10} "
                  f"{n_terms:>6} {elen:>8} {elapsed:>6}")


# ── Config loading ──────────────────────────────────────────────────

def load_config(config_path):
    """Load experiment config from JSON file.

    Required fields: grammars.n_nonterminals, methods.
    Optional top-level fields: output, workdir, quiet.
    """
    with open(config_path) as f:
        cfg = json.load(f)

    # Validate
    if 'grammars' not in cfg:
        raise ValueError("Config must have 'grammars' section")
    if 'methods' not in cfg:
        raise ValueError("Config must have 'methods' section")

    g = cfg['grammars']
    if 'n_nonterminals' not in g:
        raise ValueError("grammars.n_nonterminals required")

    # Ensure n_nonterminals is a list
    if isinstance(g['n_nonterminals'], int):
        g['n_nonterminals'] = [g['n_nonterminals']]

    # Validate methods
    for m in cfg['methods']:
        if 'name' not in m:
            raise ValueError("Each method must have a 'name'")
        if 'kernel_method' not in m:
            raise ValueError(
                f"Method '{m['name']}' must have 'kernel_method'")
        valid = ('divergence_ordered', 'minimal', 'gold',
                 'supervised', 'target')
        if m['kernel_method'] not in valid:
            raise ValueError(
                f"Method '{m['name']}': kernel_method must be one "
                f"of {valid}, got '{m['kernel_method']}'")

    return cfg


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate anchor selection methods on random PCFGs. "
        "All settings are in a JSON config file; CLI flags override.")
    parser.add_argument('--config', type=str, required=True,
                        help='JSON config file (required)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file (overrides config)')
    parser.add_argument('--workdir', type=str, default=None,
                        help='Directory for intermediate files '
                        '(overrides config)')
    parser.add_argument('--quiet', action='store_true', default=None,
                        help='Suppress detailed output (overrides config)')

    args = parser.parse_args()
    cfg = load_config(args.config)

    # Resolve settings: CLI overrides config
    output = args.output or cfg.get('output', 'results.json')
    workdir = args.workdir or cfg.get('workdir', None)
    quiet = args.quiet if args.quiet is not None else cfg.get('quiet', False)

    if workdir is not None:
        workdir = os.path.abspath(workdir)
        os.makedirs(workdir, exist_ok=True)
        if not quiet:
            print(f"Work directory: {workdir}")

    g = cfg['grammars']
    methods = cfg['methods']
    n_nonterminals_list = g['n_nonterminals']
    n_terminals = g.get('n_terminals', 1000)
    n_grammars = g.get('n_grammars', 3)
    n_sentences = g.get('n_sentences', 100000)
    base_seed = g.get('base_seed', 1)

    if not quiet:
        print(f"Config: {args.config}")
        print(f"Output: {output}")
        print(f"Grammars: {n_nonterminals_list} NTs, {n_terminals} terminals, "
              f"{n_grammars} per size, {n_sentences} sentences, "
              f"seed={base_seed}")
        print(f"Methods: {', '.join(m['name'] for m in methods)}")

    # Build grammar configs
    grammar_configs = []
    seed = base_seed
    for n_nt in n_nonterminals_list:
        for _ in range(n_grammars):
            grammar_configs.append({
                'n_nonterminals': n_nt,
                'n_terminals': n_terminals,
                'n_sentences': n_sentences,
                'grammar_seed': seed,
                'corpus_seed': seed + 1000,
            })
            seed += 1

    # Run experiments
    all_results = []
    total = len(grammar_configs)
    for i, gc in enumerate(grammar_configs):
        if not quiet:
            print(f"\n[{i+1}/{total}]", end='')
        results = run_experiment(gc, methods, verbose=not quiet,
                                 workdir=workdir)
        all_results.extend(results)

        # Save incrementally
        with open(output, 'w') as f:
            json.dump(all_results, f, indent=2)

    # Print summary
    print_summary(all_results)


if __name__ == '__main__':
    main()
