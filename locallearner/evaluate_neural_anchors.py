#!/usr/bin/env python3
"""Evaluate neural anchor selection on randomly generated PCFGs.

Uses syntheticpcfg.FullPCFGFactory to generate grammars of varying size,
samples corpora, trains RNN cloze models, runs divergence-ordered anchor
selection, and evaluates the results against the true grammar.

Usage:
    python3 evaluate_neural_anchors.py results.json
    python3 evaluate_neural_anchors.py results.json --n_grammars 5 --n_sentences 100000
    python3 evaluate_neural_anchors.py results.json --nonterminals 5 8 10
"""

import sys
import os
import json
import argparse
import tempfile
import time
import logging
import random as pyrandom

import numpy as np

# Add packages to path
sys.path.insert(0, os.path.dirname(__file__))
SYNTHETICPCFG_PATH = os.path.join(os.path.dirname(__file__),
    '..', '..', '..', 'cursor_syntheticpcfg', 'syntheticpcfg')
sys.path.insert(0, SYNTHETICPCFG_PATH)

import wcfg
import evaluation

from syntheticpcfg import pcfgfactory, pcfg as spcfg, utility as sutil


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
            # max depth exceeded — skip
            continue
    return sentences


def run_neural_anchor_selection(corpus_path, alpha=2.0, epochs=20,
                                max_terminals=300, verbose=True):
    """Train RNN models and run divergence-ordered anchor selection.

    Returns:
        anchors: list of selected anchor words
        nl: NeuralLearner instance (for diagnostics)
        elapsed: total time in seconds
    """
    from neural_learner import NeuralLearner

    nl = NeuralLearner(corpus_path)
    nl.alpha = alpha
    nl.model_type = 'rnn'
    nl.n_epochs = epochs

    t0 = time.time()
    nl.train_single_model(verbose=verbose)
    nl.train_split_models(verbose=verbose)
    anchors = nl.select_anchors_divergence_ordered(
        max_terminals=max_terminals, verbose=verbose)
    elapsed = time.time() - t0

    return anchors, nl, elapsed


def evaluate_anchors(grammar_path, anchors):
    """Evaluate anchor quality against the target grammar.

    Anchors are non-S terminals (S is handled as a virtual anchor).
    The correct count is n_target - 1 (excluding S).

    Returns dict with evaluation metrics including per-anchor NT posteriors.
    """
    target = wcfg.load_wcfg_from_file(grammar_path)
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
    from collections import Counter
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


def run_experiment(config, verbose=True):
    """Run a single experiment. Returns a result dict."""
    n_nt = config['n_nonterminals']
    n_t = config['n_terminals']
    n_sentences = config['n_sentences']
    grammar_seed = config['grammar_seed']
    corpus_seed = config['corpus_seed']
    alpha = config.get('alpha', 2.0)
    epochs = config.get('epochs', 20)

    result = {
        'n_nonterminals': n_nt,
        'n_terminals': n_t,
        'n_sentences': n_sentences,
        'grammar_seed': grammar_seed,
        'corpus_seed': corpus_seed,
    }

    try:
        # Generate grammar
        if verbose:
            print(f"\n{'='*60}")
            print(f"Grammar: {n_nt} NTs, {n_t} terminals, "
                  f"seed={grammar_seed}, corpus={n_sentences} sentences")
            print(f"{'='*60}")

        grammar = generate_grammar(n_nt, n_t, grammar_seed)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Store grammar
            grammar_path = os.path.join(tmpdir, 'grammar.pcfg')
            grammar.store(grammar_path)

            # Sample corpus
            sentences = sample_corpus(grammar, n_sentences, corpus_seed)
            corpus_path = os.path.join(tmpdir, 'corpus.txt')
            with open(corpus_path, 'w') as f:
                for s in sentences:
                    f.write(s + '\n')

            result['actual_sentences'] = len(sentences)
            result['vocab_size'] = len(set(w for s in sentences
                                           for w in s.split()))

            # Run anchor selection
            anchors, nl, elapsed = run_neural_anchor_selection(
                corpus_path, alpha=alpha, epochs=epochs, verbose=verbose)

            result['elapsed'] = round(elapsed, 2)
            result['anchors'] = anchors

            # Evaluate
            eval_result = evaluate_anchors(grammar_path, anchors)
            result.update(eval_result)

            if verbose:
                print(f"\nResult: found {len(anchors)} anchors "
                      f"(target: {n_nt})")
                if eval_result.get('mean_posterior') is not None:
                    print(f"  Mean posterior: "
                          f"{eval_result['mean_posterior']:.3f}")
                    print(f"  Min posterior:  "
                          f"{eval_result['min_posterior']:.3f}")
                    print(f"  Accuracy:       "
                          f"{eval_result['accuracy']:.3f}")
                print(f"  NTs covered: {eval_result.get('nts_covered', [])}")
                if eval_result.get('duplicate_nts'):
                    print(f"  Duplicates: {eval_result['duplicate_nts']}")
                print(f"  Time: {elapsed:.1f}s")
                for w in anchors:
                    info = eval_result.get('anchor_info', {}).get(w, {})
                    nt = info.get('best_nt', '?')
                    post = info.get('posterior', 0)
                    print(f"    {w}: {nt} ({post:.3f})")

    except Exception as e:
        result['error'] = str(e)
        if verbose:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate neural anchor selection on random PCFGs")
    parser.add_argument('output', help='Output JSON file for results')
    parser.add_argument('--nonterminals', type=int, nargs='+',
                        default=[5, 8, 10],
                        help='Numbers of nonterminals to test (default: 5 8 10)')
    parser.add_argument('--terminals', type=int, default=1000,
                        help='Number of terminals (default: 1000)')
    parser.add_argument('--n_grammars', type=int, default=3,
                        help='Number of random grammars per size (default: 3)')
    parser.add_argument('--n_sentences', type=int, default=100000,
                        help='Corpus size (default: 100000)')
    parser.add_argument('--alpha', type=float, default=2.0,
                        help='Renyi divergence alpha (default: 2.0)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='RNN training epochs (default: 20)')
    parser.add_argument('--base_seed', type=int, default=1,
                        help='Starting seed for grammar generation (default: 1)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress detailed output')

    args = parser.parse_args()

    configs = []
    seed = args.base_seed
    for n_nt in args.nonterminals:
        for i in range(args.n_grammars):
            configs.append({
                'n_nonterminals': n_nt,
                'n_terminals': args.terminals,
                'n_sentences': args.n_sentences,
                'grammar_seed': seed,
                'corpus_seed': seed + 1000,
                'alpha': args.alpha,
                'epochs': args.epochs,
            })
            seed += 1

    results = []
    for i, config in enumerate(configs):
        if not args.quiet:
            print(f"\n[{i+1}/{len(configs)}]", end='')
        result = run_experiment(config, verbose=not args.quiet)
        results.append(result)

        # Save incrementally
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'NTs':>4} {'Seed':>5} {'Found':>6} {'NonS':>5} "
          f"{'Match':>6} {'Covered':>8} {'Dups':>5} {'MeanPost':>9} "
          f"{'MinPost':>8} {'Time':>6}")
    print('-' * 72)
    for r in results:
        match = 'YES' if r.get('correct_count') else 'no'
        mp = f"{r['mean_posterior']:.3f}" if r.get('mean_posterior') is not None else '-'
        mnp = f"{r['min_posterior']:.3f}" if r.get('min_posterior') is not None else '-'
        elapsed = f"{r.get('elapsed', 0):.0f}s"
        covered = r.get('n_nts_covered', '?')
        dups = sum(v - 1 for v in r.get('duplicate_nts', {}).values())
        dups_s = str(dups) if dups > 0 else '-'
        non_s = r.get('n_non_s_target', r['n_nonterminals'] - 1)
        print(f"{r['n_nonterminals']:>4} {r['grammar_seed']:>5} "
              f"{r.get('n_detected', '?'):>6} {non_s:>5} "
              f"{match:>6} {covered:>8} {dups_s:>5} {mp:>9} "
              f"{mnp:>8} {elapsed:>6}")

    correct = sum(1 for r in results if r.get('correct_count'))
    print(f"\nCorrect count: {correct}/{len(results)}")


if __name__ == '__main__':
    main()
