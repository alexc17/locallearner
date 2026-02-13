#!/usr/bin/env python3
"""Compare NMF kernel detection using marginal vs joint context features.

Generates synthetic PCFGs, samples corpora, runs the NMF kernel detection
with both feature modes, and evaluates:
  1. Fixed-count kernel quality (first N kernels, no stopping criterion)
  2. Auto-detection accuracy (correct number of NTs detected)

Produces a JSON results file and comparison plots.

Usage:
    python3 compare_joint_marginal.py results.json --plot results.png
    python3 compare_joint_marginal.py results.json --quick
"""

import sys
import os
import json
import argparse
import tempfile
import time

import numpy as np
import numpy.random

# Add packages to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
SYNTHETICPCFG_PATH = os.path.join(os.path.dirname(__file__),
    '..', '..', '..', 'cursor_syntheticpcfg', 'syntheticpcfg')
sys.path.insert(0, SYNTHETICPCFG_PATH)

import wcfg
import locallearner as ll_module
import evaluation

from syntheticpcfg import pcfgfactory, pcfg as spcfg, utility as sutil

# Reuse grammar/corpus generation from evaluate_autodetect
from evaluate_autodetect import (
    generate_grammar, sample_corpus, compute_grammar_difficulty,
    evaluate_kernels_against_target,
)


def run_detection(corpus_path, n_nonterminals, seed=42, min_count=10,
                  number_clusters=10, feature_mode='marginal',
                  auto=False, max_nt=30):
    """Run NMF kernel detection with specified feature mode.

    Args:
        corpus_path: Path to corpus file
        n_nonterminals: Number of NTs (used if auto=False)
        seed: Clustering seed
        min_count: Minimum word count for NMF
        number_clusters: Ney-Essen cluster count
        feature_mode: 'marginal' or 'joint'
        auto: If True, use auto-detection (ignore n_nonterminals)
        max_nt: Maximum NTs for auto-detection

    Returns:
        (kernels, learner, elapsed_seconds)
    """
    learner = ll_module.LocalLearner(corpus_path)
    learner.number_clusters = number_clusters
    learner.min_count_nmf = min_count
    learner.seed = seed
    learner.feature_mode = feature_mode

    if auto:
        learner.nonterminals = 0
        learner.max_nonterminals = max_nt
    else:
        learner.nonterminals = n_nonterminals

    t0 = time.time()
    kernels = learner.find_kernels(verbose=False)
    elapsed = time.time() - t0

    return kernels, learner, elapsed


def run_experiment(config):
    """Run a single experiment: both marginal and joint, fixed and auto."""
    n_nt = config['n_nonterminals']
    n_t = config['n_terminals']
    n_sentences = config['n_sentences']
    n_clusters = config.get('n_clusters', 10)
    grammar_seed = config['grammar_seed']
    corpus_seed = config['corpus_seed']

    result = {
        'n_nonterminals': n_nt,
        'n_terminals': n_t,
        'n_sentences': n_sentences,
        'n_clusters': n_clusters,
        'grammar_seed': grammar_seed,
        'corpus_seed': corpus_seed,
    }

    try:
        # Generate grammar
        grammar = generate_grammar(n_nt, n_t, grammar_seed)
        difficulty = compute_grammar_difficulty(grammar)
        result['eff_ctx_log_condition_number'] = difficulty['eff_ctx_log_condition_number']
        result['min_best_anchor'] = difficulty['min_best_anchor']

        with tempfile.TemporaryDirectory() as tmpdir:
            grammar_path = os.path.join(tmpdir, 'grammar.pcfg')
            grammar.store(grammar_path)

            sentences = sample_corpus(grammar, n_sentences, corpus_seed)
            corpus_path = os.path.join(tmpdir, 'corpus.txt')
            with open(corpus_path, 'w') as f:
                for s in sentences:
                    f.write(s + '\n')

            min_count = max(5, n_sentences // 1000)

            for mode in ['marginal', 'joint']:
                prefix = mode[:4]  # 'marg' or 'join'

                # --- Fixed-count detection ---
                kernels_f, learner_f, elapsed_f = run_detection(
                    corpus_path, n_nt, seed=42, min_count=min_count,
                    number_clusters=n_clusters, feature_mode=mode,
                    auto=False)
                eval_f = evaluate_kernels_against_target(grammar_path, kernels_f)
                result[f'{prefix}_fixed_kernels'] = kernels_f
                result[f'{prefix}_fixed_mean_post'] = eval_f['mean_posterior']
                result[f'{prefix}_fixed_min_post'] = eval_f['min_posterior']
                result[f'{prefix}_fixed_accuracy'] = eval_f['accuracy']
                result[f'{prefix}_fixed_elapsed'] = round(elapsed_f, 2)

                # --- Auto-detection ---
                kernels_a, learner_a, elapsed_a = run_detection(
                    corpus_path, n_nt, seed=42, min_count=min_count,
                    number_clusters=n_clusters, feature_mode=mode,
                    auto=True, max_nt=30)
                eval_a = evaluate_kernels_against_target(grammar_path, kernels_a)
                result[f'{prefix}_auto_n_detected'] = len(kernels_a)
                result[f'{prefix}_auto_correct'] = len(kernels_a) == n_nt
                result[f'{prefix}_auto_delta'] = len(kernels_a) - n_nt
                result[f'{prefix}_auto_mean_post'] = eval_a['mean_posterior']
                result[f'{prefix}_auto_accuracy'] = eval_a['accuracy']
                result[f'{prefix}_auto_elapsed'] = round(elapsed_a, 2)
                result[f'{prefix}_auto_stop'] = getattr(learner_a, 'stop_reason', None)

    except Exception as e:
        import traceback
        result['error'] = str(e)
        result['traceback'] = traceback.format_exc()

    return result


def build_experiment_grid(args):
    """Build experiment configurations."""
    configs = []

    if args.quick:
        nt_sizes = [8, 10]
        corpus_sizes = [50000]
        n_terminals = 1000
        n_repeats = 3
    else:
        nt_sizes = [8, 10]
        corpus_sizes = [50000, 100000]
        n_terminals = 1000
        n_repeats = args.repeats

    n_clusters = 10

    for n_nt in nt_sizes:
        for n_s in corpus_sizes:
            for rep in range(n_repeats):
                configs.append({
                    'n_nonterminals': n_nt,
                    'n_terminals': n_terminals,
                    'n_sentences': n_s,
                    'n_clusters': n_clusters,
                    'grammar_seed': 1000 * rep + n_nt * 7 + 1,
                    'corpus_seed': 2000 * rep + 1,
                })

    return configs


def run_all_experiments(args):
    """Run the full experiment grid."""
    configs = build_experiment_grid(args)
    total = len(configs)
    results = []

    for i, config in enumerate(configs):
        label = (f"NT={config['n_nonterminals']}, "
                 f"N={config['n_sentences']}, "
                 f"seed={config['grammar_seed']}")
        print(f"[{i+1}/{total}] {label} ...", flush=True)

        result = run_experiment(config)
        results.append(result)

        if result.get('error'):
            print(f"  ERROR: {result['error'][:80]}")
        else:
            for mode in ['marginal', 'joint']:
                p = mode[:4]
                fa = result.get(f'{p}_fixed_accuracy', None)
                ad = result.get(f'{p}_auto_delta', None)
                ac = result.get(f'{p}_auto_correct', False)
                ok = 'OK' if ac else 'WRONG'
                fa_str = f'{fa:.3f}' if fa is not None else 'N/A'
                print(f"  {mode:8s}: fixed_acc={fa_str}, "
                      f"auto_delta={ad:+d} ({ok}), "
                      f"stop={result.get(f'{p}_auto_stop', '?')}")

    return results


def print_summary(results):
    """Print comparison summary."""
    from collections import defaultdict

    valid = [r for r in results if not r.get('error')]
    if not valid:
        print("No valid results.")
        return

    nt_vals = sorted(set(r['n_nonterminals'] for r in valid))

    print("\n" + "=" * 100)
    print("FIXED-COUNT KERNEL QUALITY (first N kernels, no stopping criterion)")
    print("-" * 100)
    print(f"{'NTs':>4s} {'Trials':>6s} "
          f"{'Marg Acc':>9s} {'Joint Acc':>10s} "
          f"{'Marg Post':>10s} {'Joint Post':>11s}")
    print("-" * 100)

    for n_nt in nt_vals:
        subset = [r for r in valid if r['n_nonterminals'] == n_nt]
        n = len(subset)

        marg_acc = [r['marg_fixed_accuracy'] for r in subset
                    if r.get('marg_fixed_accuracy') is not None]
        join_acc = [r['join_fixed_accuracy'] for r in subset
                    if r.get('join_fixed_accuracy') is not None]
        marg_post = [r['marg_fixed_mean_post'] for r in subset
                     if r.get('marg_fixed_mean_post') is not None]
        join_post = [r['join_fixed_mean_post'] for r in subset
                     if r.get('join_fixed_mean_post') is not None]

        ma = np.mean(marg_acc) if marg_acc else float('nan')
        ja = np.mean(join_acc) if join_acc else float('nan')
        mp = np.mean(marg_post) if marg_post else float('nan')
        jp = np.mean(join_post) if join_post else float('nan')

        print(f"{n_nt:4d} {n:6d} {ma:9.3f} {ja:10.3f} {mp:10.3f} {jp:11.3f}")

    # Auto-detection
    print("\n" + "=" * 100)
    print("AUTO-DETECTION ACCURACY")
    print("-" * 100)
    print(f"{'NTs':>4s} {'Trials':>6s} "
          f"{'Marg Corr':>10s} {'Marg Rate':>10s} {'Marg MnDelta':>12s} "
          f"{'Joint Corr':>10s} {'Joint Rate':>10s} {'Joint MnDelta':>13s}")
    print("-" * 100)

    total_marg_correct = 0
    total_join_correct = 0
    total_n = 0

    for n_nt in nt_vals:
        subset = [r for r in valid if r['n_nonterminals'] == n_nt]
        n = len(subset)
        total_n += n

        mc = sum(1 for r in subset if r.get('marg_auto_correct'))
        jc = sum(1 for r in subset if r.get('join_auto_correct'))
        total_marg_correct += mc
        total_join_correct += jc

        md = np.mean([r['marg_auto_delta'] for r in subset
                      if r.get('marg_auto_delta') is not None])
        jd = np.mean([r['join_auto_delta'] for r in subset
                      if r.get('join_auto_delta') is not None])
        mr = mc / n if n > 0 else 0
        jr = jc / n if n > 0 else 0

        print(f"{n_nt:4d} {n:6d} "
              f"{mc:10d} {mr:10.1%} {md:+12.1f} "
              f"{jc:10d} {jr:10.1%} {jd:+13.1f}")

    print("-" * 100)
    mr = total_marg_correct / total_n if total_n > 0 else 0
    jr = total_join_correct / total_n if total_n > 0 else 0
    print(f"{'ALL':>4s} {total_n:6d} "
          f"{total_marg_correct:10d} {mr:10.1%} {'':>12s} "
          f"{total_join_correct:10d} {jr:10.1%}")


def plot_results(results, output_path):
    """Generate comparison plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    valid = [r for r in results if not r.get('error')]
    if not valid:
        print("No valid results to plot.")
        return

    nt_vals = sorted(set(r['n_nonterminals'] for r in valid))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Marginal vs Joint Context Features for NMF Kernel Detection',
                 fontsize=14, fontweight='bold', y=0.98)

    # --- Plot 1: Fixed-count kernel accuracy ---
    ax = axes[0, 0]
    x_pos = np.arange(len(nt_vals))
    bar_w = 0.35
    for j, (mode, label, color) in enumerate([
            ('marg', 'Marginal', '#2196F3'),
            ('join', 'Joint', '#FF9800')]):
        rates = []
        for n_nt in nt_vals:
            subset = [r for r in valid if r['n_nonterminals'] == n_nt]
            accs = [r[f'{mode}_fixed_accuracy'] for r in subset
                    if r.get(f'{mode}_fixed_accuracy') is not None]
            rates.append(np.mean(accs) if accs else 0)
        offset = (j - 0.5) * bar_w
        ax.bar(x_pos + offset, rates, bar_w * 0.9, label=label,
               color=color, alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{nt} NTs' for nt in nt_vals])
    ax.set_ylabel('Mean accuracy (Hungarian)')
    ax.set_title('Fixed-count: kernel quality')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.2, axis='y')

    # --- Plot 2: Fixed-count mean posterior ---
    ax = axes[0, 1]
    for j, (mode, label, color) in enumerate([
            ('marg', 'Marginal', '#2196F3'),
            ('join', 'Joint', '#FF9800')]):
        posts = []
        for n_nt in nt_vals:
            subset = [r for r in valid if r['n_nonterminals'] == n_nt]
            ps = [r[f'{mode}_fixed_mean_post'] for r in subset
                  if r.get(f'{mode}_fixed_mean_post') is not None]
            posts.append(np.mean(ps) if ps else 0)
        offset = (j - 0.5) * bar_w
        ax.bar(x_pos + offset, posts, bar_w * 0.9, label=label,
               color=color, alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{nt} NTs' for nt in nt_vals])
    ax.set_ylabel('Mean posterior P(NT|anchor)')
    ax.set_title('Fixed-count: anchor quality')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.2, axis='y')

    # --- Plot 3: Auto-detection accuracy ---
    ax = axes[1, 0]
    for j, (mode, label, color) in enumerate([
            ('marg', 'Marginal', '#2196F3'),
            ('join', 'Joint', '#FF9800')]):
        rates = []
        annots = []
        for n_nt in nt_vals:
            subset = [r for r in valid if r['n_nonterminals'] == n_nt]
            n = len(subset)
            c = sum(1 for r in subset if r.get(f'{mode}_auto_correct'))
            rates.append(c / n if n > 0 else 0)
            annots.append(f'{c}/{n}')
        offset = (j - 0.5) * bar_w
        bars = ax.bar(x_pos + offset, rates, bar_w * 0.9, label=label,
                      color=color, alpha=0.85)
        for k, (bar, ct) in enumerate(zip(bars, annots)):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02, ct,
                    ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{nt} NTs' for nt in nt_vals])
    ax.set_ylabel('Accuracy (exact NT count)')
    ax.set_title('Auto-detection accuracy')
    ax.legend()
    ax.set_ylim(0, 1.25)
    ax.grid(True, alpha=0.2, axis='y')

    # --- Plot 4: Auto-detection delta distribution ---
    ax = axes[1, 1]
    for j, (mode, label, color) in enumerate([
            ('marg', 'Marginal', '#2196F3'),
            ('join', 'Joint', '#FF9800')]):
        box_data = []
        positions = []
        for k, n_nt in enumerate(nt_vals):
            subset = [r for r in valid if r['n_nonterminals'] == n_nt]
            deltas = [r[f'{mode}_auto_delta'] for r in subset
                      if r.get(f'{mode}_auto_delta') is not None]
            if deltas:
                box_data.append(deltas)
                positions.append(k * 3 + j)
        if box_data:
            bp = ax.boxplot(box_data, positions=positions, widths=0.8,
                            patch_artist=True, showmeans=True,
                            meanprops=dict(marker='D', markerfacecolor='red',
                                           markersize=4))
            for box in bp['boxes']:
                box.set_facecolor(color)
                box.set_alpha(0.6)

    tick_positions = [k * 3 + 0.5 for k in range(len(nt_vals))]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f'{nt} NTs' for nt in nt_vals])
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_ylabel('Delta (detected - true)')
    ax.set_title('Auto-detection bias')
    # Manual legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor='#2196F3', alpha=0.6, label='Marginal'),
        Patch(facecolor='#FF9800', alpha=0.6, label='Joint'),
    ])
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare marginal vs joint context features for NMF')
    parser.add_argument('output', help='Output JSON file for results')
    parser.add_argument('--plot', help='Output plot file (e.g. results.png)')
    parser.add_argument('--repeats', type=int, default=5,
                        help='Number of random grammars per config (default 5)')
    parser.add_argument('--quick', action='store_true',
                        help='Run a smaller grid for quick testing')
    args = parser.parse_args()

    results = run_all_experiments(args)

    # Save results (drop large lists)
    save_results = []
    for r in results:
        sr = dict(r)
        for key in list(sr.keys()):
            if 'kernels' in key or key == 'traceback':
                sr.pop(key, None)
        save_results.append(sr)

    with open(args.output, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    print_summary(results)

    if args.plot:
        plot_results(results, args.plot)
