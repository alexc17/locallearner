#!/usr/bin/env python3
"""Sweep alpha values for RNN model across all grammars.

Reuses cached RNN models (normal + gap). Only re-estimates xi parameters
and re-evaluates for each alpha, so much faster than retraining.
"""
import os
import sys
import math
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from scipy.optimize import linear_sum_assignment

import wcfg
import evaluation
import neural_learner as nl_module
from run_sgd_io import run_epoch
from run_gold_pipeline import get_gold_kernels

BASE = '/Users/alexc/research/cursor_locallearner/locallearner/tmp/jm_experiment'
ALPHAS = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, float('inf')]

import glob
grammar_ids = sorted([
    os.path.basename(d) for d in glob.glob(os.path.join(BASE, 'g*'))
    if os.path.isdir(d) and os.path.exists(os.path.join(d, 'grammar.pcfg'))
])

print(f"Grammars: {grammar_ids}")
print(f"Alphas: {ALPHAS}")
print()

all_results = {}  # alpha -> list of results

for alpha in ALPHAS:
    alpha_label = 'inf' if alpha == float('inf') else str(alpha)
    print(f"\n{'='*60}")
    print(f"Alpha = {alpha_label}")
    print(f"{'='*60}")

    results_for_alpha = []

    for gid in grammar_ids:
        gdir = os.path.join(BASE, gid)
        grammar_path = os.path.join(gdir, 'grammar.pcfg')
        corpus_path = os.path.join(gdir, 'corpus.txt')

        target = wcfg.load_wcfg_from_file(grammar_path)
        gold_anchors = get_gold_kernels(target)

        # Set up NeuralLearner, load cached models
        nl = nl_module.NeuralLearner(corpus_path)
        nl.k = 0
        nl.model_type = 'rnn'
        nl.alpha = alpha
        nl.single_model_file = os.path.join(gdir, 'rnn_cloze.pt')
        nl.gap_model_file = os.path.join(gdir, 'rnn_gap_cloze.pt')

        # Load cached models
        nl.train_single_model(verbose=False)
        nl.train_pair_model(verbose=False)

        # Set gold anchors
        nl.anchors = gold_anchors
        nl.nonterminals = ['S'] + [f'NT_{w}' for w in gold_anchors]
        nl.anchor2nt = {w: f'NT_{w}' for w in gold_anchors}

        # Estimate xi with this alpha
        t0 = time.time()
        nl.estimate_lexical_xi(verbose=False)
        nl.estimate_binary_xi(verbose=False)
        nl.build_wcfg(verbose=False)
        init_pcfg = nl.convert_to_pcfg(verbose=False)
        xi_time = time.time() - t0

        # One epoch of SGD
        corpus_100k = os.path.join(gdir, 'corpus_100k.txt')
        sgd_pcfg = run_epoch(
            init_pcfg, corpus_100k,
            maxlength=15, maxcount=1000, stepsize=0.5,
            update='linear', verbose=0)

        # Evaluate
        eval_kwargs = {'max_length': 20, 'seed': 42, 'samples': 1000}

        def evaluate(hyp):
            h = hyp.copy()
            h.renormalise_locally()
            h.set_log_parameters()
            mapping = evaluation.estimate_bijection(target, h, **eval_kwargs)
            h_relab = h.relabel({a: b for b, a in mapping.items()})
            scores = evaluation.do_parseval_monte_carlo(
                target, [h_relab], **eval_kwargs)
            denom = scores['trees_denominator']
            lab_d = scores['labeled_denominator']
            ulab_d = scores['unlabeled_denominator']
            kld = evaluation.smoothed_kld_exact(
                target, h, compute_bijection=True)
            return {
                'kld': kld,
                'labeled_exact': (
                    scores['original:hypothesis0:labeled:exact_match'] / denom),
                'unlabeled_exact': (
                    scores['original:hypothesis0:unlabeled:exact_match'] / denom),
                'labeled_micro': (
                    scores['original:hypothesis0:labeled:microaveraged'] / lab_d),
                'unlabeled_micro': (
                    scores['original:hypothesis0:unlabeled:microaveraged']
                    / ulab_d),
                'elen': hyp.expected_length(),
            }

        init_scores = evaluate(init_pcfg)
        sgd_scores = evaluate(sgd_pcfg)

        results_for_alpha.append({
            'grammar_id': gid,
            'alpha': alpha_label,
            'init': init_scores,
            'sgd': sgd_scores,
            'xi_time': xi_time,
        })

        print(f"  {gid}: init={init_scores['labeled_micro']:.4f} "
              f"sgd={sgd_scores['labeled_micro']:.4f} "
              f"({xi_time:.1f}s)", flush=True)

    all_results[alpha_label] = results_for_alpha

# Save results
out_path = os.path.join(BASE, 'rnn_alpha_sweep_results.json')
with open(out_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to {out_path}")

# Print summary table
print(f"\n{'Grammar':<8}", end='')
for alpha in ALPHAS:
    a = 'inf' if alpha == float('inf') else str(alpha)
    print(f"  {'a='+a:>10}", end='')
print("  (init labeled_micro)")

print(f"{'Grammar':<8}", end='')
for alpha in ALPHAS:
    a = 'inf' if alpha == float('inf') else str(alpha)
    print(f"  {'a='+a:>10}", end='')
print("  (sgd labeled_micro)")

for gid in grammar_ids:
    line_init = f"{gid:<8}"
    line_sgd = f"{gid:<8}"
    for alpha in ALPHAS:
        a = 'inf' if alpha == float('inf') else str(alpha)
        r = next((x for x in all_results[a] if x['grammar_id'] == gid), None)
        if r:
            line_init += f"  {r['init']['labeled_micro']:>10.4f}"
            line_sgd += f"  {r['sgd']['labeled_micro']:>10.4f}"
        else:
            line_init += f"  {'N/A':>10}"
            line_sgd += f"  {'N/A':>10}"
    print(line_init)

print()
for gid in grammar_ids:
    line_sgd = f"{gid:<8}"
    for alpha in ALPHAS:
        a = 'inf' if alpha == float('inf') else str(alpha)
        r = next((x for x in all_results[a] if x['grammar_id'] == gid), None)
        if r:
            line_sgd += f"  {r['sgd']['labeled_micro']:>10.4f}"
        else:
            line_sgd += f"  {'N/A':>10}"
    print(line_sgd)

# Compute mean across grammars for each alpha
print(f"\n{'Mean':<8}", end='')
for alpha in ALPHAS:
    a = 'inf' if alpha == float('inf') else str(alpha)
    vals = [r['init']['labeled_micro'] for r in all_results[a]]
    print(f"  {np.mean(vals):>10.4f}", end='')
print("  (init)")

print(f"{'Mean':<8}", end='')
for alpha in ALPHAS:
    a = 'inf' if alpha == float('inf') else str(alpha)
    vals = [r['sgd']['labeled_micro'] for r in all_results[a]]
    print(f"  {np.mean(vals):>10.4f}", end='')
print("  (sgd)")

# Plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('RNN Model: Effect of Rényi Alpha (20 grammars)', fontsize=14)

alpha_labels = ['0.5', '1.0', '1.5', '2.0', '3.0', '5.0', 'inf']
x_pos = list(range(len(alpha_labels)))

for ax, (key, title) in zip(axes, [
    ('labeled_micro', 'Labeled Micro-averaged'),
    ('labeled_exact', 'Labeled Exact Match'),
]):
    init_data = []
    sgd_data = []
    for a in alpha_labels:
        init_data.append([r['init'][key] for r in all_results[a]])
        sgd_data.append([r['sgd'][key] for r in all_results[a]])

    # Plot init and sgd as separate box plots
    bp1 = ax.boxplot(init_data, positions=[x - 0.18 for x in x_pos],
                     widths=0.3, patch_artist=True, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='black',
                                    markersize=3))
    bp2 = ax.boxplot(sgd_data, positions=[x + 0.18 for x in x_pos],
                     widths=0.3, patch_artist=True, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='black',
                                    markersize=3))

    for patch in bp1['boxes']:
        patch.set_facecolor('#85c1e9')
        patch.set_alpha(0.7)
    for patch in bp2['boxes']:
        patch.set_facecolor('#82e0aa')
        patch.set_alpha(0.7)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'α={a}' for a in alpha_labels], fontsize=8)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend([bp1['boxes'][0], bp2['boxes'][0]], ['Init', 'SGD'],
              fontsize=8, loc='lower right')

    all_vals = [v for d in init_data + sgd_data for v in d]
    if all_vals:
        ymin = max(0.5, min(all_vals) - 0.05)
        ymax = min(1.02, max(all_vals) + 0.02)
        ax.set_ylim(ymin, ymax)

plt.tight_layout()
plot_path = os.path.join(BASE, 'rnn_alpha_sweep_boxplot.png')
plt.savefig(plot_path, dpi=150)
print(f'\nPlot saved to {plot_path}')
