#!/usr/bin/env python3
"""Run gold pipeline for all grammars in parallel, then plot results."""
import subprocess
import os
import sys
import time
import json
import glob
import argparse

sys.stdout.reconfigure(line_buffering=True)

BASE = '/Users/alexc/research/cursor_locallearner/locallearner/tmp/jm_experiment'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(SCRIPT_DIR, 'run_gold_pipeline.py')
PYTHON = '/opt/anaconda3/bin/python3'

parser = argparse.ArgumentParser(description='Run gold pipeline for all grammars')
parser.add_argument('--k', type=int, default=2, help='Context width for neural models')
parser.add_argument('--parallel', type=int, default=8, help='Max parallel jobs')
args = parser.parse_args()

K = args.k
MAX_PARALLEL = args.parallel

# Find all grammar directories
grammar_ids = sorted([
    os.path.basename(d) for d in glob.glob(os.path.join(BASE, 'g*'))
    if os.path.isdir(d) and os.path.exists(os.path.join(d, 'grammar.pcfg'))
])

print(f"Found {len(grammar_ids)} grammars: {grammar_ids}")
print(f"Running {MAX_PARALLEL}-way parallel, k={K}")
print()

# Build jobs
jobs = []
for gid in grammar_ids:
    cmd = [PYTHON, WORKER, gid, '--base', BASE, '--k', str(K)]
    jobs.append((gid, cmd))

# Run with bounded parallelism
t_start = time.time()
running = {}
pending = list(jobs)
completed = 0

while pending or running:
    while pending and len(running) < MAX_PARALLEL:
        gid, cmd = pending.pop(0)
        log_path = os.path.join(BASE, gid, f'pipeline_k{K}.log')
        log_f = open(log_path, 'w')
        proc = subprocess.Popen(
            cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=SCRIPT_DIR)
        running[gid] = (proc, time.time(), log_f)
        print(f'[START] {gid} (pid={proc.pid}, {len(running)} running, {len(pending)} pending)')

    done = []
    for gid, (proc, t0, log_f) in running.items():
        ret = proc.poll()
        if ret is not None:
            elapsed = time.time() - t0
            log_f.close()
            completed += 1
            # Check if results exist
            result_path = os.path.join(BASE, gid, f'gold_pipeline_results_k{K}.json')
            if ret == 0 and os.path.exists(result_path):
                with open(result_path) as f:
                    r = json.load(f)
                print(f'[DONE {completed}/{len(jobs)}] {gid} ({elapsed:.0f}s) '
                      f'lab_micro: target={r["target"]["labeled_micro"]:.3f} '
                      f'init={r["init"]["labeled_micro"]:.3f} '
                      f'sgd={r["sgd"]["labeled_micro"]:.3f}')
            else:
                print(f'[FAIL {completed}/{len(jobs)}] {gid} ({elapsed:.0f}s) rc={ret}')
            done.append(gid)

    for gid in done:
        del running[gid]

    if running and not done:
        time.sleep(5)

total_time = time.time() - t_start
print(f'\nAll {len(jobs)} jobs completed in {total_time:.0f}s')

# Collect results and plot
print('\n=== Collecting results ===')
all_results = []
for gid in grammar_ids:
    path = os.path.join(BASE, gid, f'gold_pipeline_results_k{K}.json')
    if os.path.exists(path):
        with open(path) as f:
            all_results.append(json.load(f))

if not all_results:
    print("No results found!")
    sys.exit(1)

# Save collected results
collected_path = os.path.join(BASE, f'gold_pipeline_all_results_k{K}.json')
with open(collected_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"Saved {len(all_results)} results to {collected_path}")

# Print summary table
print(f'\n{"Grammar":<8} {"Target":>8} {"Init":>8} {"SGD":>8}  (labeled micro-avg)')
print('-' * 40)
for r in all_results:
    print(f'{r["grammar_id"]:<8} {r["target"]["labeled_micro"]:>8.4f} '
          f'{r["init"]["labeled_micro"]:>8.4f} {r["sgd"]["labeled_micro"]:>8.4f}')

# Plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

metrics = [
    ('labeled_exact', 'Labeled Exact Match'),
    ('labeled_micro', 'Labeled Micro-avg'),
    ('unlabeled_exact', 'Unlabeled Exact Match'),
    ('unlabeled_micro', 'Unlabeled Micro-avg'),
]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(f'Gold Kernels + Rényi Init (k={K}) + 1 Epoch C SGD ({len(all_results)} grammars)',
             fontsize=14)

for ax, (key, title) in zip(axes.flat, metrics):
    target_vals = [r['target'][key] for r in all_results]
    init_vals = [r['init'][key] for r in all_results]
    sgd_vals = [r['sgd'][key] for r in all_results]

    positions = [1, 2, 3]
    bp = ax.boxplot([target_vals, init_vals, sgd_vals],
                    positions=positions, widths=0.6,
                    patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='black', markersize=5))

    colors = ['#a8d5e2', '#f9a03f', '#7bc77b']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(['Target\n(ceiling)', 'Init\n(Rényi)', 'SGD\n(1 epoch)'])
    ax.set_title(title)
    ax.set_ylabel(title)
    ax.grid(True, alpha=0.3, axis='y')

    # Set y range to focus on differences
    all_vals = target_vals + init_vals + sgd_vals
    ymin = max(0.5, min(all_vals) - 0.05)
    ymax = min(1.0, max(all_vals) + 0.02)
    ax.set_ylim(ymin, ymax)

plt.tight_layout()
plot_path = os.path.join(BASE, f'gold_pipeline_boxplot_k{K}.png')
plt.savefig(plot_path, dpi=150)
print(f'\nPlot saved to {plot_path}')

# Also plot KLD
fig2, ax2 = plt.subplots(figsize=(7, 5))
init_kld = [r['init']['kld'] for r in all_results]
sgd_kld = [r['sgd']['kld'] for r in all_results]

bp2 = ax2.boxplot([init_kld, sgd_kld],
                  positions=[1, 2], widths=0.6,
                  patch_artist=True, showmeans=True,
                  meanprops=dict(marker='D', markerfacecolor='black', markersize=5))
bp2['boxes'][0].set_facecolor('#f9a03f')
bp2['boxes'][0].set_alpha(0.7)
bp2['boxes'][1].set_facecolor('#7bc77b')
bp2['boxes'][1].set_alpha(0.7)

ax2.set_xticks([1, 2])
ax2.set_xticklabels(['Init (Rényi)', 'SGD (1 epoch)'])
ax2.set_ylabel('Smoothed KLD')
ax2.set_title(f'KLD (target || hypothesis) — k={K}, {len(all_results)} grammars')
ax2.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
kld_path = os.path.join(BASE, f'gold_pipeline_kld_boxplot_k{K}.png')
plt.savefig(kld_path, dpi=150)
print(f'KLD plot saved to {kld_path}')
