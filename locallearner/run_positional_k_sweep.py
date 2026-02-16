#!/usr/bin/env python3
"""Run positional model pipeline for k=1,2,3 across all grammars, then plot comparison."""
import subprocess
import os
import sys
import time
import json
import glob

sys.stdout.reconfigure(line_buffering=True)

BASE = '/Users/alexc/research/cursor_locallearner/locallearner/tmp/jm_experiment'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(SCRIPT_DIR, 'run_gold_pipeline.py')
PYTHON = '/opt/anaconda3/bin/python3'
MAX_PARALLEL = 10

K_VALUES = [1, 2, 3]

# Find all grammar directories
grammar_ids = sorted([
    os.path.basename(d) for d in glob.glob(os.path.join(BASE, 'g*'))
    if os.path.isdir(d) and os.path.exists(os.path.join(d, 'grammar.pcfg'))
])

print(f"Found {len(grammar_ids)} grammars: {grammar_ids}")
print(f"K values: {K_VALUES}, model=positional")
print(f"Running {MAX_PARALLEL}-way parallel")
print()

# Build all jobs: (grammar_id, k)
jobs = []
for k in K_VALUES:
    for gid in grammar_ids:
        cmd = [PYTHON, WORKER, gid, '--base', BASE, '--k', str(k),
               '--model_type', 'positional']
        jobs.append((f'{gid}_k{k}', gid, k, cmd))

print(f"Total jobs: {len(jobs)}")
print()

# Run with bounded parallelism
t_start = time.time()
running = {}
pending = list(jobs)
completed = 0

while pending or running:
    while pending and len(running) < MAX_PARALLEL:
        job_id, gid, k, cmd = pending.pop(0)
        log_path = os.path.join(BASE, gid, f'pipeline_pos_k{k}.log')
        log_f = open(log_path, 'w')
        proc = subprocess.Popen(
            cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=SCRIPT_DIR)
        running[job_id] = (proc, time.time(), log_f, gid, k)
        print(f'[START] {job_id} (pid={proc.pid}, {len(running)} running, {len(pending)} pending)')

    done = []
    for job_id, (proc, t0, log_f, gid, k) in running.items():
        ret = proc.poll()
        if ret is not None:
            elapsed = time.time() - t0
            log_f.close()
            completed += 1
            result_path = os.path.join(BASE, gid, f'gold_pipeline_results_pos_k{k}.json')
            if ret == 0 and os.path.exists(result_path):
                with open(result_path) as f:
                    r = json.load(f)
                print(f'[DONE {completed}/{len(jobs)}] {job_id} ({elapsed:.0f}s) '
                      f'init={r["init"]["labeled_micro"]:.3f} '
                      f'sgd={r["sgd"]["labeled_micro"]:.3f}')
            else:
                print(f'[FAIL {completed}/{len(jobs)}] {job_id} ({elapsed:.0f}s) rc={ret}')
            done.append(job_id)

    for job_id in done:
        del running[job_id]

    if running and not done:
        time.sleep(5)

total_time = time.time() - t_start
print(f'\nAll {len(jobs)} jobs completed in {total_time:.0f}s')

# Collect results
print('\n=== Collecting results ===')
all_results = {}  # k -> list of results
for k in K_VALUES:
    all_results[k] = []
    for gid in grammar_ids:
        path = os.path.join(BASE, gid, f'gold_pipeline_results_pos_k{k}.json')
        if os.path.exists(path):
            with open(path) as f:
                all_results[k].append(json.load(f))

for k in K_VALUES:
    print(f"  k={k}: {len(all_results[k])} results")

# Save collected results
collected_path = os.path.join(BASE, 'positional_k_sweep_results.json')
# Convert keys to strings for JSON
with open(collected_path, 'w') as f:
    json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)
print(f"Saved to {collected_path}")

# Print summary table
print(f'\n{"Grammar":<8}', end='')
for k in K_VALUES:
    print(f'  {"Init k="+str(k):>10} {"SGD k="+str(k):>10}', end='')
print(f'  {"Target":>8}')
print('-' * (8 + len(K_VALUES) * 22 + 10))

for gid in grammar_ids:
    print(f'{gid:<8}', end='')
    for k in K_VALUES:
        r = next((x for x in all_results[k] if x['grammar_id'] == gid), None)
        if r:
            print(f'  {r["init"]["labeled_micro"]:>10.4f} {r["sgd"]["labeled_micro"]:>10.4f}', end='')
        else:
            print(f'  {"N/A":>10} {"N/A":>10}', end='')
    # Target from any k
    r0 = next((x for k in K_VALUES for x in all_results[k] if x['grammar_id'] == gid), None)
    if r0:
        print(f'  {r0["target"]["labeled_micro"]:>8.4f}')
    else:
        print()

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

colors_init = ['#e8a87c', '#d5a6e6', '#85c1e9']
colors_sgd = ['#f0b27a', '#bb8fce', '#5dade2']
colors_target = '#a8d5e2'

n_grammars = len(grammar_ids)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Positional Model: k=1 vs k=2 vs k=3 ({n_grammars} grammars)', fontsize=14)

for ax, (key, title) in zip(axes.flat, metrics):
    # Gather data: target, init_k1, init_k2, init_k3, sgd_k1, sgd_k2, sgd_k3
    target_vals = []
    init_data = {k: [] for k in K_VALUES}
    sgd_data = {k: [] for k in K_VALUES}

    for gid in grammar_ids:
        for k in K_VALUES:
            r = next((x for x in all_results[k] if x['grammar_id'] == gid), None)
            if r:
                init_data[k].append(r['init'][key])
                sgd_data[k].append(r['sgd'][key])
                if k == K_VALUES[0]:
                    target_vals.append(r['target'][key])

    # Box positions: target, init_k1, init_k2, init_k3, sgd_k1, sgd_k2, sgd_k3
    positions = [1, 2.5, 3.5, 4.5, 6, 7, 8]
    data = [target_vals] + [init_data[k] for k in K_VALUES] + [sgd_data[k] for k in K_VALUES]
    box_colors = [colors_target] + colors_init + colors_sgd

    bp = ax.boxplot(data, positions=positions, widths=0.7,
                    patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='black', markersize=4))

    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticks(positions)
    ax.set_xticklabels(['Target', 'Init\nk=1', 'Init\nk=2', 'Init\nk=3',
                        'SGD\nk=1', 'SGD\nk=2', 'SGD\nk=3'], fontsize=8)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')

    all_vals = [v for d in data for v in d]
    if all_vals:
        ymin = max(0.5, min(all_vals) - 0.05)
        ymax = min(1.02, max(all_vals) + 0.02)
        ax.set_ylim(ymin, ymax)

plt.tight_layout()
plot_path = os.path.join(BASE, 'positional_k_sweep_boxplot.png')
plt.savefig(plot_path, dpi=150)
print(f'\nPlot saved to {plot_path}')

# Also plot KLD
fig2, ax2 = plt.subplots(figsize=(8, 5))
init_kld_data = {k: [r['init']['kld'] for r in all_results[k]] for k in K_VALUES}
sgd_kld_data = {k: [r['sgd']['kld'] for r in all_results[k]] for k in K_VALUES}

positions_kld = [1, 2, 3, 4.5, 5.5, 6.5]
data_kld = [init_kld_data[k] for k in K_VALUES] + [sgd_kld_data[k] for k in K_VALUES]
kld_colors = colors_init + colors_sgd

bp2 = ax2.boxplot(data_kld, positions=positions_kld, widths=0.7,
                  patch_artist=True, showmeans=True,
                  meanprops=dict(marker='D', markerfacecolor='black', markersize=4))
for patch, color in zip(bp2['boxes'], kld_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

ax2.set_xticks(positions_kld)
ax2.set_xticklabels(['Init k=1', 'Init k=2', 'Init k=3',
                     'SGD k=1', 'SGD k=2', 'SGD k=3'], fontsize=9)
ax2.set_ylabel('Smoothed KLD')
ax2.set_title(f'KLD: Positional Model k=1,2,3 ({n_grammars} grammars)')
ax2.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
kld_path = os.path.join(BASE, 'positional_k_sweep_kld.png')
plt.savefig(kld_path, dpi=150)
print(f'KLD plot saved to {kld_path}')
