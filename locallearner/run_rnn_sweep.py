#!/usr/bin/env python3
"""Run RNN model pipeline across all grammars and compare to positional results."""
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
MAX_PARALLEL = 2  # RNN shares MPS GPU, limit contention
N_EPOCHS = 3      # 3 epochs is sufficient (matches 10-epoch ppl)

# Find all grammar directories
grammar_ids = sorted([
    os.path.basename(d) for d in glob.glob(os.path.join(BASE, 'g*'))
    if os.path.isdir(d) and os.path.exists(os.path.join(d, 'grammar.pcfg'))
])

print(f"Found {len(grammar_ids)} grammars: {grammar_ids}")
print(f"Model: RNN, {MAX_PARALLEL}-way parallel")
print()

# Build jobs
jobs = []
for gid in grammar_ids:
    cmd = [PYTHON, WORKER, gid, '--base', BASE, '--k', '0',
           '--model_type', 'rnn', '--n_epochs', str(N_EPOCHS)]
    jobs.append((gid, cmd))

print(f"Total jobs: {len(jobs)}")
print()

# Run with bounded parallelism
t_start = time.time()
running = {}
pending = list(jobs)
completed = 0

while pending or running:
    while pending and len(running) < MAX_PARALLEL:
        gid, cmd = pending.pop(0)
        log_path = os.path.join(BASE, gid, 'pipeline_rnn.log')
        log_f = open(log_path, 'w')
        proc = subprocess.Popen(
            cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=SCRIPT_DIR)
        running[gid] = (proc, time.time(), log_f)
        print(f'[START] {gid} (pid={proc.pid}, {len(running)} running, '
              f'{len(pending)} pending)')

    done = []
    for gid, (proc, t0, log_f) in running.items():
        ret = proc.poll()
        if ret is not None:
            elapsed = time.time() - t0
            log_f.close()
            completed += 1
            result_path = os.path.join(BASE, gid,
                                        'gold_pipeline_results_rnn_k0.json')
            if ret == 0 and os.path.exists(result_path):
                with open(result_path) as f:
                    r = json.load(f)
                print(f'[DONE {completed}/{len(jobs)}] {gid} ({elapsed:.0f}s) '
                      f'init={r["init"]["labeled_micro"]:.3f} '
                      f'sgd={r["sgd"]["labeled_micro"]:.3f}')
            else:
                print(f'[FAIL {completed}/{len(jobs)}] {gid} ({elapsed:.0f}s) '
                      f'rc={ret}')
            done.append(gid)

    for gid in done:
        del running[gid]

    if running and not done:
        time.sleep(10)

total_time = time.time() - t_start
print(f'\nAll {len(jobs)} jobs completed in {total_time:.0f}s')

# Collect RNN results
print('\n=== Collecting results ===')
rnn_results = []
for gid in grammar_ids:
    path = os.path.join(BASE, gid, 'gold_pipeline_results_rnn_k0.json')
    if os.path.exists(path):
        with open(path) as f:
            rnn_results.append(json.load(f))

print(f"  RNN: {len(rnn_results)} results")

# Load positional results for comparison
pos_path = os.path.join(BASE, 'positional_k_sweep_results.json')
if os.path.exists(pos_path):
    with open(pos_path) as f:
        pos_results = json.load(f)
else:
    pos_results = {}
    print("  WARNING: positional results not found")

# Save RNN results
rnn_path = os.path.join(BASE, 'rnn_sweep_results.json')
with open(rnn_path, 'w') as f:
    json.dump(rnn_results, f, indent=2)
print(f"Saved to {rnn_path}")

# Print comparison table
print(f'\n{"Grammar":<8}  '
      f'{"RNN init":>10} {"RNN sgd":>10}  '
      f'{"Pos k=2 init":>12} {"Pos k=2 sgd":>12}  '
      f'{"Target":>8}')
print('-' * 80)

for gid in grammar_ids:
    r_rnn = next((x for x in rnn_results if x['grammar_id'] == gid), None)
    r_pos = next((x for x in pos_results.get('2', [])
                  if x['grammar_id'] == gid), None)

    line = f'{gid:<8}  '
    if r_rnn:
        line += f'{r_rnn["init"]["labeled_micro"]:>10.4f} '
        line += f'{r_rnn["sgd"]["labeled_micro"]:>10.4f}  '
    else:
        line += f'{"N/A":>10} {"N/A":>10}  '

    if r_pos:
        line += f'{r_pos["init"]["labeled_micro"]:>12.4f} '
        line += f'{r_pos["sgd"]["labeled_micro"]:>12.4f}  '
    else:
        line += f'{"N/A":>12} {"N/A":>12}  '

    if r_rnn:
        line += f'{r_rnn["target"]["labeled_micro"]:>8.4f}'
    elif r_pos:
        line += f'{r_pos["target"]["labeled_micro"]:>8.4f}'
    print(line)

# Plot comparison
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Gather data for all model types
model_configs = [('1', 'Pos k=1'), ('2', 'Pos k=2'), ('3', 'Pos k=3')]
colors_init = ['#e8a87c', '#d5a6e6', '#85c1e9', '#a8e6cf']
colors_sgd = ['#f0b27a', '#bb8fce', '#5dade2', '#66cdaa']
colors_target = '#a8d5e2'

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f'RNN vs Positional Models ({len(grammar_ids)} grammars)',
             fontsize=14)

for ax, (key, title) in zip(axes, [
    ('labeled_micro', 'Labeled Micro-averaged'),
    ('labeled_exact', 'Labeled Exact Match'),
]):
    target_vals = []
    init_data = {}
    sgd_data = {}

    for k_str, label in model_configs:
        init_data[label] = [r['init'][key] for r in pos_results.get(k_str, [])]
        sgd_data[label] = [r['sgd'][key] for r in pos_results.get(k_str, [])]

    init_data['RNN'] = [r['init'][key] for r in rnn_results]
    sgd_data['RNN'] = [r['sgd'][key] for r in rnn_results]
    target_vals = [r['target'][key] for r in rnn_results]

    labels_all = ['Target'] + [f'Init\n{l}' for _, l in model_configs] + ['Init\nRNN'] + \
                 [f'SGD\n{l}' for _, l in model_configs] + ['SGD\nRNN']
    data_all = [target_vals] + \
               [init_data[l] for _, l in model_configs] + [init_data['RNN']] + \
               [sgd_data[l] for _, l in model_configs] + [sgd_data['RNN']]

    positions = [1, 2.5, 3.5, 4.5, 5.5, 7.5, 8.5, 9.5, 10.5]
    box_colors = [colors_target] + colors_init + colors_sgd

    bp = ax.boxplot(data_all, positions=positions, widths=0.7,
                    patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='black',
                                   markersize=4))

    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels_all, fontsize=7)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')

    all_vals = [v for d in data_all for v in d if d]
    if all_vals:
        ymin = max(0.5, min(all_vals) - 0.05)
        ymax = min(1.02, max(all_vals) + 0.02)
        ax.set_ylim(ymin, ymax)

plt.tight_layout()
plot_path = os.path.join(BASE, 'rnn_vs_positional_boxplot.png')
plt.savefig(plot_path, dpi=150)
print(f'\nPlot saved to {plot_path}')
