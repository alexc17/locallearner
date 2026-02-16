#!/usr/bin/env python3
"""Plot heatmaps from hyperparameter sweep results."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/Users/alexc/research/cursor_locallearner/locallearner/tmp/jm_experiment/g000/'
results_path = os.path.join(BASE, 'sweep_results', 'all_results.json')
out_path = os.path.join(BASE, 'sweep_results', 'sweep_heatmaps.png')

with open(results_path) as f:
    results = json.load(f)

# Build grid
stepsizes = sorted(set(r['stepsize'] for r in results))
maxcounts = sorted(set(r['maxcount'] for r in results))

lookup = {}
for r in results:
    lookup[(r['stepsize'], r['maxcount'])] = r

metrics = [
    ('kld', 'Smoothed KLD', 'Reds_r', True),
    ('labeled_exact', 'Labeled Exact Match', 'Blues', False),
    ('labeled_micro', 'Labeled Micro-avg', 'Greens', False),
    ('elen', 'Expected Length', 'Purples', False),
]

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle('Single-Epoch SGD Sweep: stepsize × maxcount (maxlength=15)', fontsize=14)

for ax, (key, title, cmap, lower_better) in zip(axes.flat, metrics):
    grid = np.zeros((len(stepsizes), len(maxcounts)))
    for i, ss in enumerate(stepsizes):
        for j, mc in enumerate(maxcounts):
            r = lookup.get((ss, mc))
            grid[i, j] = r[key] if r else np.nan

    im = ax.imshow(grid, aspect='auto', cmap=cmap, origin='lower')
    ax.set_xticks(range(len(maxcounts)))
    ax.set_xticklabels([str(mc) for mc in maxcounts])
    ax.set_yticks(range(len(stepsizes)))
    ax.set_yticklabels([str(ss) for ss in stepsizes])
    ax.set_xlabel('maxcount (batch size)')
    ax.set_ylabel('stepsize (learning rate)')
    ax.set_title(title)

    # Annotate cells
    for i in range(len(stepsizes)):
        for j in range(len(maxcounts)):
            val = grid[i, j]
            if not np.isnan(val):
                fmt = f'{val:.3f}' if key != 'elen' else f'{val:.2f}'
                color = 'white' if im.norm(val) > 0.6 else 'black'
                ax.text(j, i, fmt, ha='center', va='center',
                        fontsize=8, fontweight='bold', color=color)

    plt.colorbar(im, ax=ax, shrink=0.8)

    # Mark best cell
    if lower_better:
        best_idx = np.unravel_index(np.nanargmin(grid), grid.shape)
    else:
        best_idx = np.unravel_index(np.nanargmax(grid), grid.shape)
    ax.plot(best_idx[1], best_idx[0], 'r*', markersize=15, markeredgecolor='red',
            markerfacecolor='none', markeredgewidth=2)

plt.tight_layout()
plt.savefig(out_path, dpi=150)
print(f'Plot saved to {out_path}')
