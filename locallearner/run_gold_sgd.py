#!/usr/bin/env python3
"""Run SGD epochs on gold-kernel initialized grammar and evaluate after each epoch."""
import sys, time
sys.stdout.reconfigure(line_buffering=True)

import wcfg, evaluation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base = '/Users/alexc/research/cursor_locallearner/locallearner/tmp/jm_experiment/g000/'
target = wcfg.load_wcfg_from_file(base + 'grammar.pcfg')

# Load corpus
corpus = []
with open(base + 'corpus.txt') as f:
    for line in f:
        s = tuple(line.split())
        if s:
            corpus.append(s)

# Evaluation helper
eval_kwargs = {'max_length': 20, 'seed': 42, 'samples': 1000}

def evaluate_grammar(hyp, label=""):
    """Evaluate hypothesis against target, return dict of metrics."""
    h = hyp.copy()
    h.renormalise_locally()
    h.set_log_parameters()
    mapping = evaluation.estimate_bijection(target, h, **eval_kwargs)
    h_relabeled = h.relabel({a: b for b, a in mapping.items()})
    
    scores = evaluation.do_parseval_monte_carlo(target, [h_relabeled], **eval_kwargs)
    denom = scores['trees_denominator']
    lab_d = scores['labeled_denominator']
    ulab_d = scores['unlabeled_denominator']
    
    kld = evaluation.smoothed_kld_exact(target, h, compute_bijection=True)
    
    return {
        'kld': kld,
        'labeled_exact': scores['original:hypothesis0:labeled:exact_match'] / denom,
        'unlabeled_exact': scores['original:hypothesis0:unlabeled:exact_match'] / denom,
        'labeled_micro': scores['original:hypothesis0:labeled:microaveraged'] / lab_d,
        'unlabeled_micro': scores['original:hypothesis0:unlabeled:microaveraged'] / ulab_d,
        'elen': hyp.expected_length(),
        'terminals': len(hyp.terminals),
    }

# Get target ceiling
target_scores = evaluation.do_parseval_monte_carlo(target, [], **eval_kwargs)
target_denom = target_scores['trees_denominator']
target_lab_d = target_scores['labeled_denominator']
target_ulab_d = target_scores['unlabeled_denominator']
target_ceiling = {
    'labeled_exact': target_scores['original:gold viterbi:labeled:exact_match'] / target_denom,
    'unlabeled_exact': target_scores['original:gold viterbi:unlabeled:exact_match'] / target_denom,
    'labeled_micro': target_scores['original:gold viterbi:labeled:microaveraged'] / target_lab_d,
    'unlabeled_micro': target_scores['original:gold viterbi:unlabeled:microaveraged'] / target_ulab_d,
}
print(f"Target ceiling: lab_exact={target_ceiling['labeled_exact']:.4f}  "
      f"lab_micro={target_ceiling['labeled_micro']:.4f}")

# Start from init grammar
grammar = wcfg.load_wcfg_from_file(base + 'gold_init.pcfg')

epochs = []
results = []

# Epoch 0 = init
print(f'\n{"Epoch":>5}  {"E[len]":>7}  {"Terms":>5}  {"KLD":>8}  '
      f'{"LabExact":>8}  {"LabMicro":>8}  {"Time":>6}')

t0 = time.time()
r = evaluate_grammar(grammar, "init")
elapsed = time.time() - t0
epochs.append(0)
results.append(r)
print(f'    0  {r["elen"]:>7.3f}  {r["terminals"]:>5}  {r["kld"]:>8.4f}  '
      f'{r["labeled_exact"]:>8.4f}  {r["labeled_micro"]:>8.4f}  {elapsed:>5.0f}s')

# SGD epochs
for epoch in range(1, 11):
    t0 = time.time()
    grammar = grammar.estimate_inside_outside_from_list(
        corpus, maxlength=10, maxcount=5000, stepsize=0.5)
    grammar.set_log_parameters()
    t_sgd = time.time() - t0
    
    t0 = time.time()
    r = evaluate_grammar(grammar, f"SGD{epoch}")
    t_eval = time.time() - t0
    
    epochs.append(epoch)
    results.append(r)
    print(f'   {epoch:>2}  {r["elen"]:>7.3f}  {r["terminals"]:>5}  {r["kld"]:>8.4f}  '
          f'{r["labeled_exact"]:>8.4f}  {r["labeled_micro"]:>8.4f}  '
          f'{t_sgd:>4.0f}+{t_eval:>3.0f}s')

grammar.store(base + 'gold_sgd10.pcfg')
print(f'\nSaved gold_sgd10.pcfg')

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('Gold Kernels + Rényi Init + SGD (g000)', fontsize=14)

ep = epochs

# KLD
ax = axes[0, 0]
ax.plot(ep, [r['kld'] for r in results], 'o-', color='tab:red', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Smoothed KLD')
ax.set_title('KLD (target || hypothesis)')
ax.grid(True, alpha=0.3)

# Labeled exact match
ax = axes[0, 1]
ax.plot(ep, [r['labeled_exact'] for r in results], 'o-', color='tab:blue', linewidth=2, label='Hypothesis')
ax.axhline(target_ceiling['labeled_exact'], color='tab:blue', linestyle='--', alpha=0.6, label=f'Target ceiling ({target_ceiling["labeled_exact"]:.3f})')
ax.set_xlabel('Epoch')
ax.set_ylabel('Labeled Exact Match')
ax.set_title('Labeled Exact Match')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.7, 1.0)

# Labeled micro-avg
ax = axes[1, 0]
ax.plot(ep, [r['labeled_micro'] for r in results], 's-', color='tab:green', linewidth=2, label='Hypothesis')
ax.axhline(target_ceiling['labeled_micro'], color='tab:green', linestyle='--', alpha=0.6, label=f'Target ceiling ({target_ceiling["labeled_micro"]:.3f})')
ax.set_xlabel('Epoch')
ax.set_ylabel('Labeled Micro-avg')
ax.set_title('Labeled Micro-averaged Accuracy')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.85, 1.0)

# Expected length
ax = axes[1, 1]
ax.plot(ep, [r['elen'] for r in results], 'D-', color='tab:purple', linewidth=2, label='Hypothesis')
ax.axhline(target.expected_length(), color='tab:purple', linestyle='--', alpha=0.6, label=f'Target ({target.expected_length():.2f})')
ax.set_xlabel('Epoch')
ax.set_ylabel('Expected Length')
ax.set_title('Expected Length')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
outpath = base + 'gold_sgd_progress.png'
plt.savefig(outpath, dpi=150)
print(f'\nPlot saved to {outpath}')
