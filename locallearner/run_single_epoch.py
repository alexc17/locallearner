#!/usr/bin/env python3
"""Run a single SGD epoch with given hyperparameters and write results to JSON."""
import argparse
import json
import sys
import time

sys.path.insert(0, '.')
import wcfg
import evaluation

parser = argparse.ArgumentParser(description='Single SGD epoch experiment')
parser.add_argument('--base', type=str, required=True, help='Base directory for grammar files')
parser.add_argument('--stepsize', type=float, required=True)
parser.add_argument('--maxcount', type=int, required=True)
parser.add_argument('--maxlength', type=int, default=15)
parser.add_argument('--output', type=str, required=True, help='Output JSON file')
args = parser.parse_args()

base = args.base.rstrip('/') + '/'
target = wcfg.load_wcfg_from_file(base + 'grammar.pcfg')
grammar = wcfg.load_wcfg_from_file(base + 'gold_init.pcfg')

# Load corpus
corpus = []
with open(base + 'corpus.txt') as f:
    for line in f:
        s = tuple(line.split())
        if s:
            corpus.append(s)

# Initial metrics
init_elen = grammar.expected_length()
init_kld = evaluation.smoothed_kld_exact(target, grammar, compute_bijection=True)

# Run one SGD epoch
t0 = time.time()
result = grammar.estimate_inside_outside_from_list(
    corpus, maxlength=args.maxlength, maxcount=args.maxcount, stepsize=args.stepsize)
result.set_log_parameters()
sgd_time = time.time() - t0

# Evaluate
t0 = time.time()
kld = evaluation.smoothed_kld_exact(target, result, compute_bijection=True)

eval_kwargs = {'max_length': 20, 'seed': 42, 'samples': 1000}
h = result.copy()
h.renormalise_locally()
h.set_log_parameters()
mapping = evaluation.estimate_bijection(target, h, **eval_kwargs)
h_relabeled = h.relabel({a: b for b, a in mapping.items()})

scores = evaluation.do_parseval_monte_carlo(target, [h_relabeled], **eval_kwargs)
denom = scores['trees_denominator']
lab_d = scores['labeled_denominator']
ulab_d = scores['unlabeled_denominator']

eval_time = time.time() - t0

out = {
    'stepsize': args.stepsize,
    'maxcount': args.maxcount,
    'maxlength': args.maxlength,
    'init_elen': init_elen,
    'init_kld': init_kld,
    'elen': result.expected_length(),
    'kld': kld,
    'terminals': len(result.terminals),
    'labeled_exact': scores['original:hypothesis0:labeled:exact_match'] / denom,
    'unlabeled_exact': scores['original:hypothesis0:unlabeled:exact_match'] / denom,
    'labeled_micro': scores['original:hypothesis0:labeled:microaveraged'] / lab_d,
    'unlabeled_micro': scores['original:hypothesis0:unlabeled:microaveraged'] / ulab_d,
    'sgd_time': sgd_time,
    'eval_time': eval_time,
}

with open(args.output, 'w') as f:
    json.dump(out, f, indent=2)

print(f"stepsize={args.stepsize} maxcount={args.maxcount} maxlength={args.maxlength} "
      f"KLD={kld:.4f} lab_exact={out['labeled_exact']:.4f} "
      f"lab_micro={out['labeled_micro']:.4f} E[len]={out['elen']:.3f} "
      f"time={sgd_time:.0f}s")
