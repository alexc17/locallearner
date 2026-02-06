#evaluate_kernels.py

import utility
import wcfg
import argparse
import evaluation
import json

parser = argparse.ArgumentParser(description='Evaluate kernels against a target pcfg')

parser.add_argument('target', type=str, help='filename of original (gold) grammar')
parser.add_argument('kernels', help='filename of kernels')
parser.add_argument('--json', help='filename of json file')
parser.add_argument('--corpus', help='corpus file to count terminal occurrences')
parser.add_argument('--verbose', action='store_true', help='Print per-kernel details')
args = parser.parse_args()

target = wcfg.load_wcfg_from_file(args.target)

with open(args.kernels, 'r') as inf:
	for line in inf:
		kernels = json.loads(line)

# Count terminal occurrences in corpus if provided
corpus_counts = {}
if args.corpus:
	from collections import Counter
	corpus_counts = Counter()
	with open(args.corpus) as inf:
		for line in inf:
			for w in line.split():
				corpus_counts[w] += 1

results = evaluation.evaluate_kernels_hungarian(target, kernels)

# Print summary
print(f"Kernels: {results['n_kernels']} total ({results['n_kernels']-1} non-start), "
	  f"Target NTs: {results['n_nonterminals']} total")
print(f"Start correct:     {results['start_correct']}")
print(f"Accuracy:          {results['accuracy']:.3f}")
print(f"Weighted accuracy: {results['weighted_accuracy']:.3f}")
print(f"Mean posterior:    {results['mean_posterior']:.4f}")
print(f"Min posterior:     {results['min_posterior']:.4f}")
print(f"Product:           {results['product']:.6f}")
print(f"Coverage:          {results['coverage']:.3f}")
print(f"Injective:         {results['injective']}")

# Per-kernel details
if args.verbose:
	print()
	header = (f"{'Kernel':<15} {'Assigned NT':<15} {'Posterior':>10} "
			  f"{'Greedy NT':<15} {'Correct':>8} {'Freq':>10} {'E[NT]':>10}")
	if corpus_counts:
		header += f" {'Count':>10}"
	print(header)
	print("-" * (85 + (11 if corpus_counts else 0)))
	for a in kernels:
		info = results['per_kernel'].get(a)
		if info is None:
			continue
		nt = info['assigned_nt']
		line = (f"{a:<15} {str(nt):<15} "
				f"{info['posterior']:>10.4f} "
				f"{str(info['greedy_nt']):<15} "
				f"{'yes' if info['correct'] else 'NO':>8} "
				f"{info['frequency']:>10.6f} "
				f"{info.get('nt_expectation', 0.0):>10.4f}")
		if corpus_counts:
			line += f" {corpus_counts.get(a, 0):>10}"
		print(line)

# Save JSON
scores = {
	'accuracy': results['accuracy'],
	'weighted_accuracy': results['weighted_accuracy'],
	'mean_posterior': results['mean_posterior'],
	'min_posterior': results['min_posterior'],
	'product': results['product'],
	'coverage': results['coverage'],
	'injective': results['injective'],
	'start_correct': results['start_correct'],
	'n_kernels': results['n_kernels'],
	'n_nonterminals': results['n_nonterminals'],
	'assignment': results['assignment'],
	'per_kernel': results['per_kernel'],
}

if corpus_counts:
	for a in kernels:
		if a in scores['per_kernel']:
			scores['per_kernel'][a]['corpus_count'] = corpus_counts.get(a, 0)

if args.json:
	with open(args.json, 'w') as outf:
		json.dump(scores, outf, sort_keys=True, indent=4)
