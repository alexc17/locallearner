#gold_kernels.py
#
# Given a PCFG, find a good set of kernels: for each nonterminal,
# pick a terminal with high posterior P(NT | a) and high expected
# count E[a], using an optimal assignment so that each nonterminal
# gets a distinct kernel.

import argparse
import json
import math
import numpy as np
from scipy.optimize import linear_sum_assignment

import wcfg

parser = argparse.ArgumentParser(
	description='Find gold kernels for a PCFG: pick the best anchor '
	'terminal for each nonterminal.')
parser.add_argument('grammar', type=str, help='PCFG file')
parser.add_argument('--output', type=str, help='Output JSON file for kernels')
parser.add_argument('--alpha', type=float, default=0.5,
	help='Weight for posterior vs frequency: '
	'score = alpha * log(posterior) + (1-alpha) * log(frequency). '
	'Default 0.5.')
parser.add_argument('--verbose', action='store_true',
	help='Print per-nonterminal details')
args = parser.parse_args()

g = wcfg.load_wcfg_from_file(args.grammar)

te = g.terminal_expectations()
pe = g.production_expectations()
nte = g.nonterminal_expectations()

non_start_nts = [nt for nt in g.nonterminals if nt != g.start]
terminals = sorted(g.terminals)
nn = len(non_start_nts)
nt = len(terminals)

if nn == 0:
	print("Grammar has no non-start nonterminals.")
	exit(1)

if nt == 0:
	print("Grammar has no terminals.")
	exit(1)

# Build score matrix: score[j, i] = alpha * log(P(NT_j | a_i)) + (1-alpha) * log(E[a_i])
# We want to assign one terminal to each non-start nonterminal.
# Use Hungarian algorithm on the cost = -score.

alpha = args.alpha

# Compute posteriors: P(NT | a) = E[NT -> a] / E[a]
dim = max(nn, nt)  # pad to square
score_matrix = np.full((nn, nt), -1e30)

for j, ntj in enumerate(non_start_nts):
	for i, a in enumerate(terminals):
		ea = te.get(a, 0.0)
		if ea <= 0:
			continue
		e_prod = pe.get((ntj, a), 0.0)
		if e_prod <= 0:
			continue
		posterior = e_prod / ea
		log_post = math.log(posterior) if posterior > 0 else -1e30
		log_freq = math.log(ea) if ea > 0 else -1e30
		score_matrix[j, i] = alpha * log_post + (1 - alpha) * log_freq

# Pad to square for Hungarian algorithm
if nn != nt:
	dim = max(nn, nt)
	padded = np.full((dim, dim), -1e30)
	padded[:nn, :nt] = score_matrix
	cost_matrix = -padded
else:
	cost_matrix = -score_matrix

row_ind, col_ind = linear_sum_assignment(cost_matrix)

# Extract assignment
kernels = ['S']  # start kernel
assignment = {}

for j, i in zip(row_ind, col_ind):
	if j < nn and i < nt:
		assignment[non_start_nts[j]] = terminals[i]

# Order kernels to match nonterminal order
for ntj in non_start_nts:
	if ntj in assignment:
		kernels.append(assignment[ntj])
	else:
		kernels.append(None)

# Print results
print(f"Grammar: {len(g.nonterminals)} nonterminals, {len(g.terminals)} terminals")
print(f"Alpha (posterior weight): {alpha}")
print()

if args.verbose:
	print(f"{'Nonterminal':<15} {'Kernel':<15} {'P(NT|a)':>10} "
		  f"{'E[a]':>10} {'E[NT]':>10} {'Score':>10}")
	print("-" * 75)

	# Start symbol
	print(f"{g.start:<15} {'S':<15} {'(start)':>10} "
		  f"{'':>10} {nte.get(g.start, 0.0):>10.4f} {'':>10}")

	for j, ntj in enumerate(non_start_nts):
		a = assignment.get(ntj)
		if a is None:
			print(f"{ntj:<15} {'NONE':<15}")
			continue
		ea = te.get(a, 0.0)
		e_prod = pe.get((ntj, a), 0.0)
		posterior = e_prod / ea if ea > 0 else 0.0
		log_post = math.log(posterior) if posterior > 0 else -1e30
		log_freq = math.log(ea) if ea > 0 else -1e30
		score = alpha * log_post + (1 - alpha) * log_freq
		print(f"{ntj:<15} {a:<15} {posterior:>10.4f} "
			  f"{ea:>10.6f} {nte.get(ntj, 0.0):>10.4f} {score:>10.4f}")

	# Also show runner-up for each nonterminal
	print()
	print("Alternatives (top 3 per nonterminal):")
	print(f"{'Nonterminal':<15} {'Rank':>5} {'Terminal':<15} {'P(NT|a)':>10} {'E[a]':>10} {'Score':>10}")
	print("-" * 70)
	for j, ntj in enumerate(non_start_nts):
		# Gather all candidates
		candidates = []
		for i, a in enumerate(terminals):
			ea = te.get(a, 0.0)
			if ea <= 0:
				continue
			e_prod = pe.get((ntj, a), 0.0)
			if e_prod <= 0:
				continue
			posterior = e_prod / ea
			log_post = math.log(posterior) if posterior > 0 else -1e30
			log_freq = math.log(ea) if ea > 0 else -1e30
			score = alpha * log_post + (1 - alpha) * log_freq
			candidates.append((a, posterior, ea, score))
		candidates.sort(key=lambda x: -x[3])
		for rank, (a, posterior, ea, score) in enumerate(candidates[:3], 1):
			marker = " *" if a == assignment.get(ntj) else ""
			print(f"{ntj:<15} {rank:>5} {a:<15} {posterior:>10.4f} {ea:>10.6f} {score:>10.4f}{marker}")

else:
	print("Kernels:", kernels)

# Cross-check: evaluate quality
posteriors = []
for ntj in non_start_nts:
	a = assignment.get(ntj)
	if a is None:
		posteriors.append(0.0)
		continue
	ea = te.get(a, 0.0)
	e_prod = pe.get((ntj, a), 0.0)
	posteriors.append(e_prod / ea if ea > 0 else 0.0)

print()
print(f"Mean posterior:  {np.mean(posteriors):.4f}")
print(f"Min posterior:   {min(posteriors):.4f}")
product = 1.0
for p in posteriors:
	product *= p
print(f"Product:         {product:.6f}")

# Save kernels
if args.output:
	with open(args.output, 'w') as outf:
		json.dump(kernels, outf)
	print(f"\nKernels written to {args.output}")
