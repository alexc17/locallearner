#inspect_terminal.py
#
# Print the production expectations and posterior distribution
# for a given terminal symbol in a grammar.

import argparse
import wcfg

parser = argparse.ArgumentParser(
	description='Inspect a terminal: show production expectations and '
	'posterior distribution over nonterminals.')
parser.add_argument('grammar', type=str, help='PCFG file')
parser.add_argument('terminal', type=str, help='Terminal symbol to inspect')
args = parser.parse_args()

g = wcfg.load_wcfg_from_file(args.grammar)

if args.terminal not in g.terminals:
	print(f"Terminal '{args.terminal}' not found in grammar.")
	print(f"Available terminals: {sorted(g.terminals)}")
	exit(1)

a = args.terminal
te = g.terminal_expectations()
pe = g.production_expectations()
nte = g.nonterminal_expectations()

print(f"Terminal: {a}")
print(f"E[count({a})] = {te[a]:.6f}")
print()

# Collect all lexical productions with this terminal on the rhs
prods = [(nt, pe.get((nt, a), 0.0)) for nt in g.nonterminals
		 if (nt, a) in g.parameters]

if not prods:
	print("No productions found with this terminal on the RHS.")
	exit(0)

prods.sort(key=lambda x: -x[1])

print(f"{'Nonterminal':<15} {'P(NT -> a)':>10} {'E[NT -> a]':>12} "
	  f"{'P(NT | a)':>10} {'E[NT]':>10}")
print("-" * 60)

for nt, e_prod in prods:
	param = g.parameters[(nt, a)]
	posterior = e_prod / te[a] if te[a] > 0 else 0.0
	nt_exp = nte.get(nt, 0.0)
	print(f"{nt:<15} {param:>10.6f} {e_prod:>12.6f} "
		  f"{posterior:>10.4f} {nt_exp:>10.4f}")

print()
print(f"{'Sum':>15} {'':>10} {te[a]:>12.6f} {'1.0000':>10}")
