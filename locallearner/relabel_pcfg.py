#relabel_pcfg

import utility
import wcfg
import argparse
import evaluation
import logging

parser = argparse.ArgumentParser(description='Map the labels of the first grammar onto the second grammar and save it')

parser.add_argument('target', type=str, help='filename of grammar with the right nonterminals')
parser.add_argument('hypothesis', type=str, help='filename of grammar to be relabeled.')

parser.add_argument('output', type=str, help='filename of output grammar (isomorphic to hypothesis')
parser.add_argument('--samples', type=int, default=1000, help='Number of samples to use (default 1000)')
parser.add_argument('--verbose', action="store_true",  help='Print out relabelling')

args = parser.parse_args()

target = wcfg.load_wcfg_from_file(args.target)
hypothesis = wcfg.load_wcfg_from_file(args.hypothesis)

minn = min(target.nonterminal_expectations().values())
if args.samples < 10 / minn:
	logging.warning("May be too few samples to correctly estimate; since minimum nonterminal expectation is %f", minn)
mapping = evaluation.estimate_bijection(target, hypothesis,args.samples)

if args.verbose:
	for a,b in mapping.items():
		print(a,"->",b)
output = hypothesis.relabel({a:b for b,a in mapping.items()})

output.store(args.output)
