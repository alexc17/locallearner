#run_find_kernels.py
# just compute the kernels and save them in a file.
# 



import utility
import wcfg
import argparse
import locallearner
import json

parser = argparse.ArgumentParser(description='Run a quick learner on a single corpus')
parser.add_argument('input', type=str, help='filename of input corpus')
parser.add_argument('output', type=str, help='filename of output kernel json file.')

parser.add_argument('--nonterminals', type=int, default= 0,  help="Number of nonterminals (0 = auto-detect, default 0).")
parser.add_argument('--max_nonterminals', type=int, default= 20,  help="Maximum nonterminals for auto-detection (default 20).")
parser.add_argument('--cheat',   help="Filename of target grammar to be used for setting number of nonterminals only.")

parser.add_argument('--seed', type=int, default=None, help='Random seed for initialisation of clustering. (default None)')
parser.add_argument('--number_clusters', type=int, default= 10,  help="Number of clusters for neyessen, (default 10)")
parser.add_argument('--min_count_nmf', type=int, default= 100,  help="Minimum frequency of words that can be considered to be amchors for nonterminals.(default 100)")

parser.add_argument('--verbose', action="store_true", help="Print out some useful information")


args = parser.parse_args()
ll = locallearner.LocalLearner(args.input)
if args.cheat:
	target_pcfg = wcfg.load_wcfg_from_file(args.cheat)
	n = len(target_pcfg.nonterminals)
	print(f"Number of nonterminals {n} (from target grammar)")
	ll.nonterminals = n
else:
	ll.nonterminals = args.nonterminals
	ll.max_nonterminals = args.max_nonterminals

ll.seed = args.seed
ll.number_clusters = args.number_clusters
ll.min_count_nmf = args.min_count_nmf

kernels = ll.find_kernels(verbose=args.verbose)
with open(args.output,'w') as outf:
	json.dump(kernels,outf)
