#run_quick_learner.py



import utility
import wcfg
import argparse
import locallearner
import json

parser = argparse.ArgumentParser(description='Run a quick learner on a single corpus')
parser.add_argument('input', type=str, help='filename of input corpus')
parser.add_argument('output', type=str, help='filename of output grammar')

parser.add_argument('--skipio',action="store_true",  help="Skip the IO renormalisation.")

parser.add_argument('--nonterminals', type=int, default= 0,  help="Number of nonterminals (0 = auto-detect, default 0).")
parser.add_argument('--max_nonterminals', type=int, default= 20,  help="Maximum nonterminals for auto-detection (default 20).")
parser.add_argument('--seed', type=int, default=None, help='Random seed for initialisation of clustering. (default None)')
parser.add_argument('--io_max_samples', type=int, default= 100000,  help="Number of samples to be used for the IO training. (default 10e5)")
parser.add_argument('--io_max_length', type=int, default= 10,  help="Maximum length of strings fo IO Reestimation, (default 10)")
parser.add_argument('--number_clusters', type=int, default= 10,  help="Number of clusters for neyessen, (default 10)")
parser.add_argument('--min_count_nmf', type=int, default= 100,  help="Minimum frequency of words that can be considered to be amchors for nonterminals.(default 100)")
parser.add_argument('--kernels',  help="save the kernels in a file for evaluation purposes.")
parser.add_argument('--wcfg',  help="save the raw WCFG in a file for evaluation purposes.")
parser.add_argument('--verbose', action="store_true", help="Print out some useful information")


args = parser.parse_args()

ll = locallearner.LocalLearner(args.input)
ll.nonterminals = args.nonterminals
ll.max_nonterminals = args.max_nonterminals
ll.seed = args.seed
ll.em_max_samples = args.io_max_samples
ll.em_max_length = args.io_max_length
ll.number_clusters = args.number_clusters
ll.min_count_nmf = args.min_count_nmf
if args.skipio:
	ll.em_max_samples=1

header = []
ll.learn(verbose=args.verbose)
if args.kernels:
	with open(args.kernels,'w') as outf:
		json.dump(ll.kernels,outf)
if args.skipio:
	
	g = ll.output_pcfg
	header.append("Raw PCFG from learner without IO reestimation.")
else:
	ll.reestimate()
	g = ll.reestimated_pcfg
	header.append("Reestimated PCFG")
	

if args.wcfg:
	ll.output_grammar.store(args.wcfg,header = [ 'Raw wcfg'])
g.store(args.output,header=header)
