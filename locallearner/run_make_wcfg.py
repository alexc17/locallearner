#run_make_wcfg.py
# From the kernels make a wcfg, no IO stuff.
# 



import utility
import wcfg
import argparse
import locallearner
import json

parser = argparse.ArgumentParser(description='Run a quick learner on a single corpus')
parser.add_argument('input', type=str, help='filename of input corpus')
parser.add_argument('kernels', type=str, help='filename of kernel json file.')
parser.add_argument('output', type=str, help='filename of output wcfg file.')

parser.add_argument('--seed', type=int, default=None, help='Random seed for initialisation of clustering. (default None)')
parser.add_argument('--number_clusters', type=int, default= 10,  help="Number of clusters for neyessen, (default 10)")

parser.add_argument('--verbose', action="store_true", help="Print out some useful information")


args = parser.parse_args()


## load kernels 
with open(args.kernels) as inf:
	data=inf.read()
kernels = json.loads(data)

ll = locallearner.LocalLearner(args.input)
ll.kernels = kernels

ll.seed = args.seed
ll.number_clusters = args.number_clusters
#ll.min_count_nmf = args.min_count_nmf

print("Starting parameter estimation using kernels",kernels)
raw_wcfg = ll.learn_wcfg_from_kernels_renyi(kernels,verbose=True)


raw_wcfg.store(args.output)

