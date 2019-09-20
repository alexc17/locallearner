#estimate_hyperparameters.py

# given a grammar, estimate how much data etc we need to learn it.
# Obviously don't use this on a grammar you are learning, rather use it to tune hyperparameters for a
# a given set of grammar hyperparameters


import utility
import wcfg
import argparse
import json
import math

def find_most_likely_kernel(target, nt, thresh):
	te = target.terminal_expectations()
	pe = target.production_expectations()
	best = None
	beste = 0
	for a in target.terminals:
		e = te[a]
		if (nt,a) in pe:
			posterior = pe[(nt,a)]/te[a]
			if posterior > thresh:
				if e > beste:
					beste = e
					best = a
	return (best,beste)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Estimate hyperparameters for a learner from a PCFG. Dont use this on a grammar you will learn. Of course.')
	parser.add_argument('input', type=str, help='filename of input grammar')
#	parser.add_argument('output', type=str, help='filename of output json file')
	parser.add_argument("--posterior", help="Kernel posterior (default 0.9)", default=0.9,type=float)


	args = parser.parse_args()
	target = wcfg.load_wcfg_from_file(args.input)
	te = target.terminal_expectations()
	pe = target.production_expectations()
	## Hyperparams
	threshold = args.posterior
	## Samples needed
	## Number of clusters
	## Min count 

	## Number of clusters. 

	number_clusters = 2 * len(target.nonterminals)

	result = {}
	result["kernel"] = True

	sample_size = 1000
	es = []
	for nt in target.nonterminals:
		if nt != target.start:
			a,e = find_most_likely_kernel(target,nt,threshold)
			if a == None:
				sample_size = math.inf
				result["kernel"] = False
				break
			else:
				post = pe[(nt,a)]/te[a]
				N  = 10 * number_clusters * ( 1.0 / e ) 
				print(a,e,N,post)
				sample_size = max(sample_size, int(N) + 1)
				es.append(e)
	if result["kernel"]:
		result["number_clusters"] = number_clusters
		result["sample_size"] = sample_size
		result["min_count_nmf"] = int(1.0 / min(es)) + 1
	print(json.dumps(result, sort_keys=True, indent=4))
