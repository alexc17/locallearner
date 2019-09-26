#evaluate_pcfg.py

import utility
import wcfg
import argparse
import evaluation
import utility 
import math
import sys
import json
from collections import defaultdict

parser = argparse.ArgumentParser(description='Evaluate kernels against a target pcfg')

parser.add_argument('target', type=str,help='filename of original (gold) grammar')
parser.add_argument('kernels', help='filename of kernels')
parser.add_argument('--json', help='filename of json file')
args = parser.parse_args()
scores = {}
target = wcfg.load_wcfg_from_file(args.target)

with open(args.kernels,'r') as inf:
	for line in inf:
		#print(line)
		kernels = json.loads(line)

results = set()
product = 1.0
for a in kernels:
	if a == 'S':
		print("Skipping S")
		results.add('S')
	else:
		x = target.find_best_lhs(a)
		print(x)
		scores[a] = x
		results.add(x[0]) 
		product *= x[1]
print(len(results) == len(target.nonterminals))
print(product)
scores['_product'] = product
scores['_overall'] = (len(results) == len(target.nonterminals))
if args.json:
	with open(args.json,'w') as outf:
		json.dump(scores, outf, sort_keys=True, indent=4)
