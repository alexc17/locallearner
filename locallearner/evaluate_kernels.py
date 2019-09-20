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

parser.add_argument('target', type=str, required=True,help='filename of original (gold) grammar')
parser.add_argument('kernels', help='filename of kernels')
parser.add_argument('--json', help='filename of json file')
args = parser.parse_args()
scores = defaultdict(float)
target = wcfg.load_wcfg_from_file(args.target)



if args.json:
	with open(args.json,'w') as outf:
		json.dump(scores, outf, sort_keys=True, indent=4)