#smooth_pcfg.py

import utility
import wcfg
import argparse
import collections



parser = argparse.ArgumentParser(description='Smooth PCFG format so every production has non zero probability. Crude smoothing just adding a small quantity to each rule.')
parser.add_argument('input', type=str, help='filename of input grammar')
parser.add_argument('sigma',type=str, help='Full alphabet.')
parser.add_argument('output', type=str, help='filename of output grammar')
parser.add_argument('--epsilon', type=float,default=1e-6, help='Add this to each production. (default 1e-6)')


args = parser.parse_args()

def add(prod,e):
	if prod in mywcfg.parameters:
		mywcfg.parameters[prod] += e
	else:
		mywcfg.productions.append(prod)
		mywcfg.parameters[prod] = e


lexicon = []
with open(args.sigma) as inf:
	for line in inf:
		lexicon.append(line.rstrip())


mywcfg = wcfg.load_wcfg_from_file(args.input)
e = args.epsilon

result = mywcfg.smooth_full(lexicon,epsilon)
result.store(args.output,header = [" Smoothed a little "])

