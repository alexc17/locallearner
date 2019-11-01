#!/usr/bin/env python3

import utility
import wcfg
import argparse


parser = argparse.ArgumentParser(description='Save the set of terminals in a W/PCFG.')
parser.add_argument('input', type=str, help='filename of input grammar')
parser.add_argument('output', type=str, help='filename of output file (one per line)')

args = parser.parse_args()


mywcfg = wcfg.load_wcfg_from_file(args.input)

terminals = list(mywcfg.terminals)
terminals.sort()
with open(args.output,'w') as out:
	for a in terminals:
		out.write(a + "\n")
