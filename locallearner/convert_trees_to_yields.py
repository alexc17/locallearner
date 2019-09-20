#convert_trees_to_yields.py

import utility

import argparse


parser = argparse.ArgumentParser(description='Convert treebank to yields')
parser.add_argument('input', type=str, help='filename of input treebank')
parser.add_argument('output', type=str, help='filename of output yield file')


args = parser.parse_args()

with open(args.output,'w') as outf:
	with open(args.input) as inf:
		for line in inf:
			tree = utility.string_to_tree(line)
			syield = utility.collect_yield(tree)
			outf.write(" ".join(syield) + "\n")


