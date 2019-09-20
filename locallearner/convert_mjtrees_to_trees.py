import utility
import argparse


parser = argparse.ArgumentParser(description='Convert treebank from MJIO to normal')
parser.add_argument('input', type=str, help='filename of input treebank')
parser.add_argument('output', type=str, help='filename of output treebank')

args = parser.parse_args()



with open(args.input) as inf:
	with open(args.output,'w') as outf:
		for line in inf:
#			print(line)
			if line.startswith("("):
				tree = utility.string_to_tree(line)
				ltree = utility.convert_mjtree(tree)
				output = utility.tree_to_string(ltree)
				outf.write(output + "\n")
