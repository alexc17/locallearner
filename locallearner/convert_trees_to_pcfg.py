#convert_trees_to_pcfg.py

import utility
import wcfg
import argparse


parser = argparse.ArgumentParser(description='Convert treebank to PCFG or WCFG')
parser.add_argument('input', type=str, help='filename of input treebank')
parser.add_argument('output', type=str, help='filename of output grammar')
parser.add_argument('--wcfg', action="store_true", help="save as bottom up WCFG, default is PCFG")
parser.add_argument('--length', type=int, default= 0,  help="maximum length of strings to be considered. Ones longer than this are discarded.")
parser.add_argument('--n', type=int, default=-1,  help="Only do the first n samples (default all of them).")


args = parser.parse_args()

mypcfg = wcfg.load_wcfg_from_treebank(args.input, args.length, args.n,not(args.wcfg))
mypcfg.store(args.output)

