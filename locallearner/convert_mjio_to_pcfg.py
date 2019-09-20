#convert_trees_to_pcfg.py

import utility
import wcfg
import argparse


parser = argparse.ArgumentParser(description='Convert Grammar from MJIO format to PCFG')
parser.add_argument('input', type=str, help='filename of input grammar')
parser.add_argument('output', type=str, help='filename of output grammar')

args = parser.parse_args()

wcfg.convert_mjio(args.input, args.output)

