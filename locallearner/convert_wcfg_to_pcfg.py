#convert_wcfg_to_pcfg.py

import utility
import wcfg
import argparse


parser = argparse.ArgumentParser(description='Convert Grammar from potentially inconsistent BUWCFG to a PCFG that defines the same condirional distribution of trees given strings.')
parser.add_argument('input', type=str, help='filename of input grammar')
parser.add_argument('output', type=str, help='filename of output grammar')

args = parser.parse_args()

mywcfg = wcfg.load_wcfg_from_file(args.input)

if not mywcfg.is_convergent():
	mywcfg = mywcfg.renormalise_divergent_wcfg2()

mywcfg.renormalise()
mywcfg.store(args.output)

