#!/usr/bin/env python3

import utility
import wcfg
import argparse


parser = argparse.ArgumentParser(description='Return the number of terminals in a W/PCFG.')
parser.add_argument('input', type=str, help='filename of input grammar')

args = parser.parse_args()


mywcfg = wcfg.load_wcfg_from_file(args.input)


print(len(mywcfg.terminals))