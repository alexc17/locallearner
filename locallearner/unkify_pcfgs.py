#unkify_pcfgs.py

import utility
import wcfg
import argparse
import evaluation
import utility 
import math
import sys
import json
from collections import Counter



class Unkifier:
	def __init__(self, filename, unk, threshold):
		self.lexicon = Counter()
		with open(filename) as inf:
			for line in inf:
				for a in line.split():
					self.lexicon[a] += 1

		self.frequent = set()
		self.rare = 0
		self.unk = unk
		for a in self.lexicon:
			if self.lexicon[a] >= threshold:
				self.frequent.add(a)
			else:
				self.rare += 1

		print(f"{self.rare} rare tokens, {len(self.frequent)} frequent tokens")

	def unkify_string(self,s):
		s2 = []
		for a in s:
			if a in self.frequent:
				s2.append(a)
			else:
				s2.append(self.unk)
		return tuple(s2)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Unkify a bunch of grammars.')

	parser.add_argument('--data', type=str, help='filename of corpus')
	parser.add_argument('--vocabsize', type=int, default=0, help='Pick the N most frequent symbols .')
	parser.add_argument('--minexpectation', type=float, default=0, help='Use expectation from grammar rather than corpus.')
	
	parser.add_argument('--verbose', action="store_true", default=False,help='Print out details')
	parser.add_argument('--mincount', type=int, default=10, help='All symbols with frequency less than this are replaced with the unk symbol.')
	parser.add_argument('--unk', default="UNK", help='Token used')
	parser.add_argument('--ingrammars', type=str, required=True, help='filenames of grammars to be unkified.', nargs='*')
	parser.add_argument('--outgrammars', type=str, help='output filenames of grammars to be unkified. (optional -- default just adds .unk to the end of the filename) ', nargs='*')


	args = parser.parse_args()
	unk = args.unk
	if args.data:
		print("Unkifying using file.")
		myunk = Unkifier(args.filename,args.unk,args.mincount)
		frequent = myunk.frequent
	else:
		print("Using first pcfg, ", args.ingrammars[0])
		target = wcfg.load_wcfg_from_file(args.ingrammars[0])
		if args.minexpectation > 0:
			frequent = set(target.frequent_terminals(args.minexpectation))
		elif args.vocabsize > 0:
			frequent = target.most_frequent_terminals(args.vocabsize)
			rarest = frequent[-1]
			print("rarest", rarest, "expectation", target.terminal_expectations()[rarest])
			frequent = set(frequent)
		else:
			raise ValueError("No option set")

	if args.outgrammars and len(args.outgrammars) == len(args.ingrammars):
		outg = args.outgrammars
	else:
		outg = [ f + ".unk" for f in args.ingrammars]

	frequent.add(unk)

	for i,(f,f2) in enumerate(zip(args.ingrammars,outg)):
		g = wcfg.load_wcfg_from_file(f)
		g2 = g.unkify(frequent, unk)
		newts = g2.terminals
		if not newts <= frequent or not newts >= frequent:
			print("Warning: some missing terminals.")
		print(f"Convert {f} {f2}")
		print(f"Old terminals = {len(g.terminals)} new terminals = {len(g2.terminals)}")
		print("expectation of UNK symbol:", g2.terminal_expectations()[unk])
		g2.store(f2)
		



