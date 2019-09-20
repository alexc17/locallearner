#smooth_pcfg.py

import utility
import wcfg
import argparse
import collections
import sgt
parser = argparse.ArgumentParser(description='Smooth PCFG format so every production has non zero probability. Crude smoothing just adding a small quantity to each rule.')
parser.add_argument('input', type=str, help='filename of input grammar')
parser.add_argument('--data',type=str, help='filename of corpus to increase the alphabet size.')
parser.add_argument('--unk',type=str, help='UNK symbol', default="UNK")
parser.add_argument('output', type=str, help='filename of output grammar')
parser.add_argument('--epsilon', type=float,default=1e-6, help='Add this to each production. (default 1e-6)')


args = parser.parse_args()

def add(prod):
	if prod in mywcfg.parameters:
		mywcfg.parameters[prod] += e
	else:
		mywcfg.productions.append(prod)
		mywcfg.parameters[prod] = e

lexicon = collections.Counter()

## Smooth lexical params

FIXME add some code to only load the first n lines less than x

if args.data:
	with open(args.data) as inf:
		for line in inf:
			for a in line.split():
				lexicon[a] += 1
print("Doing simple Good-Turing smoothing")
smoothed,p0 = sgt.simpleGoodTuringProbs(lexicon)
N = sum(lexicon.values())
# p0 is the probability mass that should be reserved for unknown words.
#So all lexical probs are adjusted by smoothed[a]  / lexicon[a]/N
# and we add an UNK for 

# So what about known words whete the production has zero prob?
## smooth these. 

mywcfg = wcfg.load_wcfg_from_file(args.input)
L = mywcfg.expected_length()
unk = args.unk


# Add productions (nt, unk) for each nt.
# we want these to have total expectation p0 * L
# So for a nonterminal wwith length 1 expectation e we want the total expectation of the production to be
#  p0 * L  *  ( posterior)

nte1 = { nt: 0 for nt in mywcfg.nonterminals }
nte2 = { nt: 0 for nt in mywcfg.nonterminals }

for prod,e in mywcfg.production_expectations():
	if len(prod) == 2:
		nte1[prod[0]] += e
	
	nte2[prod[0]] += e

for nt, nte in nte1.items():
	# prob that a preterminal is nt
	posterior = nte/L

	expectation = posterior * p0 * L
	parameter = expectation/ nte2[nt]
	## add production with this value
	prod = (nt,unk)
	mywcfg.productions.append(prod)
	mywcfg.parameters[prod] == parameter

e = args.epsilon
for a in mywcfg.nonterminals:
	for b in mywcfg.nonterminals:
		for c in mywcfg.nonterminals:
			prod = (a,b,c)
			add(prod)
	for b in mywcfg.terminals:
		prod = (a,b)
		add(prod)

## re normalize.
mywcfg.locally_normalise()
mywcfg.store(args.output,header = [" Smoothed a little "])

