#run_stepwise_io.py

# Run stepwise IO for one iteration on bathches. IO using a local copy of the IO code.

import datetime
import tempfile
import sys
import os
import time
import subprocess
import wcfg
import math
from tempfile import TemporaryDirectory
from tempfile import mkdtemp
from collections import defaultdict
import argparse


parser = argparse.ArgumentParser(description='Run EM algorithm locally using C code server')
parser.add_argument('grammar', type=str, help='PCFG file in local format')
parser.add_argument('data', type=str, help='data file')
parser.add_argument('outputgrammar', type=str, help='output PCFG in local format')

parser.add_argument("--io",help="Location of binary.", default='io')
parser.add_argument("--maxlength",type=int,default=20,help="Mximum length (default 20)")
parser.add_argument("--epochs",type=int,default=1,help="Number of epochs, default 1")

parser.add_argument("--batchsize",type=int,default=10000,help="Samples in batch size (default 10000)")
parser.add_argument("--maxbatches",type=float,default=math.inf,help="Number of batches (default is all of them)")
parser.add_argument("--alpha",type=float,default=0.75,help="Alpha parameter eta_k = (k+2)^{-alpha}, default 0.75")


import glob
args = parser.parse_args()
bsz = args.batchsize
batches = args.maxbatches
alpha = args.alpha
epochs = args.epochs

## Maybe set the parameters intelligently wrt to the size of the data etc.
## Create temporary directory

mywcfg = wcfg.load_wcfg_from_file(args.grammar)


tmpdir = mkdtemp()

#with TemporaryDirectory() as tmpdir:

print("Creating temp directory ", tmpdir)

## Convert file to MJIO format


mjio_filename1 = tmpdir + "/igrammar.mjio"

mjio_counts = tmpdir + "/ogrammar.counts"

#
data_file = args.data
batch = 1
with open(args.data) as inf:
	data_file = tmpdir + "/data%d.txt" % batch
	outf = open(data_file,'w')
	i = 0
	for line in inf:
		outf.write(line)
		i += 1
		if i >= bsz:
			outf.close()
			if batch < batches:
				# carry on 
				batch += 1
				data_file = tmpdir + "/data%d.txt" % batch
				outf = open(data_file,'w')
				i = 0
			else:
				break
	else:
		outf.close()
		if i == 0:
			os.remove(outf.name)

## All the data slices are written now.


k = 1

## Vector of sufficient statistics
mu = dict(mywcfg.parameters)
nte = mywcfg.nonterminal_expectations()
for prod in mu:
	mu[prod] *= nte[prod[0]]

for epoch in range(epochs):
	print("Epoch ", epoch)
	for data_file in glob.glob(tmpdir + "/data*.txt"):
		print("Processing file ", data_file)
		mywcfg.store_mjio(mjio_filename1)
		eta_k = (k +2) ** (-alpha)
		k += 1
		print("Eta_k %f" % eta_k)
		cmd = f"{args.io} -e -d 1000 -g {mjio_filename1} -l {args.maxlength} {data_file} > {mjio_counts}"
		#print(cmd)
		# #sys.exit()
		# #stderr=subprocess.STDOUT
		result = subprocess.check_output(cmd,shell=True)
		#print(result)

		## Now load the counts 

		counts = wcfg.load_mjio_counts(mjio_counts)
		# So counts is now a dictionary of the counts
		N = counts[('S1',)]
		print("Loaded ", N, " sentences")
		# ignore this.

		newwcfg = mywcfg.copy()
		for prod,old  in mu.items():
			if len(prod) > 1:
				# if not prod in counts:
				# 	print("Missing", prod)
				newwcfg.parameters[prod] = (1 - eta_k) * old + eta_k * counts.get(prod,0)/N
		
		newwcfg.locally_normalise()
		#newwcfg.store("/Users/alexc/working/locallearner/data/debug/x%d.pcfg" %k)
		mwcfg = newwcfg
		## Now create a new grammar ..

		# number i=of sentences actually processed
		# ## delete remote files?
		# if args.counts:
		# 	result = subprocess.check_output(f"cp {mjio_filename2} {args.outputgrammar}",shell=True)
		# else:
		# 	wcfg.convert_mjio(mjio_filename2, args.outputgrammar)
		# print("Saved file in ", args.outputgrammar)
		## Update - mu := eta_k new + (1- eta) old
newwcfg.store(args.outputgrammar)




