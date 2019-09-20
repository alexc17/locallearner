#run_local_io.py

# Run IO using a local copy of the IO code.

import datetime
import tempfile
import sys
import time
import subprocess
import wcfg
from tempfile import TemporaryDirectory
import argparse


parser = argparse.ArgumentParser(description='Run EM algorithm locally using C code server')
parser.add_argument('grammar', type=str, help='PCFG file in local format')
parser.add_argument('data', type=str, help='data file')
parser.add_argument('outputgrammar', type=str, help='output PCFG in local format')
parser.add_argument("--counts",action='store_true',help="Just compute the expected counts with one iteration and save that. (iterations is one automatically)")

parser.add_argument("--io",help="Location of binary.", default='io')
parser.add_argument("--maxlength",type=int,default=10,help="Mximum length (default 10)")
parser.add_argument("--maxsamples",type=int,default=-1,help="Maximum number of samples (default use them all)")
parser.add_argument("--maxiterations",type=int,default=1000,help="Maximum number of iterations (default 1000)")
parser.add_argument("--minprob",type=float,default=1e-10,help="Minimum prob of rule. default (1e-10)")



args = parser.parse_args()

## Maybe set the parameters intelligently wrt to the size of the data etc.
## Create temporary directory



with TemporaryDirectory() as tmpdir:

	print("Creating temp directory ", tmpdir)

	## Convert file to MJIO format
	mywcfg = wcfg.load_wcfg_from_file(args.grammar)

	mjio_filename1 = tmpdir + "/igrammar.mjio"

	mjio_filename2 = tmpdir + "/ogrammar.mjio"

	mywcfg.store_mjio(mjio_filename1)
	data_file = args.data
	if args.maxsamples > 0:
		## Copy them over
		data_file = tmpdir + "/data.txt"
		with open(data_file,'w') as outf:
			with open(args.data) as inf:
				i = 0
				for line in inf:
					outf.write(line)
					i += 1
					if i >= args.maxsamples:
						break
	if args.counts:
		e = "-e"
	else:
		e = ""
	cmd = f"{args.io} {e} -d 1000 -g {mjio_filename1} -l {args.maxlength} -n {args.maxiterations} -p {args.minprob} -s 1e-6 {data_file} > {mjio_filename2}"
	print(cmd)
	#sys.exit()
	#stderr=subprocess.STDOUT
	result = subprocess.check_output(cmd,shell=True)
	print(result)

	## delete remote files?
	if args.counts:
		result = subprocess.check_output(f"cp {mjio_filename2} {args.outputgrammar}",shell=True)
	else:
		wcfg.convert_mjio(mjio_filename2, args.outputgrammar)
	print("Saved file in ", args.outputgrammar)





