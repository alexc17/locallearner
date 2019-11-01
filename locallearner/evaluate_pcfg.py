#evaluate_pcfg.py

import utility
import wcfg
import argparse
import evaluation
import utility 
import math
import sys
import json
from collections import defaultdict

parser = argparse.ArgumentParser(description='Evaluate hypothesis grammar against a target')

parser.add_argument('--target', type=str, required=True,help='filename of original (gold) grammar')
parser.add_argument('--verbose', action="store_true", default=False,help='Print out details')
parser.add_argument('--samples', type=int, default=1000, help='Number of samples to use (default 1000)')
parser.add_argument('--maxlength', type=int, default=100, help='Maximum length of sentences to consider. (default 100)')
parser.add_argument('--seed', type=int, default=None, help='Random seed for Monte Carlo methods. (default None)')
parser.add_argument('--json', help='Filename to store a json version of the evaluation in.')
parser.add_argument('hypotheses', type=str, help='filenames of grammars to be evaluated.', nargs='*')
args = parser.parse_args()
scores = defaultdict(float)
target = wcfg.load_wcfg_from_file(args.target)
hypotheses =[]
kwargs = { "max_length" : args.maxlength, "seed" : args.seed, "samples" : args.samples, "verbose" : args.verbose}

for f in args.hypotheses:
	hypothesis = wcfg.load_wcfg_from_file(f)
	if not hypothesis.is_pcfg(epsilon=1e-10):
		print("ERROR: hypothesis is not PCFG: normalise.")
		print(hypothesis.check_local_normalisation())
		if hypothesis.is_pcfg(epsilon=1e-4):
			print("BUT it's close so we will renormalise.")
			hypothesis.renormalise_locally()
			print(hypothesis.check_local_normalisation())
		else:
			sys.exit()


	if set(target.nonterminals) != set(hypothesis.nonterminals):
		if args.verbose:
			print("relabeling the grammar.")
		mapping = evaluation.estimate_bijection(target, hypothesis,**kwargs)
		if args.verbose:
			for a,b in mapping.items():
				print(a,"->",b)
		hypothesis = hypothesis.relabel({a:b for b,a in mapping.items()})

	if set(target.terminals) != set(hypothesis.terminals):
		print("Warning: grammars do not have the same set of terminals")
		print("In target but not hypothesis", len(set(target.terminals) - set(hypothesis.terminals)))
		print("In  hypothesis but not target ", len(set(hypothesis.terminals) - set(target.terminals)))
		#print(set(target.terminals) - set(hypothesis.terminals))
	hypotheses.append(hypothesis)
# if not hypothesis.is_convergent():
# 	print("Hypothesis is Divergent: so skipping some evaluations")
# else:
	# if args.verbose:
	# 	print("Normalisation check:",hypothesis.check_normalisation())
	
	#print("Expected length: target %f hypothesis %f " % (target.expected_length(), hypothesis.expected_length()))
	# try:
	# 	lkld = evaluation.labeled_kld_exact(target,hypothesis,verbose=args.verbose)
	# 	print("Labeled tree KLD:", lkld)
	# except utility.ParseFailureException:
# 	print("Labeled KLD is infinite")

print("Target")
print('-----Exact Quantities-----')
print("Expected length:",target.expected_length() )
print("Tree Entropy:", target.derivational_entropy())
pf = target.compute_partition_function_fp()[hypothesis.start]
print("Partition function: %f log partition function %f " % (pf,math.log(pf)))
print('-' * 10)

print("Doing Monte Carlo logprob")
lp_scores  = evaluation.do_lp_monte_carlo(target,hypotheses,**kwargs)
scores.update(lp_scores)
## correction factor
correction_factor = target.compute_probability_short_string(args.maxlength)
print("Correction factor for short samples", correction_factor, math.log(correction_factor))
scores['short_length_correction'] = math.log(correction_factor)
print("Doing Monte Carlo parsing")
parsing_scores = evaluation.do_parseval_monte_carlo(target, hypotheses, **kwargs)
scores.update(parsing_scores)
#print(target_scores,hypothesis_scores,n)
#	for grammar,scores,label in [ (target,target_scores,"Target"), (hypothesis,hypothesis_scores,"Hypothesis")]:

n = scores['logprob:target:n']

print('-----MC Estimates: samples', n, '-----')
print("Mean length: ",scores["logprob:target:length"]/n)
print("String entropy: ",-scores["logprob:target:string"]/n)
print("Tree entropy:", -scores["logprob:target:tree"]/n)
print("H(Tree|String):",(scores["logprob:target:string"]-scores["logprob:target:tree"])/n )
print('-' * 10)

skld = evaluation.lpeval_klds(scores,typ="string")
tkld = evaluation.lpeval_klds(scores,typ="tree")
bkld = evaluation.lpeval_klds(scores,typ="bracket")

for i,hypothesis in enumerate(hypotheses):

	print("Hypothesis%d" %i)
	print(args.hypotheses[i])
	print('-----Exact Quantities-----')
	print("Expected length:",hypothesis.expected_length() )
	print("Tree Entropy:", hypothesis.derivational_entropy())
	pf = hypothesis.compute_partition_function_fp()[hypothesis.start]
	print("Partition function: %f log partition function %f " % (pf,math.log(pf)))

	lkld = evaluation.smoothed_kld_exact(target,hypothesis)
	print("exact smoothed KLD  %f  " % lkld)
	scores["exactkld:hypothesis%d" % i] = lkld
	lengthkld = evaluation.length_kld(target, hypothesis, args.maxlength * 2)
	print("Length KLD", lengthkld)
	# try:
	# 	bkld = evaluation.bracketed_kld(target,hypothesis,**kwargs)
	# 	print("Unlabeled tree KLD:", bkld)
	# except utility.ParseFailureException:
	# 	print("Unlabeled tree KLD is infinite")

	# try:
	# 	skld = evaluation.string_kld(target, hypothesis, **kwargs)
	# 	print("String KLD:", skld)
	# except utility.ParseFailureException:
	# 	print("String KLD is infinite")
	print("---Evaluation (Monte Carlo)---")
	print("String KLD", skld[i])
	print("Bracket KLD", bkld[i])
	print("Tree KLD", tkld[i])
	print("Conditional KLD", tkld[i] - skld[i])

	print("String Undergeneration:", scores['logprob:hypothesis%d:string:failures' %i]/n)



	print("---Parsing metrics---")



	print("labeled exact match", parsing_scores["original:hypothesis%d:labeled:exact_match" %i]/parsing_scores["trees_denominator"])
	print("unlabeled exact match", parsing_scores["original:hypothesis%d:unlabeled:exact_match" %i]/parsing_scores["trees_denominator"])
	print("labeled microaveraged accuracy", parsing_scores["original:hypothesis%d:labeled:microaveraged" %i]/parsing_scores["labeled_denominator"])
	print("unlabeled microaveraged accuracy", parsing_scores["original:hypothesis%d:unlabeled:microaveraged" %i]/parsing_scores["unlabeled_denominator"])

# lem = evaluation.labeled_exact_match(target, hypothesis,**kwargs)
# print("Labeled exact match:",lem)
# # lemv = evaluation.labeled_exact_match(target, hypothesis,viterbi=True,**kwargs)
# print("Labeled exact match vs Viterbi:",lemv)

#print(myjson)
if args.json:
	with open(args.json,'w') as outf:
		json.dump(scores, outf, sort_keys=True, indent=4)