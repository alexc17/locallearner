#evaluation.py

# code for evaluating a learned WCFG against the true one using various different metrics
# Do not assume that the learned one is a pcfg 

import utility 
import wcfg
import logging
import math
import numpy as np

import numpy.random
from collections import defaultdict
from collections import Counter
from scipy.optimize import linear_sum_assignment


## Utility functions for using the json files


def conditional_entropy(results, model="target"):
	denominator = results["trees_denominator"]
	e = (results[f"logprob:{model}:string"]-results[f"logprob:{model}:tree"])/denominator 


# def find_bijection(target_pcfg, hypothesis_pcfg):
# 	"""
# 	Find a suitable bijection from hypothesis to target.
# 	"""
# 	assert len(target_pcfg.nonterminals) == len(hypothesis_pcfg.nonterminals)
# 	hypo_anchors = find_best_anchor(hypothesis_pcfg)
# 	te = target_pcfg.terminal_expectations()
# 	pe = target_pcfg.production_expectations()
# 	result = {}
# 	for n_h, (a, post) in hypo_anchors.items():
# 		n_t = find_nonterminal_for_anchor(target_pcfg,a,te,pe)
# 		result[n_h] = n_t
# 	assert result[hypothesis_pcfg.start] == target_pcfg.start
# 	return result


# def find_nonterminal_for_anchor(target_pcfg, a, te, pe):
# 	best = None
# 	bestpost = 0.0
# 	for n in target_pcfg.nonterminals:
# 		if (n,a) in pe:
# 			assert a in te
# 			posterior = pe[(n,a)]/te[a]
# 			if posterior > bestpost:
# 				best = n
# 				bestpost = posterior
# 	return best


# def find_best_anchor(target_pcfg):
# 	te = target_pcfg.terminal_expectations()
# 	pe = target_pcfg.production_expectations()
# 	best_anchor = { n : (0,0) for n in target_pcfg.nonterminals}
# 	for prod, e in pe.items():
# 		if len(prod) == 2:
# 			n, a = prod
# 			posterior = e/te[a]
# 			bestpost = best_anchor[n][1]
# 			if posterior > bestpost:
# 				best_anchor[n] = (a, posterior)
# 	return best_anchor



# def evaluate_kernel(target_pcfg, hypothesis_pcfg):
# 	"""
# 	We assume that the nonterminals are labeled.
# 	"""
# 	kernels = []
# 	for nt in hypothesis_pcfg.nonterminals:
# 		if nt.startswith("NT"):
# 			a = nt[3:]
# 			if not a in hypothesis_pcfg.terminals:
# 				return False
# 			kernels.append(a)
# 		elif nt != 'S':
# 			return False
# 	return [ target_pcfg.find_best_lhs(a) for a in kernels]

def evaluate_kernel(target_pcfg, kernels):
	"""
	Check the kernels against the target to see how well they match.
	Return a number between 0 and 1.
	"""
	if len(kernels != len(target_pcfg.nonterminals)):
		return 0.0
	nts = set()
	product = 1.0
	for a in kernels[1:]:
		nt,p = target_pcfg.find_best_lhs(a)
		product *= p
		if nt == target_pcfg.start:
			return 0.0
		nts.add(nt)
	if len(nts) != len(kernels):
		return 0.0
	else:
		return product

def estimate_bijection(target_pcfg, hypothesis_pcfg, samples = 1000, seed=None, max_length=math.inf,verbose=False):
	## Essential assumption
	assert len(target_pcfg.nonterminals) == len(hypothesis_pcfg.nonterminals)
	if seed == None:
		rng = numpy.random.RandomState()
	else:
		rng = numpy.random.RandomState(seed)
	sampler = wcfg.Sampler(target_pcfg,random = rng)
	insider = wcfg.InsideComputation(hypothesis_pcfg)
	n = len(target_pcfg.nonterminals)

	c = defaultdict(Counter)
	
	for _ in range(samples):
		tree = sampler.sample_tree()

		s = utility.collect_yield(tree)
		if len(s) <= max_length:	
			try:
				learned = insider.bracketed_viterbi_parse(tree)
				collect_nonterminal_pairs(tree, learned,c)
			except utility.ParseFailureException:
				print("Failed",s)
	#print(c)	

	maximum_value = max(  max( c2.values()) for nt,c2 in c.items())
	#print(maximum_value)
	## Now use the Hungarian algorithm 

	cost_matrix  = np.zeros((n,n))
	target_list = list(target_pcfg.nonterminals)
	hypothesis_list = list(hypothesis_pcfg.nonterminals)
	for i,a in enumerate(target_list):
		for j,b in enumerate(hypothesis_list):
			count = c[a][b]
			cost_matrix[i,j] = maximum_value - count
			# if count == 0:
			# 	cost_matrix[i,j] = maximum_value
			# else:
			# 	cost_matrix[i,j] = 1/count  ## Maybe normalise so as to maximize something else?
	row_ind, col_ind = linear_sum_assignment(cost_matrix)
	answer = {}
	for i,j in zip(row_ind,col_ind):
		answer[ target_list[i]] = hypothesis_list[j]
	#print(answer)
	return answer

def collect_nonterminal_pairs(tree1, tree2, counter):
	counter[tree1[0]][tree2[0]] += 1
	if len(tree1) == 2:
		return
	if len(tree1) != len(tree2):
		print(tree1)
		print(tree2)
	assert len(tree1) == len(tree2)
	for d in [1,2]:
		collect_nonterminal_pairs(tree1[d],tree2[d], counter)
	
def test_cfg_morphism(target, hypothesis, bijection=None):
	"""
	return true if bijection is a morphism from target to hypothesis.

	"""
	if not bijection:
		bijection = { a:a for a in target.nonterminals}
	for prod in target.productions:
		if len(prod) == 2:
			newprod = (bijection[prod[0]], prod[1])
		else:
			newprod = (bijection[prod[0]], bijection[prod[1]],bijection[prod[2]])
		if not newprod in hypothesis.parameters:
			return False
	return True

def test_cfg_isomorphism(target, hypothesis, bijection=None):
	"""
	return true if bijection is an isomorphism from target to hypothesis.

	"""
	if not bijection:
		bijection = { a:a for a in target.nonterminals}
	return (test_cfg_morphism(target, hypothesis, bijection) and 
		test_cfg_morphism(hypothesis, target, { bijection[a]: a for a in bijection}))




def length_kld(target, hypothesis, l):
	try:
		targetui = wcfg.UnaryInside(target,l)
		hypothesisui = wcfg.UnaryInside(hypothesis,l)
		kld = 0
		pres = 1.0
		qres= 1.0
		for i in range(1,l+1):
			p = targetui.probability(i)
			if p > 0:
				q = hypothesisui.probability(i)
				kld += p * math.log(p/q)
				pres -= p
				qres -= q
		if pres > 0:
			kld += pres * math.log(pres/qres)
		return kld
	except ValueError as e:
		return math.inf
	
def smoothed_kld_exact(target, hypothesis,epsilon=1e-6, compute_bijection = False):
	lexicon = list(target.terminals)
	smoothed = hypothesis.smooth_full(lexicon,epsilon)
	if compute_bijection:
		bijection = estimate_bijection(target, hypothesis)
	else:
		bijection = { a:a for a in target.nonterminals}
	return labeled_kld_exact(target,smoothed,injection=bijection)



def labeled_kld_exact(target, hypothesis, injection=None,verbose=False):
	"""
	injection is a mapping from target to hypothesis.
	BOTH MUST BE PCFGs.

	We can compute this in closed form.
	"""
	# check it is an injection
	if injection:
		N = len(target.nonterminals)
		assert injection[target.start] == hypothesis.start
		assert len(injection.values()) == N
	else:
		injection = { a:a for a in target.nonterminals}
	pe = target.production_expectations()
	kld = 0.0
	for prod in pe:
		e = pe[prod]
		alpha = target.parameters[prod]
		if len(prod) == 2:
			newprod = (injection[prod[0]], prod[1])
		else:
			newprod = (injection[prod[0]], injection[prod[1]],injection[prod[2]])
		if not newprod in hypothesis.parameters:
			print("Failure newprod", newprod)
			return math.inf
		else:
			beta = hypothesis.parameters[newprod]
			delta =  e * math.log(alpha/beta)
			if verbose:
				print(prod,newprod,delta)
			kld +=delta
	return kld


## Monte Carlo methods. Maybe make a uniform interface


def monte_carlo(target, f, samples, max_length, seed = None):
	## f is called on each element
	if seed == None:
		rng = numpy.random.RandomState()
	else:
		rng = numpy.random.RandomState(seed)
	sampler = wcfg.Sampler(target,random=rng)
	for i in range(samples):
		t = sampler.sample_tree()
		s = utility.collect_yield(t)
		if len(s) <= max_length:	
			f(t,i)

class Baselines:

	def __init__(self, n):
		self.n = n
		self.nonterminals = [ 'S']
		for i in range(1,n):
			self.nonterminals.append("R" + str(i))

	def make_left_branch(self, y):			
		if len(y) == 1:
			return ( "S" ,y[0])
		subtree = ("S",  y[-1])
		for w in reversed(y[:-1]):
			subtree = ("S", ("S",w), subtree)
		return subtree

	def make_right_branch(self, y):			
		if len(y) == 1:
			return ( "S" ,y[0])
		subtree = ("S",  y[0])
		for w in y[1:]:
			subtree = ("S", subtree, ("S",w))
		return subtree

	def make_random_labeled(self, sentence):
		return utility.random_binary_tree_labeled(sentence,self.nonterminals)

def do_parseval_monte_carlo(target, hypotheses, samples= 1000,  max_length = 30, seed = None,verbose=False):
	"""
	Do a bunch of parsing based evaluations on a sample from the target.

	Hypotheses is a list of pcfgs to be evaluated. 
	"""
	inside_target = wcfg.InsideComputation(target)
	#inside_hypothesis = wcfg.InsideComputation(hypothesis)
	inside_hypotheses = [ wcfg.InsideComputation(hypothesis) for hypothesis in hypotheses ]
	baselines = Baselines(len(target.nonterminals))
	
	scores = defaultdict(int)
	def f(t,i):
		nonlocal scores
		s = utility.collect_yield(t)
		scores['trees_denominator'] += 1
		scores['labeled_denominator'] += utility.count_labeled(t)
		scores['unlabeled_denominator'] += utility.count_unlabeled(t)

		gold_viterbi = inside_target.viterbi_parse(s)

		try:
			## Viterbi/nonviterbi
			## target, hyppothesis,left,right,random
			## labeled unlabeled
			## exact match / microaveraged
			hypo_viterbis = [ ( inside_hypothesis.viterbi_parse(s), "hypothesis%d" % i ) for i,inside_hypothesis in enumerate(inside_hypotheses) ]
			lb = baselines.make_left_branch(s)
			rb = baselines.make_right_branch(s)
			rand = baselines.make_random_labeled(s)
			
			for target_tree, label1 in [ (t, "original"), (gold_viterbi, "viterbi" )]:
				for eval_tree, label2 in hypo_viterbis + [ (lb, "leftbranch"),(rb,"rightbranch"),(rand,"random"),(gold_viterbi,"gold viterbi") ]:
					scores[label1 + ":" +  label2 + ":labeled:exact_match"] += 1 if target_tree == eval_tree else 0
					scores[label1 + ":" +  label2 + ":unlabeled:exact_match"] += 1 if utility.unlabeled_tree_equal(target_tree,eval_tree) else 0
					(x,n) = utility.microaveraged_labeled(target_tree, eval_tree)
					scores[label1 + ":" +  label2 + ":labeled:microaveraged"] += x
					(x,n) = utility.microaveraged_unlabeled(target_tree, eval_tree)
					scores[label1 + ":" +  label2 + ":unlabeled:microaveraged"] += x


			
			# (x,n) = utility.microaveraged_labeled(t, hypo_viterbi)
			# if hypo_viterbi == t:
			# 	scores['labeled_exact_match'] += 1
			# if hypo_viterbi == gold_viterbi:
			# 	scores['labeled_exact_match_viterbi'] += 1
			# hvu = utility.tree_to_unlabeled_tree(hypo_viterbi)
			# goldu = utility.tree_to_unlabeled_tree(t)
			# if hvu == goldu:
			# 	scores['unlabeled_exact_match'] += 1
			# scores['labeled'] += x
			# scores['labeled_denominator'] += n
			# (x,n) = utility.microaveraged_unlabeled(t, hypo_viterbi)
			# scores['unlabeled'] += x
			# scores['unlabeled_denominator'] += n
			# ## Now some baselines.
			

			## exact match 
		except utility.ParseFailureException:
			print("Parse failure of %s " % s)
			
	monte_carlo(target,f,samples,max_length,seed)
	return scores



## Code for using this dict to do some evaluations.


def lpeval_klds(scores,typ="string"):
	assert typ in [ "string","bracket","tree"]
	nh = scores['number_hypotheses'] 
	n = scores['logprob:target:n']
	answer = []
	for i in range(nh):
		label = "logprob:hypothesis%d:" % i
		if scores[label+ typ + ":failures"] > 0:
			answer.append(math.inf)
		else:
			skld = ( scores["logprob:target:" + typ] - scores[label+typ])/n
			answer.append( skld )
	return answer


def do_lp_monte_carlo(target, hypotheses, samples= 1000,  max_length = 30, seed = None,verbose=False):
	"""
	Do all of the log prob evaluations in one pass.
	return a dict with all results stored as "logprob:model:"
	"""
	inside_target = wcfg.InsideComputation(target)
	inside_hypotheses = [ wcfg.InsideComputation(hypothesis) for hypothesis in hypotheses ]
	
	scores = defaultdict(float)
	scores['number_hypotheses'] = len(hypotheses)
	def f(t,i):
		nonlocal scores
		s = utility.collect_yield(t)
		scores['logprob:target:n'] += 1
		scores['logprob:target:length'] += len(s)
		scores["logprob:target:string"] += inside_target.inside_log_probability(s)
		scores["logprob:target:bracket"] += inside_target.inside_bracketed_log_probability(t)
		scores["logprob:target:tree"] += target.log_probability_derivation(t)
		for i, (hypothesis,inside_hypothesis) in enumerate(zip(hypotheses,inside_hypotheses)):
			label = "logprob:hypothesis%d:" % i
			try:
				scores[label+"string"] += inside_hypothesis.inside_log_probability(s)
				
			except utility.ParseFailureException:
				scores[label+"string:failures"] += 1
			try: 
				scores[label+"bracket"] += inside_hypothesis.inside_bracketed_log_probability(t)
				
			except utility.ParseFailureException:
				scores[label+"bracket:failures"] += 1
			try:
				scores[label + "tree"] += hypothesis.log_probability_derivation(t)
			except utility.ParseFailureException:
				scores[label+'tree:failures'] += 1

	monte_carlo(target,f,samples,max_length,seed)
	return scores

def bracketed_kld(target, hypothesis, samples= 1000,  max_length = 30, seed = None,verbose=False):
	### sample n trees from target. FAST
	inside_target = wcfg.InsideComputation(target)
	inside_hypothesis = wcfg.InsideComputation(hypothesis)
	n = 0
	total = 0
	def f(t,i):
		lp = inside_target.inside_bracketed_log_probability(t)
		lq = inside_hypothesis.inside_bracketed_log_probability(t)
		nonlocal n,total
		n += 1
		total += lp - lq
		if verbose:
			print("Sample %d %s, target %f, hypothesis %f" % (i,t,lp,lq))
	monte_carlo(target,f,samples,max_length,seed)
	return total/n




def labeled_exact_match(target, hypothesis, samples = 1000, max_length = 30, viterbi = False, verbose=False,seed = None):
	"""
	Proportion of trees whose viterbi parse is the same up to a relabeling of the hypothesis tree.
	Target has to be a pcfg; hypothesis can be any WCFG.

	Identical nonterminals
	"""
	if seed == None:
		rng = numpy.random.RandomState()
	else:
		rng = numpy.random.RandomState(seed)
	sampler = wcfg.Sampler(target,random = rng)
	if viterbi:
		inside_target = wcfg.InsideComputation(target)
	inside_hypothesis = wcfg.InsideComputation(hypothesis)
	
	total = 0.0
	n = 0
	for i in range(samples):
		t = sampler.sample_tree()
		s = utility.collect_yield(t)
		if len(s) >= max_length:
			continue
		n += 1
		if viterbi:
			t = inside_target.viterbi_parse(s)
		try:
			th = inside_hypothesis.viterbi_parse(s)
			#relabeled_tree = utility.relabel_tree(th, ntmap)
			relabeled_tree = th
			if relabeled_tree == t:
				total += 1
			elif verbose:
				logging.info("Mismatch in trees with parse of %s", s)
				print(relabeled_tree)
				print(t)
		except utility.ParseFailureException as e:
			# Treat this as a failure .

			print("Parse failure of %s " % s)
	return total/n


def string_kld(target, hypothesis, samples= 1000, verbose=False, max_length = math.inf, seed = None):
	### sample n trees from target. 
	if seed == None:
		rng = numpy.random.RandomState()
	else:
		rng = numpy.random.RandomState(seed)
	inside_target = wcfg.InsideComputation(target)
	inside_hypothesis = wcfg.InsideComputation(hypothesis)
	sampler = wcfg.Sampler(target,random=rng)
	total = 0.0
	n = 0
	for i in range(samples):
		t = sampler.sample_tree()
		s = utility.collect_yield(t)
		if len(s) <= max_length:
			n +=1
			lp = inside_target.inside_log_probability(s)
			lq = inside_hypothesis.inside_log_probability(s)
			if verbose:
				print("Sample %d %s, target %f, hypothesis %f" % (i,t,lp,lq))
			total += lp - lq
	return total/n


def conditional_kld(target, hypothesis, samples = 1000,verbose=False, max_length = math.inf, seed = None):
	"""
	Estimate the kld between the conditional probability distributions.

	for a given string $w$ D( P(tree|w) | Q(tree|w)).

	difference between string KLD and tree KLD.
	
	Target must be a pcfg, hyppthesis can be arbitrary wcfg, not even convergent, with same nonterminals.
	"""
	if seed == None:
		rng = numpy.random.RandomState()
	else:
		rng = numpy.random.RandomState(seed)
	inside_target = wcfg.InsideComputation(target)
	inside_hypothesis = wcfg.InsideComputation(hypothesis)
	sampler = wcfg.Sampler(target, random=rng)
	total = 0.0
	n = 0
	for i in range(samples):
		t = sampler.sample_tree()
		s = utility.collect_yield(t)
		if len(s) > max_length:
			if verbose:
				print("Skipping", len(s))
			continue
		n += 1
		ptree = target.log_probability_derivation(t)
		pstring = inside_target.inside_log_probability(s)

		
		qtree = hypothesis.log_probability_derivation(t)
		qstring = inside_hypothesis.inside_log_probability(s)

		total += (ptree - qtree) - (pstring - qstring)
		if verbose:
			print("%s p(t) = %f, p(w) = %f, q(t) = %f, q(w) = %f" % ( s, ptree,pstring,qtree,qstring))
	return total/n	