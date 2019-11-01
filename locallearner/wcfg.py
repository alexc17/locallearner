
## Python 3 

import logging
import math
import numpy as np
import numpy.linalg
import numpy.random
from collections import defaultdict
from collections import Counter
import collections
import utility
from utility import *
# import warnings
# warnings.filterwarnings("error")

RIGHT_ARROW = "->"
START_SYMBOL = "S"
UNARY_SYMBOL = "<A>"
EPSILON = 1e-15
BIG_EPSILON = 1e-10
MAX_PARTITION_FUNCTION = 1e100
PARTITION_FUNCTION_MAX_ITERATIONS=100
PARTITION_FUNCTION_EPSILON=1e-10
SAMPLE_MAX_DEPTH=100
SAMPLE_CACHE_SIZE=1000
MIN_PROB = 1e-10
CONDITION_NUMBER=10000

class WCFG:
	"""
	This class stores a WCFG where the underlying CFG is in CNF (more or less).
	"""
	def __init__(self, cfg=None):
		if cfg:
			self.nonterminals = set(cfg.nonterminals)
			self.terminals = set(cfg.terminals)
			self.start = cfg.start
			self.productions = list(cfg.productions)
		else:
			self.nonterminals = set()
			self.terminals = set()
			## Productions are tuples (A, a) or (A, B,C)
			self.start = None
			self.productions = []
		self.parameters = {}
		self.log_parameters = {}
	

	def count_lexical(self):
		return len( [ prod for prod in self.productions if len(prod) == 2])

	def count_binary(self):
		return len( [ prod for prod in self.productions if len(prod) == 3])
	
	def check_normalisation(self):
		"""
		See if this is a PCFG.
		"""
		totals = defaultdict(float)
		for prod in self.productions:
			totals[prod[0]] += self.parameters[prod]
		return totals
	
	def is_normalised(self, epsilon= 1e-5):
		totals = self.check_normalisation()
		for a in totals:
			if abs(1.0 - totals[a]) > epsilon:
				return False
		return True


	def compute_probability_short_string(self, max_length):
		"""
		Compute the probability that this will generate a string s such that len(s) <= max_length.

		We need this to correct for a bias when we are doing Monte Carlo estimates of KLD etc.
		and only restricting ourselves to short strings for efficiency reasons.
		"""
		ui = UnaryInside(self, max_length)
		return sum(ui.probability(i) for i in range(1,max_length+1))


	def make_unary(self):
		""" 
		return a new grammar which has the same distribution over lengths and only one terminal symbol.
		"""
		upcfg = WCFG()
		upcfg.terminals.add(UNARY_SYMBOL)
		upcfg.nonterminals = set(self.nonterminals)
		upcfg.start = self.start
		
		for prod in self.productions:
			p = self.parameters[prod]
			if len(prod) == 3:
				# binary
				upcfg.productions.append(prod)
				upcfg.parameters[prod] = p
			else:
				# lexical
				nt,a = prod
				newprod = (nt,UNARY_SYMBOL)
				if newprod in upcfg.parameters:
					upcfg.parameters[newprod] += p 
				else:
					upcfg.productions.append(newprod)
					upcfg.parameters[newprod] = p
		# Why normalise?
		upcfg.set_log_parameters()
		return upcfg


	def sum_lexical_probs(self, nt):
		return sum( [ self.parameters[prod] for prod in self.productions if len(prod) == 2 and prod[0] == nt ])


	def sum_expectations_with_rhs(self, nta, ntb):
		pe = self.production_expectations()
		total = 0
		for prod, e in pe.items():
			if len(prod) == 3 and prod[1] == nta and prod[2] == ntb:
				total += e
		return total

	def expected_bigrams_constituent(self):
		"""
		Compute the expected nymber of bigram constituents.
		(compare to expected length - 1) which is the expected number of bigrams per sentence.
		"""
		x = 0
		pe = self.production_expectations()
		lp = { nt : self.sum_lexical_probs(nt) for nt in self.nonterminals}
		for prod,e in pe.items():
			if len(prod) == 3:
				x += e * lp[prod[1]] * lp[prod[2]]
		return x


	def most_frequent_terminals(self, n):
		"""
		return a list of the n most frequent terminals.
		"""
		te = self.terminal_expectations()
		terms = list(self.terminals)
		terms.sort(key = lambda x : -te[x])
		return terms[:n]

	def frequent_terminals(self, threshold):
		"""
		return a list of the n most frequent terminals.
		"""
		te = self.terminal_expectations()
		terms = list(self.terminals)
		terms.sort(key = lambda x : -te[x])
		return [ a for a in terms if te[a] >= threshold]

	def unkify(self, frequent_terminals, unk):
		"""
		return a new grammar with all terminals not in frequent_terminals ( a set) 
		as unk.
		"""
		unkg = self.copy()
		assert not unk in self.terminals
		unk_probs = { nt:0 for nt in unkg.nonterminals }
		actual_terminals = set()
		newproductions  = []
		newparameters = {}
		for prod,e in unkg.parameters.items():
			if len(prod) == 3:
				newproductions.append(prod)
				newparameters[prod] = e
			else:
				nt,a = prod
				if a in frequent_terminals:
					newproductions.append(prod)
					newparameters[prod] = e
					actual_terminals.add(a)
				else:
					unk_probs[nt] += e
		added = 0
		for nt,p in unk_probs.items():
			if p > 0:
				prod = (nt,unk)
				added += 1
				newproductions.append(prod)
				newparameters[prod] = p

		if added > 0:
			actual_terminals.add(unk)
		unkg.terminals = actual_terminals
		unkg.productions = newproductions
		unkg.parameters = newparameters
		unkg.set_log_parameters()
		return unkg

	def store(self, filename,header=[]):
		"""
		Store this to a file.
		"""
		with open(filename,'w') as fhandle:
			if len(header) > 0:
				for line in header:
					fhandle.write('#' + line + "\n")
			for prod in self.productions:
				p = self.parameters[prod]
				if len(prod) == 2:	
					fhandle.write( "%e %s %s %s \n" % ( p, prod[0], RIGHT_ARROW,  prod[1] ))
				else:
					fhandle.write( "%e %s %s %s %s \n" % ( p, prod[0], RIGHT_ARROW, prod[1],prod[2]))

	def store_mjio(self, filename):
		"""
		Store this in a format suitable for use by MJ IO code.
		So introduce preterminals for all nonterminals. 
		Assume in CNF, and in a PCFG.
		"""
		with open(filename,'w') as fhandle:
			fhandle.write("1.0 S1 --> S \n")
			preterminals = { nt : ("PRE" + nt) for nt in self.nonterminals}
			## now add up probs of preterminals
			preterminal_probs = { nt : 0.0 for nt in self.nonterminals}
			for prod in self.productions:
				if len(prod) == 2:
					preterminal_probs[prod[0]] += self.parameters[prod]
			#print(preterminal_probs)

			for nt in self.nonterminals:
				if preterminal_probs[nt] > 0:
					fhandle.write( "%e %s --> %s \n" % ( preterminal_probs[nt], nt, preterminals[nt] ))
			for prod in self.productions:
				if len(prod) == 2:
					# preterminal
					p = self.parameters[prod] / preterminal_probs[prod[0]]
					fhandle.write( "%e %s --> %s \n" % ( p, preterminals[prod[0]], prod[1]))
				else:
					# binary rule so 
					p = self.parameters[prod] 
					fhandle.write( "%e %s --> %s %s \n" % ( p, prod[0], prod[1],prod[2]))

	def copy(self):
		"""
		return a new copy of this pcfg.
		"""
		copypcfg = WCFG()
		copypcfg.nonterminals = set(self.nonterminals)
		copypcfg.terminals = set(self.terminals)
		copypcfg.start = self.start
		copypcfg.productions = list(self.productions)
		copypcfg.parameters = dict(self.parameters)
		copypcfg.log_parameters = dict(self.log_parameters)
		return copypcfg


	def relabel(self, mapping):
		"""
		return a new pcfg isomrphic to this one where the nonterminals have been remapped.
		"""
		copypcfg = WCFG()
		copypcfg.nonterminals = mapping.values()
		copypcfg.terminals = set(self.terminals)
		copypcfg.start = mapping[self.start]
		prod_map = {}
		for prod in self.productions:
			if len(prod) == 2:
				new_prod = (mapping[prod[0]],prod[1])
			else:
				new_prod = (mapping[prod[0]],mapping[prod[1]],mapping[prod[2]])
			prod_map[prod] = new_prod
		copypcfg.productions = list(prod_map.values())
		copypcfg.parameters =  { prod_map[prod]: self.parameters[prod] for prod in self.productions}
		copypcfg.log_parameters =  { prod_map[prod]: self.log_parameters[prod] for prod in self.productions}
		return copypcfg

	def trim_zeros(self, threshold = 0.0):
		"""
		destructively remove all zero productions, zero nonterminals and zero terminals.
		"""
		self.productions = [ prod for prod in self.productions if self.parameters[prod] > threshold]
		self.nonterminals = set( [ prod[0] for prod in self.productions])
		self.terminals = set( [ prod[1] for prod in self.productions if len(prod) == 2])
		self.parameters = { prod : self.parameters[prod] for prod in self.productions}
	
	def set_log_parameters(self):
		self.log_parameters = {}
		for prod in self.productions:
			self.log_parameters[prod] = math.log(self.parameters[prod])

	def locally_normalise(self):
		""" Convert directly to a PCFG by local normalisation."""
		totals = defaultdict(float)
		for prod in self.productions:
			totals[prod[0]] += self.parameters[prod]
		for prod in self.productions:
			p = self.parameters[prod]
			if p == 0.0:
				raise ValueError("zero parameter",prod)
			param = p/ totals[prod[0]]
			self.parameters[prod] = param
			self.log_parameters[prod] = math.log(param)

	def locally_normalise_lax(self):
		""" Convert directly to a PCFG by local normalisation."""
		totals = defaultdict(float)
		for prod in self.productions:
			totals[prod[0]] += self.parameters[prod]
		for prod in self.productions:
			self.parameters[prod] /=  totals[prod[0]]

	def find_lhs(self, a):
		return 	[ nt for nt in self.nonterminals if (nt,a) in self.productions]

	def find_best_lhs(self, a):
		"""
		

		returns the nonterminal with highest parameter,
		and the posterior probability of that nonterminal given the terminal a
		"""
		te = self.terminal_expectations()
		pe = self.production_expectations()
		x = [  (nt,pe.get((nt,a),0)/te[a]) for nt in self.nonterminals ] 	
		x.sort(key = lambda y : y[1])
		return  x[-1]


	def check_local_normalisation(self):
		totals = defaultdict(float)
		for prod in self.productions:
			totals[prod[0]] += self.parameters[prod]
		return totals

	def log_score_derivation(self, tree):
		"""
		Compute the log prob of a derivation.
		"""
		if len(tree) == 2:
			# lexical
			return self.log_parameters[tree]
		else:
			left_lp = self.log_score_derivation(tree[1])
			right_lp = self.log_score_derivation(tree[2])
			local_tree = (tree[0], tree[1][0], tree[2][0])
			local_lp = self.log_parameters[local_tree]
			return left_lp + right_lp + local_lp

	def weight_derivation(self, tree):
		"""
		Compute the log prob of a derivation.
		"""
		if len(tree) == 2:
			# lexical
			return self.parameters[tree]
		else:
			left_lp = self.weight_derivation(tree[1])
			right_lp = self.weight_derivation(tree[2])
			local_tree = (tree[0], tree[1][0], tree[2][0])
			local_lp = self.parameters[local_tree]
			return left_lp * right_lp * local_lp


	def estimate_nonterminal_expectations_from_file(self, filename, maxlength, maxcount):
		"""
		Create a new one with parameters estimated from. 
		"""
		nlines = 0
		io = InsideComputation(self)
		
		posteriors = defaultdict(float)
		with open(filename) as inf:
			for line in inf:
				s = tuple(line.split())
				if len(s) <= maxlength:
					io.add_posteriors(s, posteriors)
					nlines += 1
				if nlines >= maxcount:
					break
		expectations = { nt:0 for nt in self.nonterminals }
		for prod,alpha in posteriors.items():
			expectations[prod[0]] += alpha/nlines
		return expectations


	def estimate_bup_from_file(self, filename, maxlength, maxcount):
		"""
		Create a new botom up grammar with the binary parameters restimated from the 
		data.
		"""
		nlines = 0
		io = InsideComputation(self)
		
		posteriors = defaultdict(float)
		with open(filename) as inf:
			for line in inf:
				s = tuple(line.split())
				if len(s) <= maxlength:
					io.add_posteriors(s, posteriors)
					nlines += 1
				if nlines >= maxcount:
					break
		expectations = { nt:0 for nt in self.nonterminals }
		for prod,alpha in posteriors.items():
			expectations[prod[0]] += alpha/nlines
		newone = self.copy()
		for prod,alpha in self.parameters.items():
			if len(prod) == 3:
				a,b,c = prod
				newalpha = 	posteriors[prod]/(nlines * expectations[b] * expectations[c])
				newone.parameters[prod] = newalpha # or maybe some weighted average?
		newone.set_log_parameters()
		return newone

	def estimate_inside_outside_from_file(self, filename, maxlength, maxcount):
		"""
		Create a new one with parameters estimated from. 
		"""
		nlines = 0
		io = InsideComputation(self)
		
		posteriors = defaultdict(float)
		with open(filename) as inf:
			for line in inf:
				s = tuple(line.split())
				if len(s) <= maxlength:
					io.add_posteriors(s, posteriors)
					nlines += 1
				if nlines >= maxcount:
					break
		newone = self.copy()		
		newone.parameters = posteriors
		for prod in self.productions:
			if not prod in newone.parameters:
				## Eg a lexical rule which doesnt appear in the data.
				newone.parameters[prod] = 0
		newone.locally_normalise_lax()
		newone.trim_zeros()
		return newone

	def estimate_inside_outside_from_list(self, data, maxlength, maxcount, robust=True):
		"""
		Create a new one with parameters estimated from. 
		"""
		nlines = 0
		io = InsideComputation(self)
		
		posteriors = defaultdict(float)
		for s in data:
			if len(s) <= maxlength:
				try:
					io.add_posteriors(s, posteriors)
				except ParseFailureException as e:
					if robust:
						print("Failed to parse:",s)
					else:
						raise e
				
				nlines += 1
			if nlines >= maxcount:
				break
		newone = self.copy()		
		newone.parameters = posteriors
		for prod in self.productions:
			if not prod in newone.parameters:
				## Eg a lexical rule which doesnt appear in the data.
				newone.parameters[prod] = 0
		newone.locally_normalise_lax()
		newone.trim_zeros()
		return newone


	def estimate_string_density(self, length, nsamples):
		## crude.
		n = 0.0
		insider = InsideComputation(self)
		termilist = list(self.terminals)
		idx = range(len(termilist))
		for _ in range(nsamples):
			w1 = [ termilist[i] for i in numpy.random.choice(idx, length)]
			try:
				insider.inside_log_probability(w1)
				n += 1
			except utility.ParseFailureException:
				pass
		return n/nsamples
	


	def estimate_ambiguity(self, samples = 1000):
		"""
		Monte Carlo estimate of the conditional entropy H(tree|string)
		"""
		mysampler = Sampler(self)
		insider = InsideComputation(self)
		total = 0.0
		for i in range(samples):
			tree = mysampler.sample_tree()
			s = collect_yield(tree)
			lp = insider.inside_log_probability(s)
			lpd = self.log_probability_derivation(tree)
			total += lp - lpd
		return total/samples
		
	def partition_nonterminals(self):
		"""
		Partition the sets of nonterminals into sets of mutually recursive nonterminals.
		"""
		graph = defaultdict(list)
		for prod in self.productions:
			if len(prod) == 3:
				for i in [1,2]:
					graph[prod[0]].append(prod[i])
		return strongly_connected_components(graph)



	def entropy_nonterminals(self):
		# entropy of the productions.
		e = defaultdict(float)
		for prod in self.productions:
			p = self.parameters[prod]
			if p > 0:
				e[prod[0]] -= p * math.log(p)
		return e

	def derivational_entropy(self):
		"""
		entropy of the distribution over derivation trees.
		"""
		nt_entropies = self.entropy_nonterminals()
		nt_expectations = self.nonterminal_expectations()
		return sum([ nt_entropies[nt] * nt_expectations[nt] for nt in self.nonterminals])
		
	def renormalise(self):
		"""
		renormalise  so that it is consistent.
		destructuve.

		Remove ones with zero parameters.
		"""
		rn = self.compute_partition_function_fp()
		for prod in self.productions:
			if rn[prod[0]] == 0:
				self.parameters[prod] = 0
			else:
				if len(prod) == 3:
					a,b,c = prod
					self.parameters[prod] *=  (rn[b] * rn[c])/rn[a] 
				else:
					self.parameters[prod] *= 1.0/rn[prod[0]]
		self.trim_zeros()
		self.set_log_parameters()
		return self

	def renormalise_locally(self):
		"""destructively renormalise as a pcfg.
		"""
		totals = self.check_local_normalisation()
		for prod in self.parameters:
			self.parameters[prod] /= totals[prod[0]]
		self.set_log_parameters()


	def compute_partition_function_fast(self):
		"""
		Solve the quadratic equations using Newton method.
		"""
		ntindex = { nt:i for i,nt in enumerate(list(self.nonterminals))}
		n = len(ntindex)
		alpha = defaultdict(float)
		beta = np.zeros(n)
		for prod in self.productions:
			p = self.parameters[prod]
			if len(prod) == 2:
				beta[ ntindex[prod[0]]] += p
			else:

				alpha[ (ntindex[prod[0]],ntindex[prod[1]],ntindex[prod[2]])]+= p
		x = np.zeros(n)

		def f(y):
			## evaluate f at this point.
			fy = beta - y
			for i,j,k in alpha:
				a = alpha[(i,j,k)]
				#i is lhs
				fy[i] += a * y[j] * y[k]
			return fy

		def J(y):
			# evalate Jacobian
			J = -1 * np.eye(n)
			for i,j,k in alpha:
				a = alpha[(i,j,k)]
				J[i,j] += a * y[k]
				J[i,k] += a * y[j]
			return J

		for i in range(PARTITION_FUNCTION_MAX_ITERATIONS):
			print(x)
			y = f(x)
			jacobian = J(x)
			cn = np.linalg.cond(J(x))
			if cn > CONDITION_NUMBER:
				logging.warning("High condition number of Jacobian: there may be numerical errors.def compute_partition_function_fast")
			try:
				x1 =  x - np.dot(np.linalg.inv(J(x)),y)
			except np.linalg.LinAlgError:
				raise ValueError("Non invertible matrix.")

			if numpy.linalg.norm(x - x1, 1) < PARTITION_FUNCTION_EPSILON:
				return { nt : x[ntindex[nt]] for nt in self.nonterminals}
			x = x1

		raise ValueError("failed to converge")
		


	def compute_io_products(self):
		i = self.compute_partition_function_fast()
		o = self.nonterminal_expectations()
		return { nt : i[nt] * o[nt] for nt in self.nonterminals}

	def compute_partition_function_fp(self):
		"""
		Return a dict mapping each nonterminal to the prob that a string terminates from that.
		Use the naive fixed point algorithm.
		"""
		bprods = [ prod for prod in self.productions if len(prod) == 3]
		lprodmap = defaultdict(float)
		for prod in self.productions:
			if len(prod) == 2:
				lprodmap[prod[0]] += self.parameters[prod]
		z = defaultdict(float)
		for i in range(PARTITION_FUNCTION_MAX_ITERATIONS):
			z = self._compute_one_step_partition(z, bprods,lprodmap)
		return z


	def production_expectations(self):
		## INCORRECT for general WCFGs fix this. 
		nte = self.nonterminal_expectations()
		return { prod: (self.parameters[prod] * nte[prod[0]]) for prod in self.productions }


	def terminal_expectations(self):
		## Incorrect in general 
		answer = defaultdict(float)
		pe = self.production_expectations()
		for prod in pe:
			if len(prod) == 2:				
				alpha = pe[prod]
				answer[prod[1]] += alpha
		return answer


	def expected_length(self):
		pe = self.production_expectations()
		return sum([ pe[prod] for prod in pe if len(prod) == 2 ])

	def compute_all_scores(self):

		self.inside_scores = self.compute_partition_function_fast()
		self.outside_scores = self._compute_outside_scores()
		self.io_products = self._compute_io_products()
		self.production_expectations = self._compute_production_expectations()
		self.terminal_expectations = self._compute_terminal_expectations()
		self.partition_function = self.inside_scores[self.start]

	def _compute_terminal_expectations(self):
		result = { a: 0 for a in self.terminals }
		for prod,e in self.production_expectations.items():
			if len(prod) == 2:
				result[prod[1]] += e
		return result

	def _compute_io_products(self):
		return { nt : self.inside_scores[nt] * self.outside_scores[nt] for nt in self.nonterminals }

	def _compute_production_expectations(self):
		result = {}
		for prod in self.productions:
			alpha = self.parameters[prod]
			if len(prod) == 2:
				nt,a = prod
				e = self.outside_scores[nt] * alpha
			else:
				nta,ntb,ntc = prod
				e = self.outside_scores[nta] * alpha * self.inside_scores[ntb] * self.inside_scores[ntb]
			result[prod] = e
		return result

	def _compute_outside_scores(self):
		"""
		Compute the sum of outside scores of a nonterminal.
		By definition O(S) = 1.
		For the rest we solve linear system.
		"""
		pf = self.inside_scores
		print(pf)
		n = len(self.nonterminals)
		transitionMatrix = np.zeros([n,n])
		#outputMatrix = np.zeros(n)
		ntlist = list(self.nonterminals)
		#print(ntlist)
		index = { nt:i for i,nt in enumerate(ntlist)}
		insides = [ pf[nt] for nt in ntlist ]
		for prod in self.productions:
			alpha = self.parameters[prod]
			
			if len(prod) == 3:
				a,b,c = [ index[nt] for nt in prod ]
				transitionMatrix[b,a] += alpha * insides[c]
				transitionMatrix[c,a] += alpha * insides[b]
		# s is a one hot vector of the start symbol
		s = np.zeros(n)
		s[index[self.start]] = 1

		# So O = s + transitionMatrix . O
		# So s = O (eye - transitionMatrix)
		print(transitionMatrix)
		print(np.eye(n) - transitionMatrix)
		
		print(numpy.linalg.inv(np.eye(n) - transitionMatrix))
		result = np.dot(numpy.linalg.inv(np.eye(n) - transitionMatrix),s)
		resultD = { nt : result[index[nt]] for nt in self.nonterminals}
		## By definition since start does not appear
#		assert resultD[self.start] == 1
		return resultD

	def nonterminal_expectations(self):
		"""
		Compute the expected number of times each nonterminal will be used in a given derivation.
		return a dict mapping nonterminals to non-negative reals.

		"""
		n = len(self.nonterminals)
		transitionMatrix = np.zeros([n,n])
		#outputMatrix = np.zeros(n)
		ntlist = list(self.nonterminals)
		#print(ntlist)
		index = { nt:i for i,nt in enumerate(ntlist)}
		for prod in self.productions:
			alpha = self.parameters[prod]
			lhs = index[prod[0]]
			if len(prod) == 3:
				transitionMatrix[lhs,index[prod[1]]] += alpha
				transitionMatrix[lhs,index[prod[2]]] += alpha
		
		#print(transitionMatrix)
		#r2 = numpy.linalg.inv(np.eye(n) - transitionMatrix)
		#print(r2)
		result = np.dot(numpy.linalg.inv(np.eye(n) - transitionMatrix),transitionMatrix)
		si = index[self.start]
		resultD = { nt : result[si, index[nt]] for nt in self.nonterminals}
		resultD[self.start] += 1
		return resultD
	
	
	def expected_lengths(self):
		"""
		Compute the expected length of a string generated by each nonterminal.
		Assume that the grammar is consistent.
		"""
		n = len(self.nonterminals)
		m = len(self.terminals)
		ntlist = list(self.nonterminals)
		transitionMatrix = np.zeros([n,n])
		outputMatrix = np.zeros([n,m])
		index = {nt : i for i,nt in enumerate(ntlist)}
		terminalIndex = { a: i for i,a in enumerate(list(self.terminals))}
		# each element stores the expected number of times that 
		# nonterminal will generate another nonterminal in a single derivation.
		for prod in self.productions:
			alpha = self.parameters[prod]
			lhs = index[prod[0]]
			if len(prod) == 2:
				outputMatrix[lhs,terminalIndex[prod[1]]] += alpha
			else:
				transitionMatrix[lhs,index[prod[1]]] += alpha
				transitionMatrix[lhs,index[prod[2]]] += alpha

		
		# n = o * (1- t)^-1
		result = np.dot(numpy.linalg.inv(np.eye(n) - transitionMatrix),outputMatrix)
		return result
		
	def _compute_one_step_partition(self, z, bprods,lprodmap):

		newz = defaultdict(float) 
		for production in bprods:
			score = self.parameters[production] * z[production[1]] * z[production[2]]
			newz[production[0]] += score
			if score > MAX_PARTITION_FUNCTION:
				raise DivergentWCFGException()
		for nt in lprodmap:
			newz[nt] += lprodmap[nt]
		return newz

	

	def is_convergent(self):
		"""
		Test if the grammar is divergent or convergent.
		"""
		
		try:
			pf = self.compute_partition_function_fp()
			return pf[self.start] < math.inf 
		except DivergentWCFGException:
			return False
		
	def is_pcfg(self, epsilon):
		"""
		return true if this is a pcfg but not necesasrily consistent.
		"""
		totals = self.check_local_normalisation()
		for v in totals.values():
			if abs(v - 1.0) > epsilon:
				return False
		return True

	def is_consistent(self,epsilon):
		
		try:
			pf = self.compute_partition_function_fp()
			return  abs(pf[self.start] - 1) < epsilon
		except DivergentWCFGException:
			return False

	def renormalise_divergent_wcfg2(self):
		"""
		Different algorithm that relies on it being CNF.
		"""
		totals = self.check_local_normalisation()
		beta = max(totals.values())
		print("Beta factor", beta)
		## scale everything by beta.
		if beta > 1:
			logging.info("Scaling all productions by factor of %f", beta)
			pcfg1 = self.copy()
			for prod in self.productions:
				pcfg1.parameters[prod] /= beta
			pcfg1.set_log_parameters()
			return pcfg1
		else:

			logging.info("Not doing anything : beta = %f", beta)
			return self

	def log_probability_derivation(self, tree):
		"""
		Compute the log prob of a derivation.
		"""
		if len(tree) == 2:
			# lexical
			try:
				return self.log_parameters[tree]
			except KeyError:
				raise utility.ParseFailureException	
		else:
			left_lp = self.log_probability_derivation(tree[1])
			right_lp = self.log_probability_derivation(tree[2])
			local_tree = (tree[0], tree[1][0], tree[2][0])
			try:
				local_lp = self.log_parameters[local_tree]
			except KeyError:
				raise utility.ParseFailureException
			
			return left_lp + right_lp + local_lp

	def renormalise_divergent_wcfg(self):
		"""
		Algorithm from Smith and Johnson (1999) 
		Weighted and Probabilistic Context-Free
		Grammars Are Equally Expressive.

		This gives us a wcfg which is convergent, then later we can convert to a PCFG.

		"""
		sigma = len(self.terminals)
		beta = max ( [ self.parameters[prod] for prod in self.productions if len(prod) == 3])
		nu = max ( [ self.parameters[prod] for prod in self.productions if len(prod) == 2])

		factor = 1.0 / ( 8 * sigma * beta * nu)

		pcfg1 = self.copy()
		for prod in self.productions:
			if len(prod) == 2:
				pcfg1.parameters[prod] *= factor
		pcfg1.set_log_parameters()
		return pcfg1

	def convert_parameters_pi2xi(self):
		nte = self.nonterminal_expectations()
		xi = {}
		for prod in self.productions:
			if len(prod) == 3:
				a,b,c = prod
				param = self.parameters[prod]
				xib = param * nte[a]/ (nte[b] * nte[c])
				xi[prod] = xib
			else:
				a,b = prod
				param = self.parameters[prod]
				xib = param * nte[a]
			xi[prod] = xib
		xipcfg = self.copy()
		xipcfg.parameters = xi
		xipcfg.set_log_parameters()
		return xipcfg
		
	def convert_parameters_xi2pi(self):
		"""
		Assume pcfg1 has parameters in xi format.
		Convert these to pi
		"""
		pcfg1 = self.copy()
		expectations = pcfg1.compute_partition_function_fast()
		for prod in pcfg1.productions:
			param = pcfg1.parameters[prod]
			if len(prod) == 2:
				nt,a = prod
				newparam = param/expectations[nt]
			else:
				a,b,c = prod
				newparam = param * expectations[b] * expectations[c] / expectations[a]
			pcfg1.parameters[prod] = newparam
		pcfg1.set_log_parameters()
		return pcfg1


	def smooth_full(self, lexicon, epsilon):
		"""
		nondestructively smooth this grammar by 
		adding all possible rules using this lexicon,
		"""
		mywcfg = self.copy()
		def add(prod):
			if prod in mywcfg.parameters:
				mywcfg.parameters[prod] += epsilon
			else:
				mywcfg.productions.append(prod)
				mywcfg.parameters[prod] = epsilon
		for a in self.nonterminals:
			for b in self.nonterminals:
				for c in self.nonterminals:
					if b != self.start and c != self.start:
						prod = (a,b,c)
						add(prod)
			for b in lexicon:
				prod = (a,b)
				add(prod)
		mywcfg.terminals = lexicon
		mywcfg.locally_normalise()
		return mywcfg

	def intersect(self, target):
		""" 
		return a new WCFG that always generates a.
		i.e. intersect with SIgma^* a Sigma*
		"""
		iwcfg = WCFG()
		# this is buggy if we have 
		for nt in self.nonterminals:
			if nt == self.start:
				iwcfg.nonterminals.add(nt + "x01")
			else:
				iwcfg.nonterminals.add(nt + "x00")
				iwcfg.nonterminals.add(nt + "x01")
				iwcfg.nonterminals.add(nt + "x11")
		iwcfg.start = self.start + 'x01'
		def ap(prod,alpha):
			if len(prod) == 2:
				if prod[0] in iwcfg.nonterminals:

					iwcfg.productions.append(prod)
					iwcfg.parameters[prod] = alpha
			else:
				a,b,c = prod
				if a in iwcfg.nonterminals and b in iwcfg.nonterminals and c in iwcfg.nonterminals:
					iwcfg.productions.append(prod)
					iwcfg.parameters[prod] = alpha


		for prod in self.productions:
			alpha = self.parameters[prod]
			if len(prod) == 2:
				nt,b = prod
				if b == target:
					ap((nt + "x01",b),alpha)
				else:
					ap((nt + "x00",b),alpha)
				ap((nt + "x11",b),alpha)
			else:
				a,b,c = prod
				ap((a + "x00",b+ "x00", c+"x00"),alpha)
				ap((a + "x01",b+ "x00", c+"x01"),alpha)
				ap((a + "x01",b+ "x01", c+"x11"),alpha)
				ap((a + "x11",b+ "x11", c+"x11"),alpha)

		iwcfg.terminals = set(self.terminals)
		# remove redundant ones
		## BUG HERE
		#print(iwcfg.nonterminals)
		nte = iwcfg.compute_io_products()
		#print(nte)
		zeros = set( [nt for nt in nte if nte[nt] < EPSILON])
		#print(zeros)
		for prod in iwcfg.productions:
			if prod[0] in zeros:
				iwcfg.parameters[prod] = 0
			if len(prod) == 3:
				a,b,c = prod
				if b in zeros or c in zeros:
					iwcfg.parameters[prod] = 0
		iwcfg.trim_zeros()
		iwcfg.set_log_parameters()
		return iwcfg	


class UnaryInside:
	"""
	Use this to do length computations
	"""

	def __init__(self, mypcfg, length):
		self.mypcfg = mypcfg
		self.ntindex = { nt : i for i,nt in enumerate(list(mypcfg.nonterminals)) }
		self.start = self.ntindex[mypcfg.start]
		self.nts = list(mypcfg.nonterminals)
		self.nnts = len(self.nts)
		self.lexical_probs = np.zeros(self.nnts)
		for prod,alpha in mypcfg.parameters.items():
			if len(prod) == 2:
				nt,a = prod
				self.lexical_probs[self.ntindex[nt]] += alpha

		self.prods = []
		for prod in mypcfg.productions:
			if len(prod) == 3:
				p = mypcfg.parameters[prod]
				a,b,c = prod
				x,y,z = self.ntindex[a], self.ntindex[b], self.ntindex[c]
				
				self.prods.append((x,y,z,p,len(self.prods)))

	
		# width 
		table = np.zeros((length+1, self.nnts))
		
		table[1,:] = self.lexical_probs

		for width in range(2,length+1):
			for middle in range(1, width):
				for x,y,z,p,_ in self.prods:
					table[width, x] += table[middle,y] * table[width-middle,z] * p 
					assert(table[width,x] <= 1.0)
		self.table = table
		

	def probability(self,i):
		return self.table[i, self.start]

class InsideComputation:

	def __init__(self, pcfg):
		## store all the probabilities in some useful form.
		self.lprod_index = collections.defaultdict(list)
		self.bprod_index = collections.defaultdict(list)
		self.rprod_index = collections.defaultdict(list)
		self.log_parameters = pcfg.log_parameters.copy()
		self.parameters = pcfg.parameters.copy()
		self.terminals = list(pcfg.terminals)
		self.nonterminals = list(pcfg.nonterminals)
		self.start = pcfg.start
		for prod in pcfg.productions:
			if len(prod) == 2:
				self.lprod_index[prod[1]].append( (prod, pcfg.log_parameters[prod]))
			if len(prod) == 3:
				self.bprod_index[prod[1]].append( (prod, pcfg.log_parameters[prod]))
				self.rprod_index[prod[2]].append( (prod, pcfg.log_parameters[prod]))


	
	

	def _bracketed_log_probability(self, tree):
		"""
		Compute log prob of all derivations with this bracketed tree.
		Return a dict mapping nontermials to log probs
		"""
		if len(tree) == 2:
			return { prod[0]: lp for prod,lp in self.lprod_index[tree[1]]}
		else:
			left = self._bracketed_log_probability(tree[1])
			right = self._bracketed_log_probability(tree[2])
			answer = {}
			for leftcat in left:
				for prod, prod_lp in self.bprod_index[leftcat]:
					a,b,c = prod
					if c in right:
						score = prod_lp + left[leftcat] + right[c]
						if a in answer:
							answer[a] = np.logaddexp(score, answer[a])
						else:
							answer[a] = score
			return answer 

	def _bracketed_viterbi_probability(self, tree, mapping):
		"""
		Compute log prob of maximum derivations with this bracketed tree.
		Return a dict mapping nontermials to log probs, prod pairs
		"""
		if len(tree) == 2:
			localt = { prod[0]: (lp,prod) for prod,lp in self.lprod_index[tree[1]]}
			mapping[tree] = localt
			return localt
		else:
			left = self._bracketed_viterbi_probability(tree[1],mapping)
			right = self._bracketed_viterbi_probability(tree[2],mapping)
			answer = {}
			for leftcat in left:
				for prod, prod_lp in self.bprod_index[leftcat]:
					a,b,c = prod
					if c in right:
						score = prod_lp + left[leftcat][0] + right[c][0]
						if a in answer:
							if score > answer[a][0]:
								answer[a] = (score, prod)
						else:
							answer[a] = (score,prod)
			if len(answer) == 0:
				raise ParseFailureException()
			mapping[tree] = answer
			return answer 

	def bracketed_viterbi_parse(self,tree):
		mapping = {}
		root = self._bracketed_viterbi_probability(tree,mapping)
		if not self.start in root:
			raise ParseFailureException()
		def reconstruct_tree(label, tree, mapping):
			lp, prod = mapping[tree][label]
			assert label == prod[0]
			if len(tree) == 2:
				return prod
			else:
				a,b,c = prod
				left_subtree = reconstruct_tree(b, tree[1], mapping)
				right_subtree = reconstruct_tree(c, tree[2], mapping)
				return (a, left_subtree, right_subtree)
		return reconstruct_tree(self.start,tree,mapping)


	def inside_probability(self,sentence):
		try:
			lp = self.inside_log_probability(sentence)
			return math.exp(lp)
		except ParseFailureException:
			return 0

	def inside_log_probability(self, sentence):
		table,_ = self._compute_inside_table(sentence)
		idx = (0,self.start,len(sentence))
		if idx in table:
			return table[idx]
		else:
			raise ParseFailureException(sentence)

	def inside_bracketed_log_probability(self, tree):
		table = self._bracketed_log_probability(tree)
		#print(table)
		if self.start in table:
			return table[self.start]
		else:
			raise ParseFailureException()

	def inside_log_probability_context(self,l,r,wildcard=""):
		sentence = l + (wildcard,) + r
		table,_ = self._compute_inside_table(sentence,wildcard=wildcard)
		idx = (0,self.start,len(sentence))
		if idx in table:
			return table[idx]
		else:
			raise ParseFailureException(sentence)

	def inside_log_probability_context2(self,l,r,wildcard=""):
		sentence = l + (wildcard,wildcard) + r
		table,_ = self._compute_inside_table(sentence,wildcard=wildcard)
		idx = (0,self.start,len(sentence))
		if idx in table:
			return table[idx]
		else:
			raise ParseFailureException(sentence)

	def _compute_inside_table(self,sentence,mode=0, wildcard = ""):
		"""
		mode 0 logprobs
		mode 1 max log prob
		mode 2 counts
		"""
		## simple cky but sparse
		l = len(sentence)
		# take the ;eft hand one and then loop through all productions _ -> A _
		## table mapping items to lp
		table = {}
		## index mapping start end to list of category,lp pairs
		table2 = collections.defaultdict(list)
		def _add_to_chart( start, category, end, score):
			table[ (start,category,end)] = score
			table2[(start,end)].append((category,score))

		for i,a in enumerate(sentence):
			if a == wildcard:
				for nt in self.nonterminals:
					if mode == 0:	
						score = math.log(sum( [ self.parameters[prod] for prod in self.parameters if prod[0] == nt and len(prod) == 2]))
					elif mode == 1:	
						score = math.log(max( [ self.parameters[prod] for prod in self.parameters if prod[0] == nt and len(prod) == 2]))
					elif mode == 2:
						score = len( [ prod for prod in self.parameters if prod[0] == nt])	
					_add_to_chart(i,nt,i+1,score)
			else:
				for prod, lp in self.lprod_index[a]:
					if mode == 0:
						score = lp
					elif mode == 1:
						score = (lp,)
					else:
						score= 1
					_add_to_chart(i,prod[0],i+1,score)
					
		#print("Lexical items", table)
		for width in range(2,l+1):
			for start in range(0, l+1 -width):
				end = start + width
				scores = {}
				for middle in range(start+1, end):
					#print(start,middle,end)
					#print("table2 ", table2[(start,middle)])
					for leftnt,left_score in table2[(start,middle)]:
						#print("table2",leftnt,lp1)
						
						#print("index of binary rules", leftnt, self.bprod_index[leftnt] )
						for prod,prod_lp in self.bprod_index[leftnt]:
							#print("trying",prod,lp2)
							lhs, left, right = prod
							assert left == leftnt
							# now see if we have a (middle,right,end ) in the chart
							try:
								right_score = table[(middle,right,end)]
								if mode == 0:
									score = left_score + prod_lp + right_score
								elif mode == 1:
									score = left_score[0] + prod_lp + right_score[0]
								else:
									# counts 
									score = left_score * right_score
								#print(lp1,lp2,lp3,score)

								if lhs in scores:
									if mode == 0:
										scores[lhs]= np.logaddexp(score, scores[lhs])
									elif mode == 1:
										# Viterbi logprob
										if score > scores[lhs][0]:

											scores[lhs]= (score, middle, prod)

									elif mode == 2:
										# counts
										scores[lhs] += score
								else:
									if mode == 1:
										scores[lhs]= (score, middle, prod)
									else:
										scores[lhs] = score
							except KeyError:
								#print("no entry on the rright", middle, right,end)
								#print(table)
								pass
				for lhs in scores:
					_add_to_chart(start,lhs,end,scores[lhs])
		#print(table)
		#print(table2)
		return table, table2
	
	def _compute_outside_table(self, sentence, table, table2):
		# Compute the outside probs
		l = len(sentence)
		otable = {}
		otable[ (0, self.start,l)] = 0.0
		for width in reversed(range(1,l)):
			for start in range(0, l+1-width):
				scores = {}
				end = start+width
				#print("Starting",start,end)
				# we compute item of the form (start, cat, end)
				for leftpos in range(0,start):
					# case where this is the right branch of a binary rule.
					for leftcat, lp1 in table2[(leftpos, start)]:
						#print("Left", leftcat, leftpos)
						for prod,lp2 in self.bprod_index[leftcat]:
							#print("Production", prod)
							lhs, left, right = prod
							assert leftcat == left
							try:
								lp3 = otable[(leftpos,lhs,end)]
								#print("outside entry for ", leftpos,lhs,end,lp3)
								score = lp1 + lp2 + lp3
								if right in scores:
									scores[right]= np.logaddexp(score, scores[right])
								else:
									scores[right] = score
							except KeyError:
								#print("No outside entry for ", leftpos,lhs,end)
								pass
				for rightpos in range(end+1, l+1):

					# case where this is the left branch of a binary rule.
					for rightcat, lp1 in table2[(end, rightpos)]:
						#print("Right", rightcat, rightpos)
						for prod,lp2 in self.rprod_index[rightcat]:
							#print("Production", prod)
							lhs, left, right = prod
							assert rightcat == right
							try:
								lp3 = otable[(start,lhs,rightpos)]
								score = lp1 + lp2 + lp3
								if left in scores:
									scores[left]= np.logaddexp(score, scores[left])
								else:
									scores[left] = score
							except KeyError:
								# there is no 
								pass
				# completed for start end
				#print(start,end, scores)
				for cat in scores:
					otable[ (start, cat, end)] = scores[cat]
		return otable

	def add_posteriors(self, sentence, posteriors, weight=1.0):
		l = len(sentence)
		table, table2 = self._compute_inside_table(sentence)
		if not (0, self.start,l) in table:
			raise ParseFailureException("Can't parse," , sentence)
		total_lp = table[(0, self.start,l)]
		otable = self._compute_outside_table(sentence, table, table2)
		
		for start,cat,end in otable:
			olp = otable[ (start,cat,end)]
			if end == start + 1:
				a = sentence[start]
				prod = (cat,a)
				if prod in self.log_parameters:
					rule_lp = self.log_parameters[prod]
					posterior = math.exp(rule_lp + olp - total_lp)
					posteriors[prod] += weight * posterior
			else:
				for middle in range(start+1,end):
					for lcat, ilp1 in table2[(start,middle)]:
						for rcat, ilp2 in table2[(middle,end)]:
							prod = (cat,lcat,rcat)
							if prod in self.log_parameters:
								rule_lp = self.log_parameters[prod]
								posterior = math.exp(olp + ilp1 + ilp2 + rule_lp - total_lp)
								posteriors[prod] += weight * posterior
		return total_lp

	def viterbi_parse_unk(self,sentence,unk):
		if unk:
			new_sentence = [ ]
			for a in sentence:
				if a in self.terminals:
					new_sentence.append(a)
				else:
					new_sentence.append(unk)
			return self.viterbi_parse(tuple(new_sentence))
		else:
			return self.viterbi_parse(sentence)

	def viterbi_parse(self,sentence):
		table, table2 = self._compute_inside_table(sentence,mode=1)
		def extract_tree(start,lhs,end):
			if end == start + 1:
				return (lhs, sentence[start])
			else:
				if not (start,lhs,end) in table:
					raise ParseFailureException()
				score, middle, prod = table[(start,lhs,end)]
				lhs,rhs1,rhs2 = prod
				left_subtree = extract_tree(start,rhs1,middle)
				right_subtree = extract_tree(middle,rhs2,end)
				return (lhs, left_subtree,right_subtree)
		return extract_tree(0, self.start, len(sentence))

	def count_parses(self,sentence):
		table, table2 = self._compute_inside_table(sentence,mode=2)
		item = (0,self.start,len(sentence))
		if not item in table:
			return 0
		return table[item]





	
def load_wcfg_mjio(filename):
	"""
	Load a file from MJIO format.
	"""
	with open(filename, 'r') as fhandle:
		preterminal_rules = []
		unary_rules = {}
		binary_rules = []
		for line in fhandle:
			tokens = line.split()
			if len(tokens) == 4:
				# Various cases
				p = float(tokens[0])
				lhs = tokens[1]
				assert tokens[2] == '-->'
				rhs = tokens[3]
				if lhs == 'S1' and rhs == 'S':
					continue
				if lhs.startswith("PRE"):
					nt = lhs[3:]
					#print "nonterminal", nt, lhs
					preterminal_rules.append( (p,nt,rhs))
				if rhs.startswith("PRE"):
					#print "NOnterminal", lhs
					unary_rules[lhs] = p
			elif len(tokens) == 5:
				p = float(tokens[0])
				lhs = tokens[1]
				assert tokens[2] == '-->'
				rhs0 = tokens[3]
				rhs1 = tokens[4]
				## BInary rules are normal 
				binary_rules.append( (p, lhs, rhs0, rhs1))
			else:

				assert len(tokens) == 0
	result = WCFG()
	for (p, lhs, rhs0, rhs1) in binary_rules:
		prod = (lhs,rhs0,rhs1)
		result.productions.append(prod)
		result.parameters[prod] = p
		result.nonterminals.add(lhs)
	for (p, lhs, rhs) in preterminal_rules:
		#print lhs, "-->", rhs
		prod = (lhs,rhs)
		result.productions.append(prod)
		result.parameters[prod] = p * unary_rules[lhs]
		result.nonterminals.add(lhs)
		result.terminals.add(rhs)
	result.start = 'S'
	result.set_log_parameters()
	return result


def load_mjio_counts(filename):
	"""
	Load a file of counts from MJIO format.
	"""
	counts = {}
	with open(filename, 'r') as fhandle:
		preterminal_rules = []
		unary_rules = {}
		binary_rules = []
		for line in fhandle:
			tokens = line.split()
			if len(tokens) == 4:
				# Various cases
				p = float(tokens[0])
				lhs = tokens[1]
				assert tokens[2] == '-->'
				rhs = tokens[3]
				if lhs == 'S1' and rhs == 'S':
					counts[('S1',)] = p
				if lhs.startswith("PRE"):
					nt = lhs[3:]
					#print "nonterminal", nt, lhs
					counts[ (nt,rhs)] = p
			elif len(tokens) == 5:
				p = float(tokens[0])
				lhs = tokens[1]
				assert tokens[2] == '-->'
				rhs0 = tokens[3]
				rhs1 = tokens[4]
				## BInary rules are normal 
				counts[ (lhs, rhs0, rhs1)] = p
			else:

				assert len(tokens) == 0
	return counts

def convert_mjio(filename, out_file):
	"""
	Convert from MJIO format to regular format.
	"""
	with open(filename, 'r') as fhandle:
		preterminal_rules = []
		unary_rules = {}
		binary_rules = []
		for line in fhandle:
			tokens = line.split()
			if len(tokens) == 4:
				# Various cases
				p = float(tokens[0])
				lhs = tokens[1]
				assert tokens[2] == '-->'
				rhs = tokens[3]
				if lhs == 'S1' and rhs == 'S':
					continue
				if lhs.startswith("PRE"):
					nt = lhs[3:]
					#print "nonterminal", nt, lhs
					preterminal_rules.append( (p,nt,rhs))
				if rhs.startswith("PRE"):
					#print "NOnterminal", lhs
					unary_rules[lhs] = p
			elif len(tokens) == 5:
				p = float(tokens[0])
				lhs = tokens[1]
				assert tokens[2] == '-->'
				rhs0 = tokens[3]
				rhs1 = tokens[4]
				## BInary rules are normal 
				binary_rules.append( (p, lhs, rhs0, rhs1))
			else:

				assert len(tokens) == 0

	with open(out_file,'w') as ohandle:
		for (p, lhs, rhs0, rhs1) in binary_rules:
			ohandle.write("%e %s -> %s %s \n" % (p,lhs,rhs0,rhs1))
		for (p, lhs, rhs) in preterminal_rules:


			ohandle.write("%e %s -> %s \n" % (p * unary_rules[lhs],lhs, rhs))






def load_wcfg_from_treebank(filename, length, maxcount, pcfg=True):
	mycounter = Counter()
	ns = 0

	with open(filename) as inf:
		for line in inf:
			tree = utility.string_to_tree(line)
			if length > 0:
				s = utility.collect_yield(tree)
				if len(s) > length:
					continue
			start = tree[0]
			ns += 1
			utility.count_productions(tree, mycounter)
			if maxcount > 0 and ns >= maxcount:
				break
	productions = list(mycounter)
	terminals = [ prod[1] for prod in productions if len(prod) == 2]
	nonterminals = [prod[0] for prod in productions ]
	nonterminal_counts = defaultdict(float)
	for prod,n  in mycounter.items():
		nonterminal_counts[prod[0]] += n
	parameters = {}
	for prod,c  in mycounter.items():
		
		if pcfg:
			# ns cancels out 
			parameters[prod] = c / nonterminal_counts[prod[0]]
		else:
			if len(prod) == 2:
				# E(A \to a) = c/ns
				parameters[prod] = float(c)/ns
			else:
				#  theta(A to BC) = E(A to BC)/ (E(B) * E(C)) = (c(A to BC)/ns)/ ( B/ns) * (C/ns)
				parameters[prod] = c * ns  / (nonterminal_counts[prod[1]] * nonterminal_counts[prod[2]])
	## Now make the grammar
	my_pcfg = WCFG()
	my_pcfg.start =  start
	my_pcfg.terminals = terminals
	my_pcfg.nonterminals = nonterminals
	my_pcfg.productions = productions
	my_pcfg.parameters = parameters
	my_pcfg.set_log_parameters()
	return my_pcfg
	

def load_wcfg_from_file(filename):
	nonterminals = set()
	## Nonterminals are things that appear on the lhs of a production.
	## Start symbol is S
	## Um thats it ?
	prods = []
	with open(filename) as infile:
		for line in infile:
			if line.startswith("#"):
				continue
			tks = line.split()
			if len(tks) == 0:
				continue
			assert len(tks) >= 4
			assert tks[2] == RIGHT_ARROW
			p = float(tks[0])
			assert p >= 0
			lhs = tks[1]
			nonterminals.add(lhs)

			if p == 0.0:
				raise ValueError("Zero probability in WCFG.")
			
			rhs = tuple(tks[3:])
			prods.append( (p,lhs,rhs))
	assert START_SYMBOL in nonterminals

	terminals = set()
	for p,lhs,rhs in prods:
		for s in rhs:
			if not s in nonterminals:
				terminals.add(s)
	my_pcfg = WCFG()
	for p,lhs,rhs in prods:
		prod = (lhs,) + rhs
		my_pcfg.productions.append(prod)
		my_pcfg.parameters[prod] = p
		
	my_pcfg.start =  START_SYMBOL
	my_pcfg.terminals = terminals
	my_pcfg.nonterminals = nonterminals
	my_pcfg.set_log_parameters()
	return my_pcfg


import collections

def convert_pcfg_to_cfg(pcfg, discard_zero = True):
	my_cfg = CFG()
	my_cfg.start = pcfg.start
	my_cfg.nonterminals = set(pcfg.nonterminals)
	my_cfg.terminals = set(pcfg.terminals)
	my_cfg.productions = set()
	assert my_cfg.start in my_cfg.nonterminals
	for p in pcfg.productions:
		if pcfg.parameters[p] > 0 or not discard_zero:
			my_cfg.productions.add(p)
		assert len(p) == 2 or len(p) == 3
		if len(p) == 2:
			A,a = p
			assert A in my_cfg.nonterminals
			assert a in my_cfg.terminals
		if len(p) == 3:
			A,B,C = p
			assert A in my_cfg.nonterminals
			assert B in my_cfg.nonterminals
			assert C in my_cfg.nonterminals

def load_cfg_from_file(filename):
	grammar = CFG()
	with open(filename) as inf:

		for line in filename:
			if len(line) == 0 or line[0] == '#':
				continue
			tokens = line.strip().split()
			l = len(tokens)
			if l == 0:
				continue
			if l == 1:
				# this is a recoverable syntax error
				raise ValueError("Only one token on line")
			if l > 1 and tokens[1] != "->":
				# another error 
				raise ValueError("wrong separator: should be ->")
			lhs = tokens[0]
			rhs = tuple(tokens[2:])
			prod = (lhs,) +  rhs
			grammar.nonterminals.add(lhs)
			grammar.productions.add(prod)
	assert "S" in grammar.nonterminals
	grammar.start = "S"
	for prod in grammar.productions:
		if len(prod) == 2:
			a = prod[1]
			assert not a in nonterminals
			grammar.terminals.add(a)
	return grammar
	
class CFG:

	"""
	a CFG in CNF form
	"""
	def __init__(self):
		self.start = None
		self.nonterminals = set()
		self.productions = set()
		self.terminals = set()
	


	def compute_coreachable_set(self):
		"""
		return the set of all nonterminals that can generate a string.
		"""
		coreachable = set()
		iteration = 0
		
		prodmap = collections.defaultdict(list)
		for prod in self.productions:
			prodmap[prod[0]].append(prod)
		remaining = set(self.nonterminals)
		def prod_reachable(prod):
			for symbol in prod[1:]:
				if (not symbol in self.terminals) and (not symbol in coreachable):
					return False
			return True

		done_this_loop = 0
		while iteration == 0 or done_this_loop > 0:
			iteration += 1
			done_this_loop = 0
			for nt in remaining:
				for prod in prodmap[nt]:
					if prod_reachable(prod):
						done_this_loop += 1
						coreachable.add(nt)
						break
			remaining = remaining - coreachable


		return coreachable

	def compute_coreachable_productions(self, coreachable_nonterminals):
		"""
		Compute productions that can be used to generate something.
		"""
		good_productions = set()
		for prod in self.productions:
			for symbol in prod[1:]:
				if not symbol in coreachable_nonterminals and not symbol in self.terminals:
					break
			else:
				# we got to the end so its ok.
				good_productions.add(prod)
		return good_productions

	def compute_usable_productions(self, trim_set):
		"""
		return a list of the productions that are usable.
		"""
		tp = []
		for prod in self.productions:
			if prod[0] in trim_set:
				if len(prod) == 2:
					tp.append(prod)
				if prod[1] in trim_set and prod[2] in trim_set:
					tp.append(prod)
		return tp

	def compute_trim_set(self):
		"""
		return the set of all nonterminals A that can both generate a string 
		and have a context.
		"""
		#print "computing coreachable"
		coreachable = self.compute_coreachable_set()
		#print "coreachable", coreachable
		trim = set()
		good_productions = self.compute_coreachable_productions(coreachable)
		
		if self.start in coreachable:
			trim.add(self.start)
		done = len(trim)
		#print("start ", done)
		while done > 0:
			done = 0
			for prod in good_productions:
				if prod[0] in trim:
					#print(prod)
					for symbol in prod[1:]:
						if symbol in self.nonterminals and not symbol in trim:
							done += 1
							trim.add(symbol)
		#print "Trim set", trim
		return trim




class Multinomial:
	"""
	this represents a collection of productions with the same left hand side.
	we cache a bunch of samples for efficiency.
	"""

	def __init__(self, pcfg, nonterminal, random, cache_size=SAMPLE_CACHE_SIZE):
		self.cache_size = cache_size
		self.productions = [ prod for prod in pcfg.productions if prod[0] == nonterminal ]
		self.n = len(self.productions)
		self.nonterminal = nonterminal
		parameters = [ pcfg.parameters[prod] for prod in self.productions]
		self.p = np.array(parameters)/np.sum(parameters)
		self.rng = random
		self._sample()

	def _sample(self):
		self.cache = self.rng.choice(range(self.n), self.cache_size,True,self.p)
		self.cache_index = 0

	def sample_production(self):
		"""
		return the rhs of a production as a tuple of length 1 or 2
		"""
		if self.cache_index >= self.cache_size:
			self._sample()
		result = self.productions[self.cache[self.cache_index]]
		self.cache_index += 1
		return result


class Sampler:
	"""
	This object is used to sample from a PCFG.
	"""
	def __init__(self, pcfg,cache_size=SAMPLE_CACHE_SIZE,max_depth = SAMPLE_MAX_DEPTH,random=None):
		## construct indices for sampling
		assert pcfg.is_normalised()
		## For reproducibility we need to initialize with the same sequence of nonterminals.
		nts = list(pcfg.nonterminals)
		nts.sort()
		self.multinomials = { nt: Multinomial(pcfg,nt,random,cache_size) for nt in nts}
		self.start = pcfg.start
		self.max_depth = max_depth
		self.insider = InsideComputation(pcfg)
		self.mypcfg = pcfg

	def sample_production(self, lhs):
		return self.multinomials[lhs].sample_production()

	def sample_tree(self):
		return self._sample_tree(self.start, 0)

	def sample_string(self):
		return utility.collect_yield(self.sample_tree())

	def _sample_tree(self, nonterminal, current_depth):
		"""
		Sample a single tree from the pcfg, and throw an exception of max_depth is exceeded.
		"""
		if current_depth >= self.max_depth:
			raise ValueError("Too deep")

		prod = self.sample_production(nonterminal)
		if len(prod) == 2:
			# lexical rule
			return prod
		else:
			left_branch = self._sample_tree(prod[1],current_depth + 1)
			right_branch = self._sample_tree(prod[2],current_depth + 1)
			return (nonterminal, left_branch,right_branch)


	def estimate_string_entropy(self,samples,max_length=50,verbose=False):
		"""
		Estimate the string entropy and perplexity.
		"""
		total_length = 0.0
		total_samples = 0
		total_lps = 0.0
		total_lpt = 0.0
		total_lpb = 0.0

		for _ in range(samples):

			tree = self.sample_tree()
			s = collect_yield(tree)
			if True:
				total_samples += 1
				lps = self.insider.inside_log_probability(s)
				lpt = self.mypcfg.log_probability_derivation(tree)
				lpb = self.insider._bracketed_log_probability(tree)[self.start]
				total_length += len(s)
				if verbose: print(s,lpt, lpb,lps)
				total_lps += lps
				total_lpt += lpt
				total_lpb += lpb
		sentential_entropy = -total_lps/total_samples
		perplexity = math.exp(-total_lps/(total_length + total_samples))
		print("SentenceEntropy %f WordPerplexity %f " % (sentential_entropy, perplexity))


