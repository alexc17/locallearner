#locallearner.py
from collections import Counter
import numpy as np
import wcfg
import neyessen
import nmf
import math
import utility
from scipy.stats import chi2_contingency

# This class is for a purely local learner that uses only the contexts to infer a grammar:
# option to produce a smoothed grammar or a non smoothed one.


def renyi_divergence(v1,v2, alpha):
	
	## v1 and v2 are unnormalised vectors of counts.
	## Compute the Renyi divergence.
	## Umsnoothed - 
	n1 = v1/np.sum(v1)
	n2 = v2/np.sum(v2)
	assert np.sum(v1) > 0
	assert np.sum(v2) > 0

	total = 0
	for x, y in np.nditer([n1,n2]):
		if x > 0 and y > 0:
			total +=  x * (x/y)** (alpha - 1)
		elif x > 0 and y == 0:
			return math.inf
	return math.log(total)/ ( alpha - 1)

class LocalLearner:

	def __init__(self,filename):
		"""
		Takes one file as input.
		"""
		## Load it up and count various things so we can set hyperparameters to sensible values.
		self.lexical_counts = Counter()
		self.start_symbol="S"
		self.number_samples = 0
		self.number_tokens = 0
		self.max_length = 0
		self.sentences = []
		with open(filename) as inf:
			for lineno, line in enumerate(inf, 1):
				if line[0] == '#':
					continue
				s = line.split()
				l = len(s)
				if l > 0:
					for w in s:
						if not w[0].islower():
							raise ValueError(
								f"Token '{w}' on line {lineno} does not "
								f"start with a lowercase letter. "
								f"Is this a corpus file (.strings/.yld)?")
						self.lexical_counts[w] += 1
					self.number_samples += 1
					self.number_tokens += l
					self.max_length = max(l,self.max_length)
					self.sentences.append(tuple(s))
		self.alphabet_size = len(self.lexical_counts)
		self.idx2word = list(self.lexical_counts)
		self.word2idx = { a:i for i,a in enumerate(self.idx2word)}

		# Set hyperparameters here.
		# Random seed used for the Ney Essen clustering.
		self.seed = None
		# Number of nonterminals. Set to 0 or None for automatic detection
		# using the SSF-corrected distance threshold.
		self.nonterminals = 2
		self.min_nonterminals = 2
		self.max_nonterminals = 20
		self.min_count_nmf = 100
		self.number_clusters = 10
		self.em_max_length = 10
		self.em_max_samples = 100000
		self.width = 1
		self.binary_smoothing = 0.0
		self.unary_smoothing = 0.0
		## These are the local distributional features (+1 because of the sentence boundary.)
		self.renyi = 5
		self.posterior_threshold= 0.9
		## Small sample factor: penalises rare words during NMF anchor selection.
		## Subtracts ssf/sqrt(n) from distance scores to correct for sampling noise.
		self.ssf = 1.0
		## Significance threshold for NMF auto-stopping.
		## The survival probability of the best candidate under a scaled
		## chi-squared null model must be below this to add a new kernel.
		## Lower values = more conservative (fewer nonterminals).
		self.nmf_significance = 0.001


	def find_kernels(self,verbose=True):
		self.stride =  (self.number_clusters + 1)
		self.number_features = 2 * self.width *self.stride
		
		print("Ney Essen clustering.")
		self.do_clustering()
		self.set_start_vector()
		self.compute_unigram_features()
		print("Non negative matrix factorisation.")
		self.do_nmf(verbose=verbose)
		print(self.kernels)
		for a in self.kernels[1:]:
			print(a, self.clusters[a])
		return self.kernels

	def learn_wcfg_from_kernels_renyi(self,kernels, verbose=True):
		"""
		We alread have the kernels.
		returns a PCFG that approximates the conditional distribution of trees given strings.

		"""
		## Computed quantities from hyperparams
		self.stride =  (self.number_clusters + 1)
		self.number_features = 2 * self.width *self.stride
		
		print("Ney Essen clustering (again)")
		self.do_clustering()
		self.set_start_vector()
		self.compute_unigram_features()
		#self.kernels = kernels
		self.binary_smoothing = 1.0 / ( self.number_tokens * self.nonterminals ** 2)
		self.unary_smoothing = 1.0 / ( self.number_tokens * self.alphabet_size)
		self.init_nmf_with_kernels(kernels)
		print("Estimating unary parameters: FW ")
		self.compute_unary_parameters_fw()
		
		print("Computing clustered bigram features with posterior threshold ", self.posterior_threshold)
		self.compute_clustered_bigram_features()
		
		print("Estimating binary parameters: Renyi")
		self.compute_binary_parameters_renyi()
		
		# for a in self.kernels:
		# 	for b in self.kernels[1:]:
		# 		for c in self.kernels[1:]:
		# 			print(a,"->",b,c,"=", self.binary_parameters[(a,b,c)])
		self.set_nonterminal_labels()
		self.make_raw_wcfg()
		#self.make_pcfg()
		return self.output_grammar

	def learn(self,binary_mode='renyi',verbose=True):
		"""
		returns a PCFG that approximates the conditional distribution of trees given strings.

		one given hyper parameter: the number of nonterminals.
		This avoids the use of an RNN to detect termination.
		"""
		## Computed quantities from hyperparams
		self.find_kernel(verbose=verbose)
		self.binary_smoothing = 1.0 / ( self.number_tokens * self.nonterminals ** 2)
		self.unary_smoothing = 1.0 / ( self.number_tokens * self.alphabet_size)
		# self.stride =  (self.number_clusters + 1)
		# self.number_features = 2 * self.width *self.stride
		
		# print("Ney Essen clustering.")
		# self.do_clustering()
		# self.set_start_vector()
		# self.compute_unigram_features()
		# print("Non negative matrix factorisation.")
		# self.do_nmf(verbose=verbose)
		# print(self.kernels)
		
		print("Estimating unary parameters: Frank-Wolfe ")
		self.compute_unary_parameters_fw()
		#print(self.unary_parameters)
		#self.compute_bigram_features()
		
		print("Computing clustered bigram features with posterior threshold ", self.posterior_threshold)

		self.compute_clustered_bigram_features()
		
		
		if binary_mode=='renyi':
			print("Estimating binary parameters: Renyi")
			self.compute_binary_parameters_renyi()
		else:
			print("Estimating binary parameters: Frank-Wolfe")
			self.compute_binary_parameters_fw()
		# for a in self.kernels:
		# 	for b in self.kernels[1:]:
		# 		for c in self.kernels[1:]:
		# 			print(a,"->",b,c,"=", self.binary_parameters[(a,b,c)])
		self.set_nonterminal_labels()
		self.make_raw_wcfg()
		self.make_pcfg()
		

	def reestimate(self):
		print("Inside Outside reestimation: max samples %d, max length %d" % (self.em_max_samples,self.em_max_length))
		try:
			self.reestimated_pcfg = self.output_pcfg.estimate_inside_outside_from_list(self.sentences, 
				self.em_max_length, 
				self.em_max_samples)
			return self.reestimated_pcfg
		except utility.ParseFailureException:
			print("Failed to parse; returning original pcfg")
			return self.output_pcfg 

	def terminal_expectation(self,w):
		return self.lexical_counts[w]/self.number_samples

	def chi2_pvalue(self,a,b):
		"""
		p value for the assumption that a and b are congruent.
		"""
		ai = self.word2idx[a]
		bi = self.word2idx[b]
		af = self.unigram_features[ai,:]
		bf = self.unigram_features[bi,:]
		s = af + bf
		ab = self.unigram_features[(ai,bi),:]
		return chi2_contingency(ab[:,(s > 5)])[1]


	def set_start_vector(self):
		self.start_vector = np.zeros(self.number_features)
		n = self.number_clusters + 1
		for j in range(2 * self.width):
			self.start_vector[j * n] = 1

	def get_from_sentence(self,i,s):
		if i < 0:
			return 0
		if i >= len(s):
			return 0
		return self.clusters[s[i]]

	def compute_unigram_features(self):
		## The final row is the start vector, 
		#the distribution of an abstract symbol that coccurs only in length 1 strings.

		self.unigram_features = np.zeros((self.alphabet_size+1, self.number_features))
		
		
		for s in self.sentences:
			

			for i,w in enumerate(s):
				ai = self.word2idx[w]
				offset = 0
				for j in range(1,self.width+1):
					self.unigram_features[ai,self.get_from_sentence(i-j,s) + offset] += 1
					offset += self.stride
					self.unigram_features[ai,self.get_from_sentence(i+j,s) + offset] += 1
					offset += self.stride
		self.unigram_features[self.alphabet_size,:] = self.start_vector


	def compute_raw_bigram_features(self):
		"""
		This uses the actual bigrams which are probably too sparse.
		"""
		## nk is the number of non start nonterminals
		nk = len(self.kernels) - 1
		#print(nk)
		kmap = { a:i for i,a in enumerate(self.kernels[1:])}
		self._compute_cluster_bigram_features(kmap)
		#print(self.bigram_features)

	def compute_clustered_bigram_features(self):
		## Use unary parameters
		kmap  = { a:i for i,a in enumerate(self.kernels[1:])}
		for (a,b),p in self.unary_parameters.items():
			posterior = p/self.terminal_expectation(b)
			if posterior > self.posterior_threshold:
				if a != 'S':
					kmap[b] = kmap[a]
		#print("KMAP",kmap)
		self._compute_cluster_bigram_features(kmap)

	def _compute_cluster_bigram_features(self,kmap):
		"""
		Compute features of cluster bigrams.
		Since we are just using the Renyi thing this is ok.

		"""
		## nk is the number of non start nonterminals
		nk = len(self.kernels) - 1
		#
		self.bigram_features = np.zeros((nk,nk,self.number_features))
		
		for s in self.sentences:
			for i,(u,v) in enumerate(zip(s,s[1:])):
				uidx = kmap.get(u,-1)
				vidx = kmap.get(v,-1)
				if uidx > -1 and vidx > -1:
					offset = 0
					for j in range(1,self.width+1):
						self.bigram_features[uidx,vidx,self.get_from_sentence(i-j,s) + offset] += 1
						offset += self.stride
						# plus 1 as it is a bigram
						self.bigram_features[uidx,vidx,self.get_from_sentence(i+j+1,s) + offset] += 1
						offset += self.stride
		self.bigram_feature_sums = np.sum(self.bigram_features,axis=2)
		self.bigram_feature_sums_left = np.sum(self.bigram_features,axis=(1,2))
		self.bigram_feature_sums_right = np.sum(self.bigram_features,axis=(0,2))



	def init_nmf_with_kernels(self, kernels):
		## Given known clusters initialise the NMF so we can do the Frank-Wolfe estimation
		## for the unary rules.
		fidx = []
		fwords = []
		for i,w in enumerate(self.idx2word):
			fidx.append(i)
			fwords.append(w)
		fidx.append(self.alphabet_size)
		fwords.append(self.start_symbol)
		self.nmf = nmf.NMF(self.unigram_features[fidx,:], fwords, ssf=self.ssf)

		# The start vector is artificial (not estimated from data), so shrinkage
		# destroys it by blending it ~92% toward the global mean. Restore it.
		start_idx = len(fidx)-1
		self.nmf.data[start_idx,:] = self.start_vector / np.sum(self.start_vector)
		self.nmf.start(start_idx)
		self.kernels = [ kernels[0] ]
		for a in kernels[1:]:
			ai = self.word2idx[a]
			self.nmf.add_basis(ai)
			self.kernels.append(a)
		self.nmf.initialise_frank_wolfe()

	def do_nmf(self,verbose=False):
		## Filter out the low frequency ones.
		fidx = []
		fwords = []
		for i,w in enumerate(self.idx2word):
			if self.lexical_counts[w] >= self.min_count_nmf:
				fidx.append(i)
				fwords.append(w)
		fidx.append(self.alphabet_size)
		fwords.append(self.start_symbol)
		self.nmf = nmf.NMF(self.unigram_features[fidx,:], fwords, ssf=self.ssf)

		# The start vector is artificial (not estimated from data), so shrinkage
		# destroys it by blending it ~92% toward the global mean. Restore it.
		start_idx = len(fidx)-1
		self.nmf.data[start_idx,:] = self.start_vector / np.sum(self.start_vector)
		self.nmf.start(start_idx)
		self.kernels = [self.start_symbol]
		assert not self.start_symbol in self.word2idx
		self.nmf.excluded.add(start_idx)

		auto_mode = not self.nonterminals
		if auto_mode:
			max_nt = self.max_nonterminals
			min_nt = self.min_nonterminals
		else:
			max_nt = self.nonterminals
			min_nt = self.nonterminals

		while len(self.kernels) < max_nt:
			a, ai, d = self.nmf.find_but_dont_add()
			if a is None:
				print("No more candidates.")
				break

			# In auto mode, test whether the candidate is a significant
			# outlier using the scaled chi-squared model.
			if auto_mode and len(self.kernels) >= min_nt:
				sig = self.nmf.candidate_significance(ai)
				if verbose:
					print(f"  candidate {a}: survival_prob={sig['survival_prob']:.2e}, "
						  f"ratio={sig['ratio']:.1f}, "
						  f"scaled_stat={sig['scaled_stat']:.1f}, "
						  f"null_max={sig['expected_null_max']:.1f}")
				if sig['survival_prob'] > self.nmf_significance:
					print(f"Stopping: candidate {a} not significant "
						  f"(p={sig['survival_prob']:.4f} > {self.nmf_significance}) "
						  f"with {len(self.kernels)} nonterminals.")
					break

			if verbose:
				print(f"Adding kernel {a}, count={self.lexical_counts[a]}, "
					  f"distance={d:.6f}")
			self.nmf.add_basis(ai)
			self.kernels.append(a)

		if auto_mode:
			self.nonterminals = len(self.kernels)
			print(f"Auto-detected {self.nonterminals} nonterminals.")
		self.nmf.initialise_frank_wolfe()


	def compute_unary_parameters_fw(self):
		## USe FRANK Wolfe to estimate the unary parameters.
		self.unary_parameters = {}
		for a in self.idx2word:
			self.unary_parameters.update(self.test_rhs_lexical_fw(a))

	def test_rhs_lexical_fw(self, a):
		"""
		Test all lexical rules with a on the rhs.
		"""
		ai = self.word2idx[a]
		e = self.terminal_expectation(a)
		y = self.unigram_features[ai,:]
		y /= np.sum(y)
		x,d2 = self.nmf.estimate_frank_wolfe(y)
		params =  x * e
		xi = {}

		for i,b in enumerate(self.kernels):
			xi[ (b,a) ] = params[i]
		#print(a,xi)
		return xi



	def compute_binary_parameters_renyi(self):
		"""
		Use the Renyi divergence based on the local cluster counts.
		"""
		nk = len(self.kernels)
		self.binary_parameters = {}
		for a in range(nk):
			for b in range(1,nk):
				for c in range(1,nk):
					alpha = self.compute_binary_parameters_renyi_single(a,b,c,self.renyi)
					self.binary_parameters[ (self.kernels[a],self.kernels[b],self.kernels[c])] = alpha

	def compute_binary_parameters_renyi_single(self,a,b,c, renyi):
		"""
		Use the Renyi divergence based on the local cluster counts.
		Do this for a single production [[a]] -> [[b]],[[c]] whcih are indices into self.kernels
		"""	
		assert b > 0 and c > 0
		N = self.number_samples * 2 * self.width
		if a == 0:
			v0 = self.start_vector
		else:
			v0 = self.unigram_features[self.word2idx[self.kernels[a]]]
		v1 = self.bigram_features[b-1,c-1]
		ebc = self.bigram_feature_sums[b-1,c-1]/ N
		if ebc == 0:
			return 0
#		divergence = renyi_divergence(v0,v1,renyi)
		# OR
		stride = self.stride
		smoothing = 1
		divergence = max(renyi_divergence(v0[i*stride:(i+1)*stride],v1[i*stride:(i+1)*stride]+smoothing,renyi) for i in range(2 * self.width))
		assert divergence >= 0
		if divergence == math.inf:
			print("Infinite divergence",a,b,c)
			return 0
		teb = self.bigram_feature_sums_left[b-1]/ N
		tec = self.bigram_feature_sums_right[c-1]/ N
		assert teb > 0
		assert tec > 0
		#print(a,b,c,divergence,ebc,teb,tec,ebc/(teb * tec) )
		return math.exp(-divergence) * ebc/(teb * tec)




	def compute_binary_parameters_fw(self):
		## Use the FW method to estimate the binary parameters.
		self.binary_parameters = {}
		nk = len(self.kernels) - 1
		for c2 in range(nk):
			for c3 in range(nk):
				self.binary_parameters.update(self.test_rhs_binary_fw(c2,c3))

	def test_rhs_binary_fw(self, c2, c3):
		"""
		Use the Frank-Wolfe and the PMI to estiate the parameters of all binary productions
		with the right hand side of the two kernel indices (both > 1)
		"""
		bf = self.bigram_features[c2,c3,:]
		if np.sum(bf) == 0:
			return {}	
		a2 = self.kernels[1 + c2]
		a3 = self.kernels[1 + c3]
		## Hack to get the expectation of the bigram 
		e23 = np.sum(bf) / (self.number_samples * 2 * self.width)
		e2 = self.terminal_expectation(a2)
		e3 = self.terminal_expectation(a3)

		total_parameter = e23 / (e2 *e3)
		## Now divide this up by doing a FW.
		x,d2 = self.nmf.estimate_frank_wolfe(bf / np.sum(bf))
		params =  x * total_parameter
		xi = {}

		for i,a1 in enumerate(self.kernels):
			xi[ (a1, a2,a3) ] = params[i]
		return xi
		

	def set_nonterminal_labels(self):
		# Map from kernels to the corresponding nonterminals;
		self.kernel2nt = {}
		## Then make them up
		self.kernel2nt[self.kernels[0]] = "S"
		for a in self.kernels[1:]:
			self.kernel2nt[a] = "NT_" + a

	def convert_production(self, prod):
		if len(prod) == 2:
			a,b = prod
			return (self.kernel2nt[a],b)
		else:
			a,b,c = prod
			return (self.kernel2nt[a],self.kernel2nt[b],self.kernel2nt[c])

	def make_raw_wcfg(self):
		"""
		No smoothing/normalisation.
		"""
		self.output_grammar = wcfg.WCFG()
		g = self.output_grammar
		g.nonterminals = self.kernel2nt.values()
		g.start = "S"
		g.terminals = self.idx2word
		for d in [self.binary_parameters,self.unary_parameters]:
			for prod,alpha  in d.items():
				newprod = self.convert_production(prod)
				g.productions.append(newprod)
				smoothedalpha = max(alpha, self.unary_smoothing if len(prod) == 2 else self.binary_smoothing)
				if smoothedalpha > 0:
					g.parameters[newprod] = smoothedalpha
		g.set_log_parameters()
		return g
		


	def make_pcfg(self):
		"""
		Given a raw WCFG, convert it to a consistent PCFG that defines the same
		conditional distribution over strings.
		"""
		if not self.output_grammar.is_convergent():
			self.output_pcfg = self.output_grammar.renormalise_divergent_wcfg2()
			
			
		else:
			self.output_pcfg = self.output_grammar.copy()
		self.output_pcfg.renormalise()
		return self.output_pcfg



	def do_clustering(self,verbose=False):
		myc = neyessen.Clustering()
		myc.clusters = self.number_clusters
		self.clusters = myc.cluster(self.sentences,seed=self.seed,verbose=verbose)






