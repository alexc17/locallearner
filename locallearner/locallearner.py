#locallearner.py
import os
from collections import Counter
import numpy as np
import wcfg
import neyessen
import ngram_counts
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
		# Path to a .clusters file for caching Ney-Essen results.
		# If set and the file exists, clusters are loaded from it.
		# If set and the file does not exist, clusters are computed and saved.
		# If None, clusters are computed fresh each time (no caching).
		self.cluster_file = None
		# Paths for caching bigram/trigram counts.
		# Same semantics as cluster_file: load if exists, compute+save if not.
		self.bigram_file = None
		self.trigram_file = None
		self.em_max_length = 10
		self.em_max_samples = 100000
		self.width = 1
		# Feature mode for NMF: 'marginal' (default) concatenates left and
		# right context cluster counts; 'joint' uses the Cartesian product
		# (left_cluster x right_cluster), capturing context dependencies.
		self.feature_mode = 'marginal'
		self.binary_smoothing = 0.0
		self.unary_smoothing = 0.0
		## These are the local distributional features (+1 because of the sentence boundary.)
		self.renyi = 5
		self.posterior_threshold= 0.9
		## Small sample factor: penalises rare words during NMF anchor selection.
		## Subtracts ssf/sqrt(n) from distance scores to correct for sampling noise.
		self.ssf = 1.0
		## Ratio threshold for NMF auto-stopping.
		## The candidate's scaled chi-squared statistic must exceed the expected
		## null maximum by at least this factor to be accepted as a real kernel.
		## Higher values = more conservative (fewer nonterminals).
		self.nmf_ratio_threshold = 10.0
		## Minimum KL divergence for NMF auto-stopping.
		## A candidate kernel must have KL(candidate || nearest_existing_kernel)
		## above this threshold to be accepted. Ensures each kernel represents
		## a genuinely distinct context distribution.
		self.nmf_min_divergence = 0.1

		## Stopping thresholds for auto-detection.
		## Distance threshold: stop if candidate's distance from the
		## hyperplane (affine span) of existing kernels is below this.
		## Uses Gram-Schmidt distances, which are more robust than
		## convex hull (FW) distances for ambiguous/low-count words.
		## Set to 0.0 to disable.
		self.distance_threshold = 0.0
		## Bootstrap: stop if candidate's p-value exceeds this threshold
		## (candidate is consistent with null hypothesis of sampling noise).
		self.boot_p_threshold = 0.01
		## Cramér's V: candidate is not distinct if min V < this threshold.
		## Higher values = more conservative (fewer NTs detected).
		self.cramers_v_threshold = 0.05


	def find_kernels(self,verbose=True):
		self.stride =  (self.number_clusters + 1)
		self.number_features = 2 * self.width *self.stride
		
		print("Ney Essen clustering.")
		self.do_clustering()
		self.set_start_vector()
		self.compute_unigram_features()

		if self.feature_mode == 'joint':
			print("Computing joint context features.")
			self.compute_joint_features()

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


	def ensure_bigrams(self):
		"""Ensure self.bigram_counts is populated (from cache or corpus)."""
		if hasattr(self, 'bigram_counts'):
			return
		if self.bigram_file and os.path.exists(self.bigram_file):
			self.bigram_counts = ngram_counts.load_bigrams(self.bigram_file)
			print(f"Loaded {len(self.bigram_counts)} bigrams from {self.bigram_file}")
		else:
			self.bigram_counts = ngram_counts.count_bigrams(self.sentences)
			if self.bigram_file:
				ngram_counts.save_bigrams(self.bigram_counts, self.bigram_file)
				print(f"Saved {len(self.bigram_counts)} bigrams to {self.bigram_file}")

	def ensure_trigrams(self):
		"""Ensure self.trigram_counts is populated (from cache or corpus)."""
		if hasattr(self, 'trigram_counts'):
			return
		if self.trigram_file and os.path.exists(self.trigram_file):
			self.trigram_counts = ngram_counts.load_trigrams(self.trigram_file)
			print(f"Loaded {len(self.trigram_counts)} trigrams from {self.trigram_file}")
		else:
			self.trigram_counts = ngram_counts.count_trigrams(self.sentences)
			if self.trigram_file:
				ngram_counts.save_trigrams(self.trigram_counts, self.trigram_file)
				print(f"Saved {len(self.trigram_counts)} trigrams to {self.trigram_file}")

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

		self.ensure_bigrams()
		BDY = ngram_counts.BOUNDARY

		# Build features from bigram counts instead of iterating over sentences.
		# Each bigram (left, right) contributes:
		#   - right's left-context feature at cluster(left), offset=0
		#   - left's right-context feature at cluster(right), offset=stride
		# Boundary token maps to cluster 0.
		for (w1, w2), count in self.bigram_counts.items():
			lc = 0 if w1 == BDY else self.clusters.get(w1, 0)
			rc = 0 if w2 == BDY else self.clusters.get(w2, 0)
			# w2's left context is w1
			if w2 != BDY:
				ai = self.word2idx.get(w2)
				if ai is not None:
					self.unigram_features[ai, lc] += count
			# w1's right context is w2
			if w1 != BDY:
				ai = self.word2idx.get(w1)
				if ai is not None:
					self.unigram_features[ai, self.stride + rc] += count

		self.unigram_features[self.alphabet_size,:] = self.start_vector

	def set_joint_start_vector(self):
		"""Start vector for joint (left_cluster x right_cluster) features.

		The start symbol represents an abstract word occurring in a
		length-1 sentence, so both its left and right contexts are
		sentence boundaries (cluster index 0).  The joint feature
		vector has a single nonzero entry at (0, 0) = index 0.
		"""
		C = self.number_clusters + 1
		self.joint_number_features = C * C
		self.joint_start_vector = np.zeros(self.joint_number_features)
		self.joint_start_vector[0] = 1  # (boundary, boundary)

	def compute_joint_features(self):
		"""Build joint context feature matrix: V x (C+1)^2.

		For each word occurrence, record the (left_cluster, right_cluster)
		pair as a single joint feature.  Sentence boundaries map to
		cluster 0.  This captures dependencies between left and right
		contexts that the marginal (concatenated) features miss.

		The matrix is stored as self.joint_features with shape
		(alphabet_size+1, (C+1)^2).  The last row is the start symbol.
		"""
		C = self.number_clusters + 1
		self.joint_number_features = C * C
		self.joint_features = np.zeros(
			(self.alphabet_size + 1, self.joint_number_features))

		self.ensure_trigrams()
		BDY = ngram_counts.BOUNDARY

		# Build features from trigram counts instead of iterating over sentences.
		# Each trigram (left, center, right) contributes to center's feature
		# at (cluster(left), cluster(right)).
		for (w1, w2, w3), count in self.trigram_counts.items():
			ai = self.word2idx.get(w2)
			if ai is None:
				continue
			lc = 0 if w1 == BDY else self.clusters.get(w1, 0)
			rc = 0 if w3 == BDY else self.clusters.get(w3, 0)
			self.joint_features[ai, lc * C + rc] += count

		# Start symbol
		self.set_joint_start_vector()
		self.joint_features[self.alphabet_size, :] = self.joint_start_vector

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
		# Note: init_nmf_with_kernels always uses marginal features
		# because it's used for parameter estimation, not kernel selection.
		self.nmf = nmf.NMF(self.unigram_features[fidx,:], fwords, ssf=self.ssf)

		# The start vector is artificial (not estimated from data), so:
		# 1. Shrinkage destroys it (~92% toward global mean). Restore it.
		# 2. The count of 2 is meaningless (deterministic, not sampled).
		#    Set a large effective count so SSF/bootstrap treat it as exact.
		start_idx = len(fidx)-1
		start_normalized = self.start_vector / np.sum(self.start_vector)
		self.nmf.data[start_idx,:] = start_normalized
		self.nmf.raw_data[start_idx,:] = start_normalized
		self.nmf.counts[start_idx] = 1e8
		self.nmf.start(start_idx)
		self.kernels = [ kernels[0] ]
		for a in kernels[1:]:
			ai = self.word2idx[a]
			self.nmf.add_basis(ai)
			self.kernels.append(a)
		self.nmf.initialise_frank_wolfe()

	def _build_word_context_matrix(self):
		"""Build word-level context count matrix from bigram counts.

		For each word w, count how many times each vocabulary word appears
		immediately to the left or right of w. This creates a V×2V matrix
		of raw bigram counts, independent of the Ney-Essen cluster space.

		The matrix is computed once and cached for reuse across multiple
		stopping-criterion checks during NMF auto-detection.

		Columns 0..V-1 are left-context counts, V..2V-1 are right-context.
		"""
		if hasattr(self, '_word_ctx_matrix'):
			return self._word_ctx_matrix

		self.ensure_bigrams()
		BDY = ngram_counts.BOUNDARY

		V = self.alphabet_size
		# Use a dense matrix; V≈1000 so V×2V ≈ 2M entries, fine.
		M = np.zeros((V, 2 * V), dtype=np.float64)

		# Build from bigram counts. Skip boundary tokens (same as
		# the original: only real word-word bigrams are counted).
		for (w1, w2), count in self.bigram_counts.items():
			if w1 == BDY or w2 == BDY:
				continue
			li = self.word2idx.get(w1)
			ri = self.word2idx.get(w2)
			if li is not None and ri is not None:
				# w2's left context is w1
				M[ri, li] += count
				# w1's right context is w2
				M[li, V + ri] += count

		self._word_ctx_matrix = M
		return M

	def _word_context_distinct(self, candidate, anchors, verbose=False):
		"""Test whether candidate's word-level context distribution is
		distinct from all existing anchors, using a chi-squared test on
		raw corpus bigram counts with an effect-size guard.

		For each existing anchor a_j, form a 2-row contingency table:
		  row 0 = candidate's context counts (over all vocabulary words)
		  row 1 = anchor a_j's context counts
		and run scipy.stats.chi2_contingency, filtering to columns with
		sufficient counts (sum > 5).

		Two checks are applied:
		  1. p-value: if the candidate is NOT significantly different from
		     some anchor (max_p >= 0.001), stop — they share an NT.
		  2. Cramér's V (effect size): if the candidate has a very small
		     V with some anchor (min_V < 0.05), stop — even if the
		     p-value is tiny (as happens with large corpora), the
		     distributions are practically identical.

		Returns:
			(is_distinct, max_p_value, min_v, closest_anchor, details_str)
		"""
		M = self._build_word_context_matrix()
		from scipy.stats import chi2_contingency

		ci = self.word2idx.get(candidate)
		if ci is None:
			return True, 0.0, float('inf'), None, ""

		c_row = M[ci, :]
		if np.sum(c_row) == 0:
			return True, 0.0, float('inf'), None, ""

		max_p = 0.0
		min_v = float('inf')
		closest_p = None
		closest_v = None
		details = []

		for anchor in anchors:
			if anchor == self.start_symbol:
				continue
			ai = self.word2idx.get(anchor)
			if ai is None:
				continue

			a_row = M[ai, :]
			if np.sum(a_row) == 0:
				continue

			# Build 2-row contingency table, keep columns with sum > 5
			table = np.vstack([c_row, a_row])
			col_sums = table.sum(axis=0)
			mask = col_sums > 5
			if np.sum(mask) < 2:
				continue

			table_filtered = table[:, mask]

			try:
				chi2_stat, p_value, dof, _ = chi2_contingency(table_filtered)
			except ValueError:
				continue

			# Cramér's V: effect size, independent of sample size.
			# For a 2×k table: V = sqrt(chi2 / (n * (min(2,k)-1))) = sqrt(chi2/n)
			n_total = table_filtered.sum()
			cramers_v = math.sqrt(chi2_stat / n_total) if n_total > 0 else 0.0

			if verbose:
				print(f"    chi2 {candidate} vs {anchor}: "
					  f"chi2={chi2_stat:.1f}, df={dof}, p={p_value:.2e}, "
					  f"V={cramers_v:.4f}, ncols={np.sum(mask)}")

			details.append(f"{anchor}:p={p_value:.1e},V={cramers_v:.3f}")

			if p_value > max_p:
				max_p = p_value
				closest_p = anchor
			if cramers_v < min_v:
				min_v = cramers_v
				closest_v = anchor

		# Distinct requires BOTH:
		#   1. All chi-squared tests are significant (max_p < 0.001)
		#   2. All effect sizes are non-trivial (min_V >= 0.05)
		# The V threshold guards against the chi-squared test's
		# excess power with large corpora: even words from the same NT
		# will be "significantly different" (tiny p), but their V will
		# be small, indicating near-identical distributions.
		is_distinct = (max_p < 0.001) and (min_v >= self.cramers_v_threshold)
		closest = closest_p if max_p >= 0.001 else closest_v
		detail_str = ', '.join(details[-3:])
		return is_distinct, max_p, min_v, closest, detail_str

	def do_nmf(self,verbose=False):
		# Clear cached word-context matrix so it's rebuilt for this corpus
		if hasattr(self, '_word_ctx_matrix'):
			del self._word_ctx_matrix

		## Filter out the low frequency ones.
		fidx = []
		fwords = []
		for i,w in enumerate(self.idx2word):
			if self.lexical_counts[w] >= self.min_count_nmf:
				fidx.append(i)
				fwords.append(w)
		fidx.append(self.alphabet_size)
		fwords.append(self.start_symbol)
		if self.feature_mode == 'joint':
			feature_matrix = self.joint_features[fidx, :]
			sv = self.joint_start_vector
		else:
			feature_matrix = self.unigram_features[fidx, :]
			sv = self.start_vector
		self.nmf = nmf.NMF(feature_matrix, fwords, ssf=self.ssf)
		self.nmf.use_gram_schmidt = True

		# The start vector is artificial (not estimated from data), so:
		# 1. Shrinkage destroys it (~92% toward global mean). Restore it.
		# 2. The count of 2 is meaningless (deterministic, not sampled).
		#    Set a large effective count so SSF/bootstrap treat it as exact.
		start_idx = len(fidx)-1
		start_normalized = sv / np.sum(sv)
		self.nmf.data[start_idx,:] = start_normalized
		self.nmf.raw_data[start_idx,:] = start_normalized
		self.nmf.counts[start_idx] = 1e8
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

		# Record stopping diagnostics for external analysis
		self.stop_reason = None       # 'bootstrap', 'word_ctx', 'max_nt', 'exhausted'
		self.stop_boot_p = None       # bootstrap p-value at stop
		self.stop_wctx_max_p = None   # word-context max p-value at stop
		self.stop_wctx_min_v = None   # word-context min Cramér's V at stop
		self.candidate_diagnostics = []  # per-candidate stats for threshold analysis

		while len(self.kernels) < max_nt:
			a, ai, d = self.nmf.find_but_dont_add()
			if a is None:
				print("No more candidates.")
				self.stop_reason = 'exhausted'
				break

			# In auto mode, two independent stopping tests must BOTH
			# pass for a candidate to be accepted as a new kernel:
			#
			# 1. Bootstrap test (cluster feature space):
			#    Sample from the candidate's FW-estimated mixture of
			#    existing kernels with the candidate's word count.
			#    If the candidate's distance is not an outlier (p > 0.01),
			#    it is consistent with sampling noise -> stop.
			#
			# 2. Word-context test (raw bigram co-occurrences):
			#    Compare the candidate's word-level context distribution
			#    to each existing anchor via chi-squared + Cramer's V.
			#    If the candidate is not distinct from some anchor -> stop.
			#
			# Either test triggering is sufficient to stop.
			if auto_mode and len(self.kernels) >= min_nt:
				# Test 0: Distance floor (hyperplane / Gram-Schmidt)
				# If the candidate's distance from the affine span of
				# existing kernels is below a threshold, stop.
				if self.distance_threshold > 0 and d < self.distance_threshold:
					if verbose:
						print(f"  candidate {a}: count={self.lexical_counts[a]}, "
							  f"d={d:.6f} < threshold {self.distance_threshold:.4f}")
					self.stop_reason = 'distance'
					self.candidate_diagnostics.append({
						'step': len(self.kernels),
						'word': a,
						'count': self.lexical_counts.get(a, 0),
						'distance': float(d),
						'boot_p': None,
						'stop_reason': 'distance',
					})
					print(f"Stopping (distance): candidate {a} too close "
						  f"to hyperplane (d={d:.6f} < {self.distance_threshold:.4f}) "
						  f"with {len(self.kernels)} nonterminals.")
					break

				# Test 1: Bootstrap
				boot_p, d_cand, boot_dists = \
					self.nmf.bootstrap_null_distance(ai, n_bootstrap=500)

				if verbose:
					med = np.median(boot_dists)
					q95 = np.percentile(boot_dists, 95)
					print(f"  candidate {a}: count={self.lexical_counts[a]}, "
						  f"d={d_cand:.6f}, "
						  f"null median={med:.6f}, 95th={q95:.6f}, "
						  f"bootstrap p={boot_p:.4f}")

				# Record per-candidate diagnostics
				cand_diag = {
					'step': len(self.kernels),
					'word': a,
					'count': self.lexical_counts.get(a, 0),
					'distance': float(d_cand),
					'boot_p': float(boot_p),
				}

				if boot_p > self.boot_p_threshold:
					self.stop_reason = 'bootstrap'
					self.stop_boot_p = boot_p
					cand_diag['stop_reason'] = 'bootstrap'
					self.candidate_diagnostics.append(cand_diag)
					print(f"Stopping (bootstrap): candidate {a} consistent "
						  f"with null (p={boot_p:.4f}, d={d_cand:.6f}) "
						  f"with {len(self.kernels)} nonterminals.")
					break

				# Test 2: Word-context distinctness
				is_distinct, max_p, min_v, closest, detail = \
					self._word_context_distinct(
						a, self.kernels, verbose=verbose)

				cand_diag['min_v'] = float(min_v)
				cand_diag['max_p'] = float(max_p)
				cand_diag['closest_v'] = closest
				cand_diag['wctx_distinct'] = is_distinct

				if verbose:
					print(f"    word-ctx: max_p={max_p:.2e}, "
						  f"min_V={min_v:.4f}, closest={closest}, "
						  f"distinct={is_distinct}")

				if not is_distinct:
					if max_p >= 0.001:
						reason = f"p={max_p:.2e}"
					else:
						reason = f"V={min_v:.4f}"
					self.stop_reason = 'word_ctx'
					self.stop_wctx_max_p = max_p
					self.stop_wctx_min_v = min_v
					self.stop_boot_p = boot_p
					cand_diag['stop_reason'] = 'word_ctx'
					self.candidate_diagnostics.append(cand_diag)
					print(f"Stopping (word-ctx): candidate {a} not "
						  f"distinct from {closest} ({reason}) "
						  f"with {len(self.kernels)} nonterminals.")
					break

				self.candidate_diagnostics.append(cand_diag)

			if verbose:
				print(f"Adding kernel {a}, count={self.lexical_counts[a]}, "
					  f"distance={d:.6f}")
			self.nmf.add_basis(ai)
			self.kernels.append(a)

		if auto_mode:
			if self.stop_reason is None:
				self.stop_reason = 'max_nt'
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
		"""Run Ney-Essen clustering, or load from cache if cluster_file is set."""
		if self.cluster_file and os.path.exists(self.cluster_file):
			self.clusters, meta = neyessen.load_cluster_dict(self.cluster_file)
			n_c = meta.get('n_clusters', '?')
			seed = meta.get('seed', '?')
			print(f"Loaded clusters from {self.cluster_file} "
				  f"(n_clusters={n_c}, seed={seed}, {len(self.clusters)} words)")
			return
		myc = neyessen.Clustering()
		myc.clusters = self.number_clusters
		self.clusters = myc.cluster(self.sentences,seed=self.seed,verbose=verbose)
		if self.cluster_file:
			neyessen.save_cluster_dict(
				self.clusters, self.cluster_file,
				n_clusters=self.number_clusters, seed=self.seed)
			print(f"Saved clusters to {self.cluster_file}")






