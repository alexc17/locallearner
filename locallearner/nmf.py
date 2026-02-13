# nmf.py
# NOn negative matrix factorisation of a matrix.

# Version 2 is optimized and has a better step size

import numpy as np
import numpy.linalg 
import scipy.spatial.distance
from scipy.stats import chi2
import math
import logging

max_iterations = 100
epsilon = 1e-5

class NMF:

	def __init__(self, data, index, ssf = 1.0, shrinkage=True):
		## Matrix of (ndata, nfeatures) 
		# all nonnegative, count data,.
		self.data = data.copy()
		self.ssf= ssf
		## Index is a list of n elements.
		self.index = index
		self.n = data.shape[0]
		self.f = data.shape[1]
		self.counts = np.zeros(self.n)

		# Compute global distribution from raw counts (before normalization).
		# This is the count-weighted mean of all word distributions.
		p_global = np.sum(self.data, axis=0)
		total = np.sum(p_global)
		if total > 0:
			p_global = p_global / total
		self.p_global = p_global

		for i in range(self.n):
			self.counts[i] = np.sum(self.data[i,:])
			self.data[i,:] = self.data[i,:]/ self.counts[i]

		# Store the raw (pre-shrinkage) normalized distributions.
		# Used by min_kl_to_bases for distributional distinctness checks,
		# since shrinkage compresses divergences between genuinely different kernels.
		self.raw_data = self.data.copy()

		# Apply shrinkage: blend each word's distribution toward the global mean.
		# Uses a Dirichlet-style prior with effective sample size = number of features.
		# lambda_i = f / (f + n_i): heavy shrinkage for rare words, negligible for frequent ones.
		if shrinkage:
			for i in range(self.n):
				lam = self.f / (self.f + self.counts[i])
				self.data[i,:] = (1 - lam) * self.data[i,:] + lam * p_global

		# list of indices
		self.bases = None
		## list of current basis vectors 
		self.M = None
		# Orthonormal basis for the span of the basis vectors. (list of vectors)
		self.orthonormal = None
		## ones that we don't consider.
		self.excluded = set()
		# When True, use Gram-Schmidt (hyperplane) distances for kernel
		# selection instead of Frank-Wolfe (convex hull) distances.
		self.use_gram_schmidt = False


	def small_sample_factor(self, n):
		return self.ssf / math.sqrt(n)

	def start(self, i):
		"""
		Pick the first dimension.
		"""
		self.M  = [ self.data[i,:] ] 
		self.bases = [i]
		self.gram_schmidt()


	def start_l2(self):
		"""
		Maximum L2 norm, with L1 norm == 1,
		will give us the most peaked vector.
		Subtract a noise correction so rare words don't win by sampling variance alone.
		"""
		norms = np.linalg.norm(self.data, axis=1)
		corrections = np.array([self.small_sample_factor(self.counts[j]) for j in range(self.n)])
		scores = norms - corrections
		i = np.argmax(scores)
		self.start(i)
		return self.index[i]


	def gram_schmidt(self):
		"""
		Compute an orthonormal basis for the set of context-vectors.
		"""
		basis = []
		for v in self.M:
			w = v - sum( np.dot(v,b)*b  for b in basis )
			if (w > 1e-10).any():  
				basis.append(w/np.linalg.norm(w))
		self.orthonormal =  np.array(basis)
		self.distances = []
		for i in range(self.n):
			self.distances.append(self.distance_from_hyperplane(self.data[i,:]))
		

	def frank_wolfe(self):
		"""
		Slower than Gram Schmidt but works in a wider range of cases without the full rank assumption.

		Compute distances for every node.
		"""
		self.initialise_frank_wolfe()
		self.distances = []
		for i,a in enumerate(self.index):
			#print(i,a)
			y = self.data[i,:]
			x,d = self.estimate_frank_wolfe(y)
			self.distances.append(d)

	def closest_point(self, x):
		result = np.zeros_like(x)
		for v in self.orthonormal:
			result += np.dot(x,v) * v
		return result

	def distance_from_hyperplane(self, x):
		xx = self.closest_point(x)
		return scipy.spatial.distance.euclidean(x,xx)





	def find_furthest(self,verbose=False):
		largestd = -1
		furthest = None

		for i in range(self.n):
			if not i in self.excluded:
				n = self.counts[i]
				d = self.distances[i]
				d -= self.small_sample_factor(n)
				if verbose:
					print(self.index[i], d, n)
				if d > largestd:
					largestd = d
					furthest = i
		if verbose:
			print("Largest d ", largestd)
		return furthest, largestd


	def add_basis(self, i, gram_schmidt=None):
		self.bases.append(i)
		self.excluded.add(i)
		self.M.append(self.data[i,:])
		if gram_schmidt is None:
			gram_schmidt = self.use_gram_schmidt
		if gram_schmidt: 
			self.gram_schmidt()
		else:
			self.frank_wolfe()

	

	def find_next_element(self):
		"""
		Find a next one and add it. We rprobably want to perform some termination tests before hand.
		"""
		a,d = self.find_furthest()
		print(a, d, self.index[a])
		self.add_basis(a)
		return self.index[a]

	def find_but_dont_add(self):
		a,d = self.find_furthest()
		if a is None:
			return None, None, d
		else:
			return self.index[a], a, d

	def candidate_significance(self, candidate_idx):
		"""
		Test whether the candidate word's distance from the current basis
		hyperplane is significantly larger than expected under the null
		hypothesis that all remaining variation is noise.

		Uses a scaled chi-squared model: for each word i, the normalised
		test statistic T_i = n_i * d_i^2 / (1 - ||p_i||_2^2) should be
		approximately scale * chi2(df) under the null, where:
		  - df = f - 1 - k  (features minus simplex constraint minus basis dim)
		  - scale is estimated empirically from the median of all T_i

		The candidate's scaled statistic is compared to the distribution of
		the maximum of m independent chi2(df) draws.

		Args:
			candidate_idx: index into self.data of the candidate word

		Returns:
			dict with:
			  'survival_prob': probability of seeing a value >= the candidate's
			    scaled statistic as the max of m chi2(df) draws (i.e. the
			    Bonferroni-corrected p-value). Small values = significant outlier.
			  'scaled_stat': candidate's test statistic divided by noise scale
			  'expected_null_max': expected maximum under the null
			  'ratio': scaled_stat / expected_null_max (>>1 means real signal)
			  'df': degrees of freedom
			  'scale': estimated noise scale factor
		"""
		k = len(self.bases)
		df = self.f - 1 - k
		if df <= 0:
			return {
				'survival_prob': 1.0, 'scaled_stat': 0.0,
				'expected_null_max': 0.0, 'ratio': 0.0,
				'df': df, 'scale': 1.0,
			}

		# Compute normalised test statistics for all non-excluded words
		test_stats = []
		for i in range(self.n):
			if i in self.excluded:
				continue
			d = self.distances[i]
			n = self.counts[i]
			l2_sq = np.dot(self.data[i,:], self.data[i,:])
			if l2_sq < 1.0 and n > 0:
				test_stats.append(n * d * d / (1 - l2_sq))

		if len(test_stats) < 2:
			return {
				'survival_prob': 1.0, 'scaled_stat': 0.0,
				'expected_null_max': 0.0, 'ratio': 0.0,
				'df': df, 'scale': 1.0,
			}

		m = len(test_stats)

		# Estimate noise scale from the median
		empirical_median = np.median(test_stats)
		chi2_med = chi2.median(df)
		scale = empirical_median / chi2_med if chi2_med > 0 else 1.0

		# Candidate's scaled statistic
		d_c = self.distances[candidate_idx]
		n_c = self.counts[candidate_idx]
		l2_sq_c = np.dot(self.data[candidate_idx,:], self.data[candidate_idx,:])
		if l2_sq_c >= 1.0 or n_c <= 0 or scale <= 0:
			return {
				'survival_prob': 1.0, 'scaled_stat': 0.0,
				'expected_null_max': 0.0, 'ratio': 0.0,
				'df': df, 'scale': scale,
			}

		t_candidate = n_c * d_c * d_c / (1 - l2_sq_c)
		scaled_stat = t_candidate / scale

		# Expected max of m chi2(df) draws:
		# approx df + 2*sqrt(df * 2*log(m)) + 2*log(m)
		expected_null_max = df + 2*math.sqrt(df * 2*math.log(m)) + 2*math.log(m)

		ratio = scaled_stat / expected_null_max if expected_null_max > 0 else 0.0

		# Survival probability: P(max of m chi2(df) >= scaled_stat)
		# = 1 - P(all m < scaled_stat) = 1 - (1 - sf(scaled_stat))^m
		single_p = chi2.sf(scaled_stat, df=df)
		if single_p >= 1.0:
			survival_prob = 1.0
		elif single_p <= 0.0:
			survival_prob = 0.0
		elif m * single_p < 0.01:
			# For small probabilities, use Bonferroni approximation
			survival_prob = min(1.0, m * single_p)
		else:
			survival_prob = 1.0 - (1.0 - single_p) ** m

		return {
			'survival_prob': survival_prob,
			'scaled_stat': scaled_stat,
			'expected_null_max': expected_null_max,
			'ratio': ratio,
			'df': df,
			'scale': scale,
		}
			

	@staticmethod
	def kl_divergence(p, q):
		"""KL(p || q) = sum_i p_i log(p_i / q_i).

		Returns inf if any p_i > 0 where q_i == 0.
		"""
		d = 0.0
		for pi, qi in zip(p, q):
			if pi > 1e-15:
				if qi <= 1e-15:
					return math.inf
				d += pi * math.log(pi / qi)
		return d

	def min_kl_to_bases(self, candidate_idx):
		"""Minimum KL(candidate || base_j) over all current basis words.

		Uses the raw (pre-shrinkage) distributions so that the divergences
		reflect the true distributional differences rather than being
		compressed by shrinkage toward the global mean.

		Measures how well the candidate's context distribution is
		"covered" by the closest existing kernel.  For a genuine new
		nonterminal anchor the minimum should be large; for a spurious
		candidate that is a mixture of existing kernels it will be small.

		Returns:
			(min_kl, closest_base_index)
		"""
		p = self.raw_data[candidate_idx, :]
		min_kl = math.inf
		closest = None
		for bi in self.bases:
			kl = self.kl_divergence(p, self.raw_data[bi, :])
			if kl < min_kl:
				min_kl = kl
				closest = bi
		return min_kl, closest

	def bootstrap_null_distance(self, candidate_idx, n_bootstrap=500, seed=None):
		"""Bootstrap the null distribution of hyperplane distances.

		Uses the candidate's FW decomposition as the null hypothesis,
		accounting for sampling noise in BOTH the candidate AND the
		existing kernel distributions.

		For each bootstrap replicate:
		  1. Resample each kernel's distribution from Multinomial(n_j, raw_j)
		     to account for estimation noise in the basis vectors.
		     (The start symbol at bases[0] is artificial and not resampled.)
		  2. Re-shrink resampled kernels to form the bootstrap basis.
		  3. Form the null mixture p = FW_weights @ resampled_raw_kernels.
		  4. Sample candidate counts ~ Multinomial(n_cand, p_null).
		  5. Normalize, shrink, and compute Gram-Schmidt distance from
		     the hyperplane spanned by the resampled basis.

		Args:
			candidate_idx: index into self.data of the candidate word
			n_bootstrap: number of bootstrap samples
			seed: random seed for reproducibility

		Returns:
			(p_value, candidate_distance, bootstrap_distances)
			p_value: fraction of bootstrap distances >= candidate's distance
		"""
		rng = np.random.RandomState(seed)
		k = len(self.bases)
		n_cand = int(self.counts[candidate_idx])
		f = self.f

		# Get the candidate's actual hyperplane distance
		d_candidate = self.distances[candidate_idx]

		# Get the candidate's FW decomposition for null mixture weights
		self.initialise_frank_wolfe()
		x_candidate, _ = self.estimate_frank_wolfe(
			self.data[candidate_idx, :])

		# Kernel info: counts and raw distributions
		kernel_counts = np.array([int(self.counts[bi]) for bi in self.bases])
		kernel_raw = np.array([self.raw_data[bi, :] for bi in self.bases])

		bootstrap_distances = np.zeros(n_bootstrap)

		for b in range(n_bootstrap):
			# Step 1: Resample kernel distributions from their counts.
			# bases[0] is the start symbol (artificial, fixed); skip it.
			resampled_raw = kernel_raw.copy()
			resampled_shrunk = np.zeros((k, f))

			for j in range(k):
				if j == 0:
					# Start symbol: artificial, don't resample
					resampled_shrunk[j] = self.data[self.bases[j], :]
				else:
					n_j = kernel_counts[j]
					if n_j > 0:
						counts_j = rng.multinomial(n_j, kernel_raw[j])
						resampled_raw[j] = counts_j / n_j
					# Apply shrinkage
					lam_j = f / (f + n_j)
					resampled_shrunk[j] = (
						(1 - lam_j) * resampled_raw[j] + lam_j * self.p_global)

			# Step 2: Form null mixture from resampled raw kernels
			p_null = x_candidate @ resampled_raw
			p_null = np.maximum(p_null, 0)
			p_sum = np.sum(p_null)
			if p_sum <= 0:
				continue
			p_null /= p_sum

			# Step 3: Sample candidate counts from null mixture
			counts_cand = rng.multinomial(n_cand, p_null)
			total = np.sum(counts_cand)
			if total == 0:
				continue
			dist_cand = counts_cand / total

			# Apply shrinkage
			lam_c = f / (f + total)
			dist_shrunk = (1 - lam_c) * dist_cand + lam_c * self.p_global

			# Step 4: Compute Gram-Schmidt distance to the hyperplane
			# spanned by the resampled (shrunk) basis vectors.
			ortho = []
			for v in resampled_shrunk:
				u = v.copy()
				for o in ortho:
					u = u - np.dot(u, o) * o
				norm = np.linalg.norm(u)
				if norm > 1e-10:
					ortho.append(u / norm)

			residual = dist_shrunk.copy()
			for o in ortho:
				residual = residual - np.dot(residual, o) * o
			bootstrap_distances[b] = np.linalg.norm(residual)

		p_value = float(np.mean(bootstrap_distances >= d_candidate))
		return p_value, d_candidate, bootstrap_distances

	def remove_last_element(self):
		self.bases = self.bases[:-1]
		self.M = self.M[:-1]
		self.gram_schmidt()

	def cluster_vertices(self):
		MMM = []
		clustering = {}
		for kk,k in enumerate(self.bases):
			ind = []
			#print( '-' * 50)
			for i in self.bases:
				d = scipy.spatial.distance.euclidean(self.data[i,:], self.data[k,:])
				if i != k: 
					ind.append(d)
			#print(min(id))
			thresh3 = min(ind)/2
			total = np.zeros(self.f)
			for i in range(len(self.index)):

				d = scipy.spatial.distance.euclidean(self.data[i,:], self.data[k,:])
				if d < thresh3:
					if i in clustering:
						logging.warning("Same letter in multiple clusters ..?", i, self.index[i])
					clustering[self.index[i]] = kk
					total += self.counts[i] * self.data[i,:] 
					a = self.index[i]
					#print(i,a,d, target_pcfg.find_best_lhs(a), te[a],nmf0.counts[i])
			MMM.append(total/np.sum(total))
		return clustering, MMM

# Frank Wolfe stuff
	def estimate_all_frank_wolfe(self):
		self.initialise_frank_wolfe()

		self.xs = {}
		for i,a in enumerate(self.index):
			#print(i,a)
			y = self.data[i,:]
			x,d = self.estimate_frank_wolfe(y)
			#print(x,d)
			self.xs[a] = x

	def initialise_frank_wolfe(self):
		self.nk = len(self.bases)
		self.MM = np.zeros((self.f,self.nk))
		for i,m in enumerate(self.M):
			self.MM[:,i] = m
		self.e = np.eye(self.nk)

	def estimate_frank_wolfe(self, y,verbose=False):
		x = np.ones(self.nk) / self.nk
		oldd2 = math.inf
		for iteration in range(max_iterations):
			y0 = np.dot(self.MM,x)
			v1 = (y - y0)
			d2 = np.linalg.norm(v1)
			if verbose: print(iteration,d2)
			if d2 < epsilon or (abs(d2 - oldd2)) < epsilon:
				break
			oldd2 = d2
			gradient =  np.dot(np.transpose(v1), self.MM)
			i = np.argmax(gradient)
			y1 = self.MM[:,i]
			## We are at the vertex. 
			if (np.linalg.norm(y1 - y0) < epsilon):
				break
			alpha = min_on_line(y, y0, y1)
			x += alpha * (self.e[i,:] - x)
		return x,d2

def min_on_line(y,y0,y1):
	#print y.shape, y0.shape, y1.shape
	alpha = np.dot(y0 - y1, y0 - y)
	v = y1 - y0
	beta = np.dot(v,v)
	if beta == 0.0:
		raise ValueError()
	gamma =  alpha/beta
	if gamma < 0.0:
		gamma = 0.0
	if gamma > 1.0:
		gamma = 1.0    
	return gamma




