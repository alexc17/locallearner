
from collections import Counter
from collections import defaultdict

import numpy as np

import numpy.random as npr
import math

def mi(bg, u1, u2):
	if (bg > 0):
		return bg * math.log(bg / (u1 * u2))
	else:
		return 0.0

   

class Clustering:

	def __init__(self):
		self.clusters = 32
		self.min_count = 10
		self.boundary  = '<eos>'
		self.changes = None
		self.unknowns = []

	def lookup(self,w):
		return self.word2idx.get(w,self.unkidx)


	def cluster(self,data,seed=None,verbose=False):
		self.load(data=data)
		self.random_initialization(seed=seed)
		of = self.recluster_until_done(verbose=verbose)
		print(f"Clustering seed {seed} objective function {of}")
		result = {}
		for i,w in enumerate(self.idx2word):
			c = self.cluster_indices[i]
			if w == self.unk:
				for w in self.unknowns:
					result[w] = c
			elif w != self.boundary:
				result[w] = c

		return result


	def load(self,filename=None, data = None):

		words = Counter()
		if data == None:
			with open(filename) as inf:
				for line in inf:
					s = line.split()
					if len(s) > 0:
						for w in s:
							words[w] +=1
		else:
			for s in data:
				for w in s:
					words[w] +=1
		self.idx2word = [ self.boundary ]
		extra = 0
		for w,n in words.items():
			if n >= self.min_count:
				self.idx2word.append(w)
			else:
				extra += 1
				self.unknowns.append(w)
		

		# + 1 for the unknown words
		if extra > 0:
			self.alphabetsize = len(self.idx2word) + 1
			self.unk = "UNKNEYESSEN"
			self.unkidx = self.alphabetsize - 1
			self.idx2word.append(self.unk)
		else:
			self.unk = False
			self.alphabetsize = len(self.idx2word) 
			self.unkidx = -1

		self.word2idx = { a:i for i,a in enumerate(self.idx2word)}
		self.bigrams = Counter()
		self.unigrams = np.zeros(self.alphabetsize)
		def process_line(s):
			left = 0

			for w in s:
				self.unigrams[left] += 1
				current = self.lookup(w)
				self.bigrams[(left,current)] += 1
				left = current
			self.unigrams[left] += 1
			self.bigrams[(left,0)] += 1
		if data == None:
			with open(filename) as inf:
				for line in inf:
					process_line(line.split())
		else:
			for line in data:
				process_line(line)
					
		self.left_index = defaultdict(list)
		self.right_index = defaultdict(list)
		for (a,b) in self.bigrams:
			self.left_index[a].append(b)
			self.right_index[b].append(a)

	def random_initialization(self, seed=None):
		# seed is None means initialised from /dev/urandom or something
		npr.seed(seed)
		
		self.cluster_indices = npr.randint(1, high=self.clusters, size=self.alphabetsize)
		# distinguished cluster for the sentence boundary which we don't change
		self.cluster_indices[0] = 0

		
		self.cluster_unigrams = self.compute_cluster_unigrams()
		self.cluster_bigrams = self.compute_cluster_bigrams()

	def compute_cluster_bigrams(self):
		cluster_bigrams = np.zeros((self.clusters,self.clusters))
		for (a,b), n in self.bigrams.items():
			cluster_bigrams[self.cluster_indices[a],self.cluster_indices[b]] += self.bigrams[a,b]
		return cluster_bigrams

	def compute_cluster_unigrams(self):
		cluster_unigrams = np.zeros(self.clusters)
		for i in range(self.alphabetsize):
			c = self.cluster_indices[i]
			cluster_unigrams[c] += self.unigrams[i]
		return cluster_unigrams

	def check_cluster_bigrams(self):
		cb = self.compute_cluster_bigrams()
		r = np.array_equal(cb,self.cluster_bigrams)
		print("cluster bigram check",r)
		cu = self.compute_cluster_unigrams()
		s = np.array_equal(cu,self.cluster_unigrams)
		print("cluster unigram check",s)

		return r and s

	def recluster_until_done(self, maxiters=math.inf, verbose=False):
		i = 0
		while self.recluster()[0] > 0:
			if verbose:
				print(f"Iteration {i} Objective function {self.objective_function()}")
			i += 1
			if i >= maxiters:
				break
		return self.objective_function()

	def recluster(self):
		"""
		return the number of ones that changed clusters.
		"""
		self.changes = Counter()
		changed = 0
		delta = 0
		for i in range(1, self.alphabetsize):
			e = self.bestCluster(i)
			delta += e
			if e > 0.0:
				changed += 1
				# print(self.cluster_bigrams)
				# break
		return changed, delta

	def bestCluster(self, i,verbose=False):
		"""
		return true if it moves
		"""
		old_cluster = self.cluster_indices[i]
		if verbose: print("testing",i,old_cluster)
		score = 0
		
		best = old_cluster
		## Compute left and right vectors of occurrences of clusters immediately before and after
		left = np.zeros(self.clusters)
		right = np.zeros(self.clusters)
		doubles = self.bigrams[(i,i)]
		## index of all the bigrams with w in in the corpus.
		for j in self.left_index[i]:
			n = self.bigrams[(i,j)]
			right[self.cluster_indices[j]] += n
		for j in self.right_index[i]:
			n = self.bigrams[(j,i)]
			left[self.cluster_indices[j]] += n
		for new_cluster in range(1,self.clusters):
			new_score = self.calculateChange(i,old_cluster,new_cluster,left,right, doubles)
			if verbose: print(new_cluster,new_score)
			if new_score > score:
				#print("New old", new_score, score)
				score = new_score
				best = new_cluster
		if old_cluster == best:
			return 0.0
		# print(len(self.idx2word),self.alphabetsize)
		# print("%d %s : %d -> %d (%f)" % (i,self.idx2word[i],old_cluster,best,score))
		self.move_word(i, old_cluster, best, left, right, doubles)
		self.changes[(old_cluster,best)] += 1
		return score

	def objective_function(self):
		"""
		Current log likelihood of the model.
		"""
		of = 0
		for ca in range(self.clusters):
			for cb in range(self.clusters):
				of += mi(self.cluster_bigrams[ca,cb],self.cluster_unigrams[ca],self.cluster_unigrams[cb] )
		return of


	def calculateChange(self, i, old_cluster, new_cluster,left,right, doubles, verbose=False):
		"""
		Compute the change from moving i from current cluster to new cluster.

		left is the vector of preceding clusters for word i 
		right is the following,
		doubles are the number of bigrams (i,i)

		"""
		#print("Word", i,"O",old_cluster,"N", new_cluster)
		before = 0.0
		after = 0.0
		new_count = self.cluster_unigrams[new_cluster]
		old_count = self.cluster_unigrams[old_cluster]
		c = self.unigrams[i]
		if c == old_count:
			#print("Singleton cluster: can't move it out.")
			return 0.0
		if old_cluster == new_cluster:
			#print("Old = new.")
			return 0.0

		for g in range(0,self.clusters):
			if g != old_cluster and g != new_cluster:
				#print("Cluster,",g)
				# new_cluster,g
				bg = self.cluster_bigrams[new_cluster,g]
				xu = new_count
				yu = self.cluster_unigrams[g]
				before += mi(bg,xu,yu)
				after += mi(bg + right[g], xu + c, yu)
				# old_cluster,g
				bg = self.cluster_bigrams[old_cluster,g]
				xu = old_count
				yu = self.cluster_unigrams[g]
				before += mi(bg,xu,yu)
				after += mi(bg - right[g], xu - c, yu)
				# g,new_cluster
				bg = self.cluster_bigrams[g,new_cluster]
				xu = self.cluster_unigrams[g]
				yu = new_count
				before += mi(bg,xu,yu)
				after += mi(bg + left[g], xu , yu +  c)
				# g,old_cluster
				bg = self.cluster_bigrams[g,old_cluster]
				xu = self.cluster_unigrams[g]
				yu = old_count
				before += mi(bg,xu,yu)
				after += mi(bg - left[g], xu , yu - c)
			## Now the four intersection points of the stripes
			## old old

		bg = self.cluster_bigrams[old_cluster,old_cluster]
		before += mi(bg,old_count,old_count)
		after += mi(bg + doubles - right[old_cluster] - left[old_cluster], old_count - c, old_count -c)
		#print("OO",bg + doubles - right[old_cluster] - left[old_cluster] )
		# new new 
		bg = self.cluster_bigrams[new_cluster,new_cluster]
		before += mi(bg,new_count,new_count)
		# change bg +  doubles to bg - doubles 
		after += mi(bg + doubles + right[new_cluster] + left[new_cluster], new_count + c, new_count + c)
		#print("NN", bg - doubles + right[new_cluster] + left[new_cluster])
		# old new 
		bg = self.cluster_bigrams[old_cluster,new_cluster]
		before += mi(bg,old_count,new_count)
		after += mi(bg - doubles - right[new_cluster] + left[old_cluster], old_count - c, new_count + c)
		#print("ON", bg - doubles - right[new_cluster] + left[old_cluster])
		# new old
		bg = self.cluster_bigrams[new_cluster,old_cluster]
		before += mi(bg,new_count,old_count)
		after += mi(bg - doubles + right[old_cluster] - left[new_cluster], new_count + c, old_count - c)
		#print("NO",bg - doubles + right[old_cluster] - left[new_cluster])
		#print("Score", after - before)
		return after - before


	def move_word(self, i, old_cluster, new_cluster, left, right, doubles):
		self.cluster_indices[i] = new_cluster
		c = self.unigrams[i]
		self.cluster_unigrams[old_cluster] -= c
		self.cluster_unigrams[new_cluster] += c
		for i in range(self.clusters):
			self.cluster_bigrams[i,old_cluster] -= left[i]
			self.cluster_bigrams[old_cluster,i] -= right[i]
		self.cluster_bigrams[old_cluster,old_cluster] += doubles

		left[old_cluster] -= doubles
		right[old_cluster] -= doubles

		left[new_cluster] += doubles
		right[new_cluster] += doubles
		
		for i in range(self.clusters):
			self.cluster_bigrams[i,new_cluster] += left[i]
			self.cluster_bigrams[new_cluster,i] += right[i]
		self.cluster_bigrams[new_cluster,new_cluster] -= doubles
		
	def save_clustering(self,filename):
		with open(filename,'w') as outf:
			for i,w in enumerate(self.idx2word):
				outf.write("%s %d \n" % (w,self.cluster_indices[i]))




