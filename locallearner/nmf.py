# nmf.py
# NOn negative matrix factorisation of a matrix.

# Version 2 is optimized and has a better step size

import numpy as np
import numpy.linalg 
import scipy.spatial.distance
import math
import logging

max_iterations = 100
epsilon = 1e-5

class NMF:

	def __init__(self, data, index, ssf = 1.0):
		## Matrix of (ndata, nfeatures) 
		# all nonnegative, count data,.
		# print(data)
		# print(idx)
		self.data = data.copy()
		self.ssf= ssf
		## Index is a list of n elements.
		self.index = index
		self.n = data.shape[0]
		self.f = data.shape[1]
		self.counts = np.zeros(self.n)
		for i in range(self.n):
			self.counts[i] = np.sum(self.data[i,:])
			self.data[i,:] = self.data[i,:]/ self.counts[i]


		# list of indices
		self.bases = None
		## list of current basis vectors 
		self.M = None
		# Orthonormal basis for the span of the basis vectors. (list of vectors)
		self.orthonormal = None
		## ones that we don't consider.
		self.excluded = set()


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
		"""
		i = np.argmax(np.linalg.norm(self.data,axis=1))
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





	def find_furthest(self,verbose=True):
		largestd = -1
		furthest = None

		for i in range(self.n):
			if not i in self.excluded:
				n = self.counts[i]
				d = self.distances[i]
				d -= self.small_sample_factor(n)
				if verbose:
					print(self.index[i], d,n)
				if d > largestd:
					largestd = d
					furthest = i
		print("Largest d ", largestd)
		return furthest, largestd


	def add_basis(self, i, gram_schmidt=False):
		self.bases.append(i)
		self.excluded.add(i)
		self.M.append(self.data[i,:])
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
		if a == None:
			print(a,d,"NONE")
			return None,None
		else:
			return self.index[a],a
			

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




