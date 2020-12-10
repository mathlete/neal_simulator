#!/usr/bin/python3


import math
import numpy as np
import random
import time
from dataclasses import dataclass



class TSPSolution:
	def __init__( self, listOfCities):
		self.route = listOfCities
		self.cost = self._costOfRoute()
		#print( [c._index for c in listOfCities] )

	def _costOfRoute( self ):
		cost = 0
		last = self.route[0]
		for city in self.route[1:]:
			cost += last.costTo(city)
			last = city
		cost += self.route[-1].costTo( self.route[0] )
		return cost

	def enumerateEdges( self ):
		elist = []
		c1 = self.route[0]
		for c2 in self.route[1:]:
			dist = c1.costTo( c2 )
			if dist == np.inf:
				return None
			elist.append( (c1, c2, int(math.ceil(dist))) )
			c1 = c2
		dist = self.route[-1].costTo( self.route[0] )
		if dist == np.inf:
			return None
		elist.append( (self.route[-1], self.route[0], int(math.ceil(dist))) )
		return elist


def nameForInt( num ):
	if num == 0:
		return ''
	elif num <= 26:
		return chr( ord('A')+num-1 )
	else:
		return nameForInt((num-1) // 26 ) + nameForInt((num-1)%26+1)








class Scenario:

	HARD_MODE_FRACTION_TO_REMOVE = 0.20 # Remove 20% of the edges

	def __init__( self, city_locations, difficulty, rand_seed ):
		self._difficulty = difficulty

		if difficulty == "Normal" or difficulty == "Hard":
			self._cities = [City( pt.x(), pt.y(), \
								  random.uniform(0.0,1.0) \
								) for pt in city_locations]
		elif difficulty == "Hard (Deterministic)":
			random.seed( rand_seed )
			self._cities = [City( pt.x(), pt.y(), \
								  random.uniform(0.0,1.0) \
								) for pt in city_locations]
		else:
			self._cities = [City( pt.x(), pt.y() ) for pt in city_locations]


		num = 0
		for city in self._cities:
			#if difficulty == "Hard":
			city.setScenario(self)
			city.setIndexAndName( num, nameForInt( num+1 ) )
			num += 1

		# Assume all edges exists except self-edges
		ncities = len(self._cities)
		self._edge_exists = ( np.ones((ncities,ncities)) - np.diag( np.ones((ncities)) ) ) > 0

		if difficulty == "Hard":
			self.thinEdges()
		elif difficulty == "Hard (Deterministic)":
			self.thinEdges(deterministic=True)

	def getCities( self ):
		return self._cities


	def randperm( self, n ):				#isn't there a numpy function that does this and even gets called in Solver?
		perm = np.arange(n)
		for i in range(n):
			randind = random.randint(i,n-1)
			save = perm[i]
			perm[i] = perm[randind]
			perm[randind] = save
		return perm

	def thinEdges( self, deterministic=False ):
		ncities = len(self._cities)
		edge_count = ncities*(ncities-1) # can't have self-edge
		num_to_remove = np.floor(self.HARD_MODE_FRACTION_TO_REMOVE*edge_count)

		can_delete	= self._edge_exists.copy()

		# Set aside a route to ensure at least one tour exists
		route_keep = np.random.permutation( ncities )
		if deterministic:
			route_keep = self.randperm( ncities )
		for i in range(ncities):
			can_delete[route_keep[i],route_keep[(i+1)%ncities]] = False

		# Now remove edges until
		while num_to_remove > 0:
			if deterministic:
				src = random.randint(0,ncities-1)
				dst = random.randint(0,ncities-1)
			else:
				src = np.random.randint(ncities)
				dst = np.random.randint(ncities)
			if self._edge_exists[src,dst] and can_delete[src,dst]:
				self._edge_exists[src,dst] = False
				num_to_remove -= 1




class City:
	def __init__( self, x, y, elevation=0.0 ):
		self._x = x
		self._y = y
		self._elevation = elevation
		self._scenario	= None
		self._index = -1
		self._name	= None

	def setIndexAndName( self, index, name ):
		self._index = index
		self._name = name

	def setScenario( self, scenario ):
		self._scenario = scenario

	''' <summary>
		How much does it cost to get from this city to the destination?
		Note that this is an asymmetric cost function.

		In advanced mode, it returns infinity when there is no connection.
		</summary> '''
	MAP_SCALE = 1000.0
	def costTo( self, other_city ):

		assert( type(other_city) == City )

		# In hard mode, remove edges; this slows down the calculation...
		# Use this in all difficulties, it ensures INF for self-edge
		if not self._scenario._edge_exists[self._index, other_city._index]:
			return np.inf

		# Euclidean Distance
		cost = math.sqrt( (other_city._x - self._x)**2 +
						  (other_city._y - self._y)**2 )

		# For Medium and Hard modes, add in an asymmetric cost (in easy mode it is zero).
		if not self._scenario._difficulty == 'Easy':
			cost += (other_city._elevation - self._elevation)
			if cost < 0.0:
				cost = 0.0					# Shouldn't it cost something to go downhill, no matter how steep??????


		return int(math.ceil(cost * self.MAP_SCALE))


# A class representing a Branch & Bound algorithm state
# Can be compared and ordered in a priority queue
@dataclass(order=True)
class BnBState:
	# The current bound representing the cost of the path so far and the state's
	# priority compared to other states
	_bound: int
	# A list of numbers correlating to the path of cities in the tour so far
	_path: list
	# A list of lists holding the current distance matrix
	_matrix: list

	def __init__( self, matrix, path_so_far, previous_bound ):
		self._matrix = matrix
		self._bound = previous_bound
		self._path = path_so_far

	# Reduces the states distance matrix and updates the bound value
	def reduce_matrix( self ):
		# Cycle through each row...
		for row in range(len(self._matrix)):
			# Find the minimum value
			min = np.inf
			for col in self._matrix[row]:
				if col < min:
					min = col
			if min != np.inf:
				# Subtract the minimum from all the values in the row
				for col in range(len(self._matrix)):
					self._matrix[row][col] -= min
				# Add the minimum to the bound
				self._bound += min
		# Cycle through each column...
		for col in range(len(self._matrix)):
			# Find the minimum value
			min = np.inf
			for row in range(len(self._matrix)):
				if self._matrix[row][col] < min:
					min = self._matrix[row][col]
			if min != np.inf:
				# Subtract the minimum from all the values in the column
				for row in range(len(self._matrix)):
					self._matrix[row][col] -= min
				# Add the minimum to the bound
				self._bound += min

	# Adds a new city to the path, updates the distance matrix, and reduces it.
	def add_to_path( self, to_city ):
		# Get the last city visited
		last_city = self._path[-1]
		# Add the new city to the path
		self._path.append(to_city)
		# Add the cost to the new city to the bound
		add_cost = self._matrix[last_city][to_city]
		self._bound += add_cost

		# Set the "from row" to infinity
		for col in range(len(self._matrix)):
			self._matrix[last_city][col] = np.inf
		# Set the "to column" to infinity
		for row in range(len(self._matrix)):
			self._matrix[row][to_city] = np.inf
		# Set the reverse edge to infinity
		self._matrix[to_city][last_city] = np.inf
		# Reduce the new matrix
		self.reduce_matrix()

	# Returns a string representation of the State
	def __str__(self):
		s = "Path: " + str(self._path)
		s += "\nBound: " + str(self._bound) + "\n"
		for row in self._matrix:
			s += "["
			for col in row:
				s += str(col) + ", "
			s += "]\n"
		return s
