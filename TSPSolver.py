# !/usr/bin/python3

from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
import numpy as np
from TSPClasses import *
import heapq
import itertools
import random

# Constants for the annealing schedule
# For now the annealing schedule is approximately T(t) = E(t)/(E(t)-s)
# Where E(t) = 2^(t/r)
R = 50  # The slope of the s-curve
S_1 = 1000
S = 2 ** S_1  # The left-right displacement of the s-curve
# We can modify these variables in order to change how fast/slow the cooling happens.
# The formula for the number of iterations that takes place is x=2*R*S_1
# Thus with these current values, we will have 1000 cycles.
k = .001


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None
        self.matrix = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario
        self.matrix = self.build_matrix()

    def build_matrix(self):
        cities = self._scenario.getCities()
        matrix = [[np.inf for i in range(len(cities))] for j in range(len(cities))]
        for row in range(len(matrix)):
            for col in range(len(matrix[row])):
                matrix[row][col] = cities[row].costTo(cities[col])
        return matrix

    ''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution,
		time spent to find solution, number of permutations tried during search, the
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
		This is the entry point for the greedy solver, which you must implement for
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

    def greedy(self, time_allowance=60.0):
        start_time = time.time()
        cities = self._scenario.getCities()

        count = 0

        # Get a quick solution (the cities in order)
        bssf = TSPSolution(cities)
        for city in cities:
            # Find a solution for the current city
            solution = self.find_greedy_solution(city, cities)

            # Check to see if we're out of time
            if time.time() - start_time >= time_allowance:
                print("Timeout!")
                break
            # If a solution was found for this city, bump up the counter
            if solution is not None:
                count += 1
                # If this solution is the best so far, remember it!
                if solution.cost < bssf.cost:
                    bssf = solution

        # If after all the looping, there still isn't a valid solution, just create
        # a random tour with the time remaining.
        if bssf.cost == np.inf:
            random = self.defaultRandomTour(time.time() - start_time)
            bssf = random['soln']

        end_time = time.time()
        results = {
            'cost': bssf.cost,
            'time': end_time - start_time,
            'count': count,
            'soln': bssf,
            'max': None,
            'total': None,
            'pruned': None
        }
        return results

    # Finds a greedy solution starting from a specific city
    def find_greedy_solution(self, start, cities):
        # List that describes the path for the solution
        path = []
        # List to quickly keep track of which cities have been visited.
        # (Prevents the need to search through the path list, which would add unnecessary complexity)
        visited = [False for x in range(len(cities))]
        current = start
        path.append(current)
        visited[current._index] = True
        # While there are still cities that haven't been visited yet...
        while len(path) != len(cities):
            # Find the closest neighbor to the current city
            next = self.find_closest_neighbor(current, cities, visited)
            # If you can't get to any more cities from the current city, there's
            # no possible path
            if next is None:
                return None
            # Add next city to the path and repeat
            path.append(next)
            current = next
            visited[current._index] = True

        # Check to see if the last city can get back to the beginning city
        if path[-1].costTo(start) != np.inf:
            return TSPSolution(path)
        else:
            return None

    # Finds the closest unvisited city to another city
    def find_closest_neighbor(self, from_city, cities, visited):
        closest_dist = np.inf
        closest = None
        # Cycle through all the cities. Only check the cost to cities that haven't
        # been visited yet.
        for city in cities:
            if not visited[city._index]:
                cost_to = from_city.costTo(city)
                if cost_to < closest_dist:
                    closest = city
                    closest_dist = cost_to

        return closest

    ''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints:
		max queue size, total number of states created, and number of pruned states.</returns>
	'''

    def branchAndBound(self, time_allowance=60.0):
        start_time = time.time()
        # Get a quick, greedy solution as our Best So Far
        greedy_sol = self.greedy(time_allowance)
        bssf = greedy_sol['soln']
        best_cost = bssf.cost

        cities = self._scenario.getCities()

        # Initialize the state queue
        state_queue = []

        # Create the origianl distance matrix
        dist_matrix = [[] for i in range(len(cities))]
        for city in cities:
            dist_matrix[city._index] = [city.costTo(j) for j in cities]

        # Initialize some returned values
        solutions = 0
        states = 1
        pruned = 0
        max_queue_size = 1

        # Create the first Branch and Bound State and reduce the original matrix
        start_city = cities[0]._index
        start_state = BnBState(dist_matrix, [start_city], 0)
        start_state.reduce_matrix()
        # Add the first state to the queue
        self.add_state_to_queue(state_queue, start_state)

        # While the state queue isn't empty...
        while state_queue:
            # Check to see if we're out of time
            if time.time() - start_time >= time_allowance:
                print("Timeout!")
                break
            # Pop the top priority state off the queue and take it out of the tuple
            current_state = heapq.heappop(state_queue)[1]
            # If this state's path contains all the cities...
            if len(current_state._path) == len(cities):
                # Add the cost back to the beginning to the cost of the tour
                last_city = current_state._path[-1]
                cost_back = current_state._matrix[last_city][start_city]
                full_cost = current_state._bound + cost_back
                # If the cost is the best so far...
                if full_cost < best_cost:
                    solutions += 1
                    # Make it the new best solution
                    bssf = TSPSolution(self.ints_to_cities(current_state._path))
                    best_cost = bssf.cost
                    # Prune the priority queue
                    state_queue, removed = self.prune_tree(state_queue, best_cost)
                    # Re-prioritize the queue
                    heapq.heapify(state_queue)
                    pruned += removed
            # If the state hasn't visited all the cities yet...
            else:
                # Create all possible child states
                children_states = self.create_children(current_state)
                for child in children_states:
                    # Increase the state counter
                    states += 1
                    # If the child's cost is less than the current best, add it
                    # to the queue
                    if child._bound <= best_cost:
                        self.add_state_to_queue(state_queue, child)
                    # Otherwise, prune it away
                    else:
                        pruned += 1
            # Check to see if the queue has hit a new maximum size
            queue_size = len(state_queue)
            if queue_size > max_queue_size:
                max_queue_size = queue_size

        # Prune any remaining states in the queue
        pruned += len(state_queue)

        if len(state_queue) > 0:
            cost = str(bssf.cost) + "*"
        else:
            cost = str(bssf.cost)

        end_time = time.time()
        result = {
            'cost': cost,
            'time': end_time - start_time,
            'count': solutions,
            'soln': bssf,
            'max': max_queue_size,
            'total': states,
            'pruned': pruned
        }
        return result

    # Add a state to the priority queue
    def add_state_to_queue(self, queue, state):
        # Determine the priority modifier that's based on path completeness
        # (attempts to prioritize paths that are closer to being finished to prune
        # the tree, queue, more often)
        modifier = self.simple_modifier(queue, state)
        priority = state._bound - modifier
        # Add the state to the queue in a tuple, with the priority being the first
        # item by which the queue is sorted, and the state being the second item.
        heapq.heappush(queue, (priority, state))

    # A simple modifier that is based soley on path completeness
    def simple_modifier(self, queue, state):
        return len(state._path) * 1000

    # A more complicated modifier that tries to balance path completeness with overall
    # problem complexity. Doesn't seem to work as well as the simple modifier.
    def complex_modifier(self, queue, state):
        if len(queue) > 0:
            worst = queue[-1][1]._bound
        else:
            worst = 1
        return len(state._path) * (10 ** (len(str(worst)) - 1))

    # Creates a list of child states based off of a parent
    def create_children(self, parent):
        children = []
        # Get the last city visited by the parent
        from_city = parent._path[-1]
        for i in range(len(parent._matrix)):
            # If a city has not been visited yet...
            if i not in parent._path:
                # Create a new distance matrix and state that is a copy of the parent
                new_matrix = [row.copy() for row in parent._matrix.copy()]
                new_state = BnBState(new_matrix, parent._path.copy(), parent._bound)
                # Add the new city to the path. This method updates and reduces the
                # distance matrix.
                new_state.add_to_path(i)
                children.append(new_state)
        return children

    # Removes all current states that cost more than the best complete solution
    def prune_tree(self, tree, max):
        removed = 0
        # Create a new list that will hold the kept states
        kept = []
        for tup in tree:
            # Get the state out of the tuple
            state = tup[1]
            # If the state costs more than the best solution, toss it.
            if state._bound > max:
                removed += 1
            # Otherwise, keep the tuple
            else:
                kept.append(tup)
        return kept, removed

    ''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found during search, the
		best solution found.  You may use the other three field however you like.
		algorithm</returns>
	'''

    def fancy(self, time_allowance=60.0):
        start_time = time.time()
        # print("\n\nStarting new Calculation\n")
        # Get an initial solution.
        init_solution = self.greedy(time_allowance)
        bssf = init_solution['soln']

        # Initialize values:
        iteration = 0
        # Number of solutions found
        solutions = 0
        # We want to keep track of our bssf and the current tour we're on.
        current_tour = self.indicesFromCities(bssf.route)
        current_cost = bssf.cost
        # Loop until done
        done = False
        while not done:
            # Check if we're out of time
            if time.time() - start_time >= time_allowance:
                print("Timeout!")
                break

            # Get the next candidate tour
            candidate_tour, candidate_cost = self.get_candidate_tour(current_tour)
            # See if we're going to accept it or not
            if self.accept_new_route(current_cost, candidate_cost, iteration):
                current_tour = candidate_tour
                current_cost = candidate_cost
                # Check to see if the new current tour is better than the BSSF
                if current_cost < bssf.cost:
                    solutions += 1
                    bssf = TSPSolution(self.ints_to_cities(current_tour))

            # Check if done (could change based on done definition)
            if iteration == 2 * R * S_1:
                done = True
            # Increment iteration counter
            iteration += 1

        end_time = time.time()
        result = {
            'cost': bssf.cost,
            'time': end_time - start_time,
            'count': solutions,
            'soln': bssf,
            'max': current_cost,
            'total': 2 * R * S_1,
            'pruned': None
        }
        return result

    def indicesFromCities(self, cities):
        return [city._index for city in cities]

    # Converts a list of integers into the corresponding path of cities
    def ints_to_cities(self, ints):
        result = []
        cities = self._scenario.getCities()
        for i in ints:
            result.append(cities[i])
        return result

    def accept_new_route(self, old_cost, new_cost, iteration):
        # print("\nIteration: " + str(iteration))
        # print("Old cost: " + str(old_cost))
        # print("New cost: " + str(new_cost))
        # print("Difference: " + str(new_cost - old_cost))
        if new_cost < old_cost:
            return True
        # temp = self.calc_temp(iteration, new_cost - old_cost)
        temp = self.calc_temp(iteration)
        # print("Temp: " + str(temp))
        odds = self.calc_odds_of_acceptance(temp, old_cost, new_cost)
        # print("Odds: " + str(odds))
        # random.seed(12345)
        random_value = random.random()
        return random_value < odds

    # TODO write in the option for Newton's Law of Cooling as annealing schedule

    def calc_odds_of_acceptance(self, temperature, oldCost,
                                newCost):  # odds of acceptance are a function of how much worse new solution is
        percentageDifference = (newCost - oldCost) / oldCost
        # print("Precent Diff: " + str(percentageDifference))
        if percentageDifference == 0:
            return temperature
        return temperature / (percentageDifference * 100)

    # T(t) = 2^(t/r)/(2^(t/r)-s)
    # exponent = temperature / R
    # numerator = 2**exponent
    # denominator = numerator - S
    # return (numerator / denominator)

    def calc_temp(self, iteration):
        return (2 ** (-1 * k * iteration))

    # def calc_temp(self, iteration, difference):
    # 	new_temp = iteration - difference
    # 	return new_temp if new_temp > 0 else 0

    def two_opt(self, route, i, k):
        # i and k are indices in route where 0 ≤ i < k < #cities
        neighborRoute = []

        # add indices 0 - i-1 in order
        for j in range(i):
            neighborRoute.append(route[j])

        # add indices i - k in reverse order
        for j in range(k, i - 1, -1):
            # make sure all these backwards edges exist
            if len(neighborRoute) != 0 and self.matrix[neighborRoute[-1]][route[j]] == np.inf:
                return None
            neighborRoute.append(route[j])

        # make sure the last backwards edge connects to the next city
        if len(route) > k + 1 and self.matrix[neighborRoute[-1]][route[k + 1]] == np.inf:
            return None

        # add indices k+1 - end in order
        for j in range(k + 1, len(route)):
            neighborRoute.append(route[j])

        # make sure that the route is a cycle i.e. the last city connects to the first
        if self.matrix[neighborRoute[-1]][neighborRoute[0]] == np.inf:
            return None

        return neighborRoute

    def get_candidate_tour(self, current_tour):
        done = False
        randint = random.randint
        while not done:
            i = randint(0, len(current_tour) - 2)
            k = randint(i + 1, len(current_tour) - 1)
            new_route = self.two_opt(current_tour, i, k)
            if new_route is None:
                continue
            else:
                done = True
        new_sol = TSPSolution(self.ints_to_cities(new_route))

        if new_sol.cost == np.inf:
            print("INFINITE ROUTE!!!")
            print("Current tour: " + str(current_tour))
            print("New tour: " + str(new_route))
            exit()
        return new_route, new_sol.cost
