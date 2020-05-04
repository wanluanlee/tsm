#!/usr/bin/python3

from PyQt5.QtCore import QLineF, QPointF

import time
import numpy as np
from TSPClasses import *
import random
import heapq
import itertools
import timeit
import heapq
import copy


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario
        #self.defaultRand();
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
    '''
    '''
    def findCityList(self):
        totalCity = self._scenario.getCities()
        BBS = np.inf
        count = 0
        bestSolution = []
        '''
        Start from the first city as the starting city and move to the next one.
        '''
        for i in range (len(totalCity)):
            cities = totalCity.copy()
            startCity = totalCity[i]
            cities.remove(startCity)
            citiesList = []
            citiesList.append(startCity)
            totalCost = 0
            #remove cities from if the visited
            while (len(cities) != 0):
                minCost = startCity.costTo(cities[0])
                visitNext = cities[0]
                for i in range(len(cities)):
                    cost = startCity.costTo(cities[i])
                    if (cost < minCost):
                        minCost = cost
                        visitNext = cities[i]
                citiesList.append(visitNext)
                cities.remove(visitNext)
                totalCost = totalCost + minCost
                startCity = visitNext
                '''
                 if the cost is infinity means no solution, 
                 so move to the next starting city and check
                '''
                if(totalCost == np.inf):
                    break


            '''
            check if the last city on the solution can go to 
            the first city, if not move to the next starting city
            '''
            firstCity = citiesList[0]
            lastCity = citiesList[-1]
            if(lastCity.costTo(firstCity) == np.inf):
                totalCost = np.inf
                continue

            '''
            update the best solution path if find a better one
            '''
            if(totalCost < BBS):
                BBS = totalCost
                bestSolution = citiesList
                count = count + 1

        '''
        return the best path found
        '''
        return [bestSolution, count]

    def greedy(self, time_allowance=60.0):
        results = {}
        count = 1
        startGreedy = time.time()
        elapsed = 0
        '''
        check the time is not excess 60 seconds
        '''
        while elapsed < time_allowance:
            bestSolution, count = self.findCityList()
            break
            elapsed = time.time() - startGreedy

        BBS = TSPSolution(bestSolution)
        results['cost'] = BBS.cost
        results['time'] = time.time() - startGreedy
        results['count'] = count
        results['soln'] = BBS
        results['max'] = None
        results['total'] = None
        results['pruned'] = None

        return results

        pass

    ''' <summary>
        This is the entry point for the branch-and-bound algorithm that you will implement
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number solutions found during search (does
        not include the initial BSSF), the best solution found, and three more ints: 
        max queue size, total number of states created, and number of pruned states.</returns> 
    '''

    def branchAndBound(self, time_allowance=60.0):
        startbranchAndBound = time.time()
        results = {}
        elapsed = 0
        count = 0
        max = 0
        pruned = 0
        total = 1
        totalCity = self._scenario.getCities()
        matr = [[0 for i in range(len(totalCity))]
                for j in range(len(totalCity))]
        greedySolution = self.greedy().get('soln')
        BSS = greedySolution.cost
        bestSolution = greedySolution.route
        '''
        create the matrix which contain all the costs between 
        all cities using build in CostTo function
        '''
        for i in range (len(totalCity)):
            for j in range (len(totalCity)):
                   matr[i][j] = totalCity[i].costTo(totalCity[j])

        '''
        reduce matrix and start from the first city
        push node object which contains all cities it has been visited, 
        reduce matrix, cost and priority and push them into heap.
        Heap will sored them using priority
        '''
        heapList = []
        startCity = self._scenario.getCities()[0]
        path = []
        path.append(startCity)
        newMatrix,cost = self.reduceMatrix(matr)
        priority = cost/len(path)
        node = TSPSolver.Node(newMatrix,path,cost,priority)
        i = 0
        total = total + 1
        heapq.heappush(heapList, node)
        if(max < len(heapList)):
            max = len(heapList)
        while(len(heapList) != 0 and elapsed < time_allowance):
            '''
            get the node which has the less priority in the heap
            '''
            startNode = heapq.heappop(heapList)
            path = startNode.path
            startCity = path[-1]
            matrix = startNode.matrix
            cost = startNode.lowerBound
            if(len(path) == len(totalCity)):
                firstCity = path[0]
                lastCity = path[-1]

                if((lastCity.costTo(firstCity)) != np.inf
                        and (BSS > cost)):
                    BSS = cost
                    count = count + 1
                    bestSolution = path
            '''
            expand the node if there is a path from the current last 
            city in the node path, to the next one by checking if the 
            cell index [start city][end city] is not infinity. It also 
            makes sure that the city is not been visited yet.
            '''
            for i in range(len(totalCity)):
                subPath = path.copy()
                subMatrx = copy.deepcopy(matrix)
                if matrix[startCity._index][i] != np.inf:
                    newMatrx,newCost = self.modifyMatrix\
                        (subMatrx,startCity._index,totalCity[i]._index)
                    totalCost = cost + newCost
                    '''
                    if the total cost of the path is greater than the 
                    best solution so far, discar it and increment number 
                    of prunted. Otherwise, create a new node and push to the heap
                    '''
                    if(totalCost < BSS):
                        subPath.append(totalCity[i])
                        priority = cost / len(path)
                        node = TSPSolver.Node(newMatrx,subPath,totalCost,priority)
                        total = total + 1
                        heapq.heappush(heapList, node)
                        if (max < len(heapList)):
                            max = len(heapList)
                    else:
                        pruned = pruned + 1
            elapsed = time.time() - startbranchAndBound

        solution = TSPSolution(bestSolution)
        results['cost'] = solution.cost
        results['time'] = time.time() - startbranchAndBound
        results['count'] = count
        results['soln'] = solution
        results['max'] = max
        results['total'] = total
        results['pruned'] = pruned

        return results
        pass

    ''' <summary>
        This is the entry point for the algorithm you'll write for your group project.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number of solutions found during search, the 
        best solution found.  You may use the other three field however you like.
        algorithm</returns> 
    '''

    '''
    redeceMatric function find the smallest number on each column and row and subtarct them from each cells
    return the total of all the samllest number found
    '''

    def reduceMatrix(self,matrix):
        cost = 0
        for i in range (len(matrix)):
            numMin = min(matrix[i])
            if(numMin == np.inf):
                numMin = 0
            cost = cost + numMin
            for j in range (len(matrix[i])):
                matrix[i][j] = matrix[i][j] - numMin

        for i in range (len(matrix)):
            column = []
            for j in range (len(matrix)):
                column.append(matrix[j][i])
            numMin = min(column)
            if (numMin == np.inf):
                numMin = 0
            cost = cost + numMin
            for j in range(len(matrix)):
                matrix[j][i] = matrix[j][i] - numMin
        return [matrix,cost]

    '''
     change the path from city index i to city index j to infinity.
     i is the starting city index and j is the ending city index. Also change the path from j to i to infinity 
     '''
    def modifyMatrix(self,matrix,i,j):
        toGoCost = matrix[i][j]
        for x in range(len(matrix)):
            matrix[i][x] = np.inf
        for z in range(len(matrix)):
            matrix[z][j] = np.inf
        matrix[j][i] = np.inf
        newMatrix, cost = self.reduceMatrix(matrix)
        totalCost = toGoCost + cost
        return [newMatrix,totalCost]

    def fancy(self, time_allowance=60.0):
        pass

    '''
     create a node object to hold information such as path: cities which has been visited
     matrix: the result of reduced matrix
     lower bound: the total lower bound so far
     pritoriy: the number which heap using to sort. This will be lowerbound / len(path)  
     '''
    class Node:
        def __init__(self,matrix,path,lowerBound,priority):
            self.matrix = matrix
            self.path = path
            self.lowerBound = lowerBound
            self.priority = priority

            '''
            overwrite the comparion function that heap can use 
            '''

        def __lt__(self, other):
            return self.priority < other.priority





