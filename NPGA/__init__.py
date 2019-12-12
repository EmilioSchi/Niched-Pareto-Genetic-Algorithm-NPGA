__version__ = '0.2.6'

# -*- coding: utf-8 -*-
#
# File: NPGA.py
# Niched Pareto Genetic Algorithm (NPGA)
#
# From Master’s Degree Thesis:
# 'Genetic Algorithm to automate Autoencoder architecture configuration'
#
# Author: Emilio Schinina' <emilioschi@gmail.com>
# Copyright (C) 2019 Emilio Schinina'
#
# Licensed under the Apache Lic ense, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random, math
import queue, threading
import numpy as np

# Utility funcion
def FlipCoin(probability):
	return random.random() < probability

def EuclideanDistance(fitnessA, fitnessB):
	d = 0
	for valueA, valueB in zip(fitnessA, fitnessB):
		d = d + math.pow((valueA - valueB), 2)
	return math.sqrt(d)

# Faster than EuclideanDistance, but less readable.
def EuclideanDistanceFast(fitnessA, fitnessB):
	return np.linalg.norm(fitnessA - fitnessB)

def IsNonDominatedable(costs):
	# FIND THE PARETO-EFFICIENT POINTS
	# Let Fi(X), i=1...n, are objective functions for minimization.
	# Efficiencies boolean array, indicating whether each point is
	# Pareto efficient.
	# costs =
	# [[ F1(X1), F2(X1), ... Fn(X1) ],
	#  [ F1(X2), F2(X2), ... Fn(X2) ],
	#  [ ...                        ],
	#  [ F1(Xm), F2(Xm), ... Fn(Xm)]]
	Difference = np.zeros((costs.shape[0] - 1, costs.shape[1]), dtype = np.float64)
	NonDominatedable = np.ones(costs.shape[0], dtype = bool)
	for k in range(costs.shape[0]):
		j = 0
		for i in range(costs.shape[0]):
			if i != k:
				Difference[j, :] = costs[k,:] - costs[i, :]
				j = j + 1
		NonDominatedable [k] = np.all(np.any(Difference < 0, axis=1))
	return NonDominatedable

# Faster than IsNonDominatedable, but less readable.
def IsNonDominatedableFast(costs, return_mask = True):
	# Find the pareto-efficient points
	# :param costs: An (n_points, n_costs) array
	# :param return_mask: True to return a mask
	# :return: An array of indices of pareto-efficient points.
	#     If return_mask is True, this will be an (n_points, ) boolean array
	#     Otherwise it will be a (n_efficient_points, ) integer array of indices.
	NonDominatedable = np.arange(costs.shape[0])
	NumPoints = costs.shape[0]
	NextPointIndex = 0  # Next index in the NonDominatedable array to search for
	while NextPointIndex < len(costs):
		NondominatedPointMask = np.any(costs < costs[NextPointIndex], axis = 1)
		NondominatedPointMask[NextPointIndex] = True
		NonDominatedable = NonDominatedable[NondominatedPointMask]  # Remove dominated points
		costs = costs[NondominatedPointMask]
		NextPointIndex = np.sum(NondominatedPointMask[:NextPointIndex]) + 1
	if return_mask:
		NonDominatedable_mask = np.zeros(NumPoints, dtype = bool)
		NonDominatedable_mask[NonDominatedable] = True
		return NonDominatedable_mask
	else:
		return NonDominatedable

class Statistics:
	def __init__(self, optimal_fitness, fastmode):
		self.NUMBER_OBJECTIVE	 = len(optimal_fitness)
		self.FASTMODE			 = fastmode
		self.COMPARE_FITNESS	 = optimal_fitness
		self.number_combination	 = 0
		self.population 		 = []
		self.population_size	 = 0
		# BUG OF PYTHON3 (SAME ADDRESS POINTER ASSIGNMENT)
		#self.BEST = [{'Genes': [], 'Value': 999999, 'Fitness' : []}] * self.NUMBER_OBJECTIVE
		self.best = []
		for _ in range(self.NUMBER_OBJECTIVE):
			self.best.append({'Genes': [], 'Value': 999999, 'Fitness' : []})

		self.EuclideanBetter = {'Genes': [], 'Distance': 999999, 'Fitness' : []}
		self.sum_fitness		 = np.zeros((self.NUMBER_OBJECTIVE,), dtype = np.float64)
		self.avg				 = np.zeros((self.NUMBER_OBJECTIVE,), dtype = np.float64)
		self.HistoryGenes		 = []
		self.HistoryFitness		 = []
		self.nonDominatedables	 = []

	def Update(self, population, history):
		self.number_combination = len(history['Genes'])
		self.sum_fitness = np.zeros((self.NUMBER_OBJECTIVE,), dtype = np.float64)
		self.population = population
		self.population_size = 0
		for individual in population:

			if self.FASTMODE:
				distance = EuclideanDistanceFast(self.COMPARE_FITNESS, individual.Fitness)
			else:
				distance = EuclideanDistance(self.COMPARE_FITNESS, individual.Fitness)

			if self.EuclideanBetter['Distance'] > distance:
				self.EuclideanBetter['Distance'] = distance
				self.EuclideanBetter['Genes'] = ''.join(individual.Genes)
				self.EuclideanBetter['Fitness'] = individual.Fitness

			for i, fitness in enumerate(individual.FitnessToMinimise):
				self.sum_fitness[i] = self.sum_fitness[i] + fitness

				if self.best[i]['Value'] > fitness:
					self.best[i]['Value'] = fitness
					self.best[i]['Genes'] = ''.join(individual.Genes)
					self.best[i]['Fitness'] = individual.Fitness

			self.population_size = self.population_size + 1

		for i, sum in enumerate(self.sum_fitness):
			self.avg[i] = sum / self.population_size

		self.HistoryGenes = history['Genes']
		self.HistoryFitness = np.asarray(history['FitnessToMinimise'], dtype = np.float64)
		# I want to know if first add fitness is nonDominatedable
		if self.FASTMODE:
			self.nonDominatedable = IsNonDominatedableFast(self.HistoryFitness)
		else:
			self.nonDominatedable = IsNonDominatedable(self.HistoryFitness)
		return self.EuclideanBetter['Genes'], self.EuclideanBetter['Fitness']

class Chromosome:
	def __init__(self, genes, dimention, fitness, fitnessToMinimise):
		self.Genes			 = genes
		self.Length			 = dimention
		self.Fitness		 = fitness
		# maximization problem is the negation of minimazion problem
		self.FitnessToMinimise  = fitnessToMinimise

class NichedParetoGeneticAlgorithm:
	def __init__(self, fnGetFitness, fnDisplay, optimal_fitness, chromosome_set,
	chromosome_length_set, population_size = 30, max_generation = 100,
	crossover_rate = 0.7, mutation_rate = 0.05, length_mutation_rate = 0,
	growth_rate = 0.5, shrink_rate = 0.5, prc_tournament_size = 0.1,
	candidate_size = 2, niche_radius = 1, fastmode = False, multithreadmode = False,
	fnMutation = None, fnCrossover = None, historyrecoverfitness = False):

		assert(crossover_rate >= 0 and crossover_rate <= 1), "Crossover Rate can take values between 0 and 1."
		assert(mutation_rate >= 0 and mutation_rate <= 1), "Mutation Rate can take values between 0 and 1."
		assert(length_mutation_rate >= 0 and length_mutation_rate <= 1), "Length Mutation Rate can take values between 0 and 1."
		assert(prc_tournament_size >= 0 and prc_tournament_size <= 1), "The percentage of tournament size can take values between 0 and 1."
		assert(population_size >= 4), "Population size is very small."
		assert(candidate_size >= 2), "Candidate can be at least 2."
		assert(max_generation >= 1), "Generation can be positive."

		# Functions
		self.OBJECTIVE_FUNCTION	 = fnGetFitness
		self.DISPLAY_FUNCTION	 = fnDisplay

		# Custom operators
		# Method to do the mutation and crossover
		if fnMutation is None:
			self.MUTATION_FUNCTION = self.__Mutation
		else:
			self.MUTATION_FUNCTION = fnMutation

		if fnCrossover is None:
			self.CROSSOVER_FUNCTION = self.__Crossover
		else:
			self.CROSSOVER_FUNCTION = fnCrossover

		# Parameter of classic Genetic Algorithm
		self.CHROMOSOME_SET		 = chromosome_set
		self.LENGTH_SET			 = chromosome_length_set
		p = population_size # even number of population size
		self.POPULATION_SIZE	 = p if p % 2 == 0 else p + 1
		self.MAX_GENERATIONS	 = max_generation
		self.CROSSOVER_RATE		 = crossover_rate
		self.MUTATION_RATE		 = mutation_rate

		# MultiObjective variables
		self.OPTIMAL_FITNESS	 = np.asarray(optimal_fitness, dtype = np.float64)
		self.NUMBER_OBJECTIVE	 = len(optimal_fitness)

		# Variable length chromosome variable
		if len(self.LENGTH_SET) == 1:
			self.LENGTH_MUTATION_RATE	 = 0
		else:
			self.LENGTH_MUTATION_RATE	 = length_mutation_rate
		self.MAX_LENGTH_SET			 = max(chromosome_length_set)
		self.MIN_LENGTH_SET			 = min(chromosome_length_set)
		self.GROWTH_RATE			 = growth_rate
		self.SHRINK_RATE			 = shrink_rate

		# Pareto Niched Selection Tournament parameters
		self.CANDIDATE_SIZE			 = candidate_size
		self.T_DOM					 = math.floor(prc_tournament_size * self.POPULATION_SIZE)
		self.NICHE_RADIUS			 = niche_radius

		self.FASTMODE				 = fastmode
		self.MULTITHREADMODE		 = multithreadmode
		self.HISTORYRECOVERFITNESS	 = historyrecoverfitness

		# Statistic parameters
		self.Statistics = Statistics(self.OPTIMAL_FITNESS, self.FASTMODE)

		self.population = []
		self.history = {'Genes' : [], 'Fitness' : [], 'FitnessToMinimise' : [], 'ProblemType' : []}

	def __AlreadySeen(self, genes):
		if self.HISTORYRECOVERFITNESS and (genes in self.history['Genes']):
			itemindex = self.history['Genes'].index(genes)
			return itemindex, True
		else:
			return 0, False

	def __ThreadObjectiveFunction(self, genes, queue):
		itemindex, historyfound = self.__AlreadySeen(genes)
		if historyfound:
			queue.put((genes, self.history['Fitness'][itemindex], self.history['ProblemType'][itemindex], historyfound))
		else:
			# call objective_function
			fitness = np.zeros((self.NUMBER_OBJECTIVE,), dtype = np.float64)
			problemtypes = []
			for i, (singlefitness, problemtype) in enumerate(self.OBJECTIVE_FUNCTION(genes)):
				fitness[i] = singlefitness
				problemtypes.append(problemtype)
			queue.put((genes, fitness, problemtypes, historyfound))

	def __CheckSolution(self, values, problemtypes):
		result = True
		for i, (value, problemtype) in enumerate(zip(values, problemtypes)):
			if problemtype == 'minimize':
				result = result and (value <= self.OPTIMAL_FITNESS[i])
			elif problemtype == 'maximize':
				result = result and (value >= self.OPTIMAL_FITNESS[i])
		return result

	def __ConvertMaximizeToMinimize(self, values, problemtypes):
		fitnessToMinimise = np.zeros((self.NUMBER_OBJECTIVE,), dtype = np.float64)
		# I want to convert all functions which are to
		# be minimized into a form which allows their
		# maximization.
		for i, (fitness, problemtype) in enumerate(zip(values, problemtypes)):
			if problemtype == 'minimize':
				fitnessToMinimise[i] = fitness
			# Maximization problem is the negation of Minimazion problem
			elif problemtype == 'maximize':
				fitnessToMinimise[i] = -fitness
			else:
				assert(False), "Problem type can be minimize or maximize."
		return fitnessToMinimise

	def __MultiThreadEvaluate(self):
		threads = []
		tmp = []
		queued_request = queue.Queue()

		for chromosome in self.population:
			# call objective_function
			process = threading.Thread(
						target = self.__ThreadObjectiveFunction,
						args=(chromosome.Genes, queued_request,)
						)
			process.start()
			threads.append(process)

		for process in threads:
			process.join()

		for _ in range(queued_request.qsize()):
			genes, fitness, problemtypes, historyfound = queued_request.get()
			fitnessToMinimise = self.__ConvertMaximizeToMinimize(fitness, problemtypes)
			tmp.append(Chromosome(genes, len(genes), fitness, fitnessToMinimise))
			# Store chromosome in already seen list
			if self.HISTORYRECOVERFITNESS and not historyfound:
				self.history['Genes'].append(genes)
				self.history['Fitness'].append(fitness)
				self.history['FitnessToMinimise'].append(fitnessToMinimise)
				self.history['ProblemType'].append(problemtypes)

			solutionfound = self.__CheckSolution(fitness, problemtypes)

		self.population = tmp

		BetterGenes, Betterfitness = self.Statistics.Update(self.population, self.history)
		self.DISPLAY_FUNCTION(self.Statistics)

		return BetterGenes, Betterfitness, solutionfound

	def __Evaluate(self):
		if self.MULTITHREADMODE:
			return self.__MultiThreadEvaluate()

		for chromosome in self.population:
			itemindex, historyfound = self.__AlreadySeen(chromosome.Genes)
			if historyfound:
				chromosome.Fitness = self.history['Fitness'][itemindex]
				chromosome.FitnessToMinimise = self.history['FitnessToMinimise'][itemindex]
			else:
				# call objective_function
				chromosome.Fitness = np.zeros((self.NUMBER_OBJECTIVE,), dtype = np.float64)
				chromosome.FitnessToMinimise = []
				problemtypes = []
				for i, (singlefitness, problemtype) in enumerate(self.OBJECTIVE_FUNCTION(chromosome.Genes)):
					chromosome.Fitness[i] = singlefitness
					problemtypes.append(problemtype)
					fitnessToMinimise = self.__ConvertMaximizeToMinimize(chromosome.Fitness, problemtypes)

				# Store chromosome in already seen list
				self.history['Genes'].append(chromosome.Genes)
				self.history['Fitness'].append(chromosome.Fitness)
				self.history['FitnessToMinimise'].append(chromosome.FitnessToMinimise)
				self.history['ProblemType'].append(problemtypes)

				solutionfound = self.__CheckSolution(chromosome.Fitness, problemtypes)

		EDGenes, EDfitness = self.Statistics.Update(self.population, self.history)
		self.DISPLAY_FUNCTION(self.Statistics)
		return EDGenes, EDfitness, solutionfound

	def __ParetoDominationTournments(self):
		# Few candidate chromosomes and a comparison set, of size T_DOM, of
		# chromosomes are chosen for selection at random from the population.
		compareindexset = random.sample(range(self.POPULATION_SIZE), k = self.CANDIDATE_SIZE + self.T_DOM)

		# Each of candidates are then compared against each individual
		# in the comparison set.
		nonDominatedable = np.ones((self.CANDIDATE_SIZE,), dtype=bool)
		for e, i in enumerate(compareindexset[:self.CANDIDATE_SIZE]):
			costs = np.zeros((1 + self.T_DOM, self.NUMBER_OBJECTIVE), dtype = np.float64)
			costs[0] = self.population[i].FitnessToMinimise

			for z, j in enumerate(compareindexset[self.CANDIDATE_SIZE:]):
				costs[z + 1] = self.population[j].FitnessToMinimise

			# I want to know if first add fitness is nonDominatedable
			if self.FASTMODE:
				nonDominatedable[e] = IsNonDominatedableFast(costs)[0]
			else:
				nonDominatedable[e] = IsNonDominatedable(costs)[0]

		# If one candidate is dominate by the comparison set, and the other
		# is not, the latter is selected for reproduction. If neither or both
		# are dominated by the comparison set, then we must use sharing to
		# choose a winner.
		if np.count_nonzero(nonDominatedable == True) == 1:
			itemindex = np.where(nonDominatedable == True)[0][0]
			candidateindex = compareindexset[itemindex]
			return self.population[candidateindex], False, []
		else:
			return None, True, compareindexset[:self.CANDIDATE_SIZE]

	def __FitnessSharing(self, candidateindexes):
		distances = np.zeros((self.CANDIDATE_SIZE,), dtype = np.float64)
		for e, i in enumerate(candidateindexes):
			distances[e] = self.__NichedCount(self.population[i])
		# If we want to maintain useful diversity, it would be best to
		# choose the candidate that has the smaller niche count.
		itemindex = np.where(distances == distances.min())[0][0]
		candidateindex = candidateindexes[itemindex]
		return self.population[candidateindex]

	def __NichedCount(self, cadidate):
		nichecount = 0;
		for individual in self.population:
			if self.FASTMODE:
				d = EuclideanDistanceFast(cadidate.Fitness, individual.Fitness)
			else:
				d = EuclideanDistance(cadidate.Fitness, individual.Fitness)

			if d <= self.NICHE_RADIUS:
				Sh = 1.0 - (d / self.NICHE_RADIUS)
			else:
				Sh = 0
			nichecount = nichecount + Sh
		return nichecount

	def __Selection(self):
		for _ in range(int(self.POPULATION_SIZE / 2)):
			parentA, dominated, candidateindexes = self.__ParetoDominationTournments()
			if dominated:
				parentA = self.__FitnessSharing(candidateindexes)

			parentB, dominated, candidateindexes = self.__ParetoDominationTournments()
			if dominated:
				parentB = self.__FitnessSharing(candidateindexes)

			yield (parentA, parentB)

	def __Generation(self):
		# It generates random string
		genlen = random.choices(self.LENGTH_SET, k = 1)[0]
		child = random.choices(self.CHROMOSOME_SET, k = genlen)
		return Chromosome(child, genlen, -1, -1)

	def __ShrinkMutation(self, parent):
		# Erase genes.
		child = []

		lengthset = [i for i in self.LENGTH_SET if i < parent.Length]
		mutationLenght = random.sample(lengthset, 1)[0]
		mutationquantity = parent.Length - mutationLenght

		# It erases genes at the end
		child.extend(parent.Genes[:-mutationquantity])

		return Chromosome(child, parent.Length - mutationquantity, -1, -1)

	def __GrowthMutation(self, parent):
		child = []

		lengthset = [i for i in self.LENGTH_SET if i > parent.Length]
		mutationLenght = random.sample(lengthset, 1)[0]
		mutationquantity = mutationLenght - parent.Length

		# Insert random correct genes at the end of chromosome.
		child.extend(parent.Genes)
		# It grows at the end
		child.extend(random.choices(self.CHROMOSOME_SET, k = mutationquantity))

		return Chromosome(child, parent.Length + mutationquantity, -1, -1)

	def __LengthMutation(self, parent):
		if FlipCoin(self.LENGTH_MUTATION_RATE):
			switcher = {
				0 : self.__GrowthMutation,
				1 : self.__ShrinkMutation
				}

			lengthprobability = {
				'growthprobability' : self.GROWTH_RATE,
				'shrinkprobability' : self.SHRINK_RATE
				}

			if self.MAX_LENGTH_SET == parent.Length:
				return self.__ShrinkMutation(parent)

			if self.MIN_LENGTH_SET == parent.Length:
				return self.__GrowthMutation(parent)

			totalprobability = sum(lengthprobability.values())
			realprobability = [p / totalprobability for p in lengthprobability.values()]

			# Generate probability intervals for each mutation operator
			probs = [sum(realprobability[:i+1]) for i in range(len(realprobability))]

			# Roulette wheel
			r = random.random()
			for i, p in enumerate(probs):
				if r <= p:
					mutationtype = i
					break

			operator = switcher.get(i,lambda :'Invalid')
			return operator(parent)
		else:
			return parent

	def __Mutation(self, parent):
		# it consists of the variation of a randomly chosen bit, belonging
		# to a randomly selected string.
		child = []
		# Mutation probability says how often will be parts of chromosome mutated.
		# If there is no mutation, offspring is taken without any change.
		for i in range(parent.Length):
			if FlipCoin(self.MUTATION_RATE):
				newGene, newGeneAlternate = random.sample(self.CHROMOSOME_SET, 2)
				child.extend(newGeneAlternate if newGene == parent.Genes[i] else newGene)
			else:
				child.extend(parent.Genes[i])
		return Chromosome(child, parent.Length, -1, -1)

	def __Crossover(self, parentA, parentB):
		# Two points on both parents' chromosomes is picked randomly, and
		# designated 'crossover points'. This results in
		# two offspring, each carrying some genetic information from both parents.

		# Crossover probability says how often will be crossover performed.
		# If there is no crossover, offspring is exact copy of parents.
		if FlipCoin(self.CROSSOVER_RATE):
			childA, childB = [], []

			# The lengths of both the parent chromosomes are checked and the
			# chromosome whose length is smaller is taken as parent A.
			lenSmall = min(parentA.Length, parentB.Length)
			if parentA.Length != lenSmall:
				# Swap the two strings
				parentA, parentB = parentB, parentA

			# Crossover points are randomly chosen for parent A
			startpoint, endpoint = random.sample(range(1, lenSmall + 1), 2)
			startpoint, endpoint = min(startpoint, endpoint), max(startpoint, endpoint)

			childA.extend(parentB.Genes[0:startpoint])
			childA.extend(parentA.Genes[startpoint:endpoint])
			childA.extend(parentB.Genes[endpoint:])

			childB.extend(parentA.Genes[0:startpoint])
			childB.extend(parentB.Genes[startpoint:endpoint])
			childB.extend(parentA.Genes[endpoint:])

			return	Chromosome(childA, parentB.Length, -1, -1), \
					Chromosome(childB, parentA.Length, -1, -1)
		else:
			return parentA, parentB


	def Evolution(self):
		for _ in range(self.POPULATION_SIZE):
			child = self.__Generation()
			self.population.append(child)

		for _ in range(self.MAX_GENERATIONS):
			# Calculation of fitness
			chromosome, fitness, found = self.__Evaluate()
			if found:
				return chromosome, fitness

			# Operators
			newpopulation = []
			for parentA, parentB in self.__Selection():
				childA, childB = self.CROSSOVER_FUNCTION(parentA, parentB)
				childA = self.MUTATION_FUNCTION(childA)
				childB = self.MUTATION_FUNCTION(childB)

				# Length Operator
				if self.LENGTH_MUTATION_RATE != 0:
					childA = self.__LengthMutation(childA)
					childB = self.__LengthMutation(childB)

				# Add to current population
				newpopulation.append(childA)
				newpopulation.append(childB)

			# Replace Population
			self.population = newpopulation

		return chromosome, fitness
