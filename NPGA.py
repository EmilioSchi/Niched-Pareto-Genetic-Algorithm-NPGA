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
	def __init__(self, number_objective, fastmode):
		self.NUMBER_OBJECTIVE	 = number_objective
		self.FASTMODE			 = fastmode
		self.number_combination	 = 0
		self.population_size	 = 0
		# BUG OF PYTHON3 (SAME ADDRESS POINTER ASSIGNMENT)
		#self.BEST = [{'Genes': [], 'Value': 999999, 'Fitness' : []}] * self.NUMBER_OBJECTIVE
		self.best = []
		for _ in range(self.NUMBER_OBJECTIVE):
			self.best.append({'Genes': [], 'Value': 999999, 'Fitness' : []})

		self.EuclideanBetter = {'Genes': [], 'Distance': 999999, 'Fitness' : []}
		self.sum_fitness		 = [0] * self.NUMBER_OBJECTIVE
		self.avg				 = [0] * self.NUMBER_OBJECTIVE
		self.HistoryGenes		 = []
		self.HistoryFitness		 = []
		self.nonDominatedables	 = []

	def Update(self, population, history):
		self.number_combination = len(history['Genes'])
		self.sum_fitness = [0] * self.NUMBER_OBJECTIVE
		self.population_size = 0
		for individual in population:

			if self.EuclideanBetter['Distance'] > np.sum(individual.Fitness):
				self.EuclideanBetter['Distance'] = np.sum(individual.Fitness)
				self.EuclideanBetter['Genes'] = ''.join(individual.Genes)
				self.EuclideanBetter['Fitness'] = individual.Fitness

			for i, fitness in enumerate(individual.Fitness):
				self.sum_fitness[i] = self.sum_fitness[i] + fitness

				if self.best[i]['Value'] > fitness:
					self.best[i]['Value'] = fitness
					self.best[i]['Genes'] = ''.join(individual.Genes)
					self.best[i]['Fitness'] = individual.Fitness

			self.population_size = self.population_size + 1

		for i, sum in enumerate(self.sum_fitness):
			self.avg[i] = sum / self.population_size

		self.HistoryGenes = history['Genes']
		self.HistoryFitness = np.asarray(history['Fitness'], dtype = np.float64)
		# I want to know if first add fitness is nonDominatedable
		if self.FASTMODE:
			self.nonDominatedable = IsNonDominatedableFast(self.HistoryFitness)
		else:
			self.nonDominatedable = IsNonDominatedable(self.HistoryFitness)
		return self.EuclideanBetter['Genes'], self.EuclideanBetter['Fitness']

class Chromosome:
	def __init__(self, dimention, genes, fitness, strategy):
		self.Length		 = dimention
		self.Genes		 = genes
		self.Fitness	 = fitness
		self.Strategy	 = strategy
		self.Age		 = 0

class NichedParetoGeneticAlgorithm:
	def __init__(self, fnGetFitness, fnDisplay, optimal_fitness, chromosome_set,
	chromosome_length_set, population_size = 30, max_generation = 100,
	crossover_rate = 0.7, mutation_rate = 0.05, length_mutation_rate = 0,
	growth_rate = 0.5, shrink_rate = 0.5, prc_tournament_size = 0.1,
	candidate_size = 2, niche_radius = 1, fastmode = False,
	fnMutation = None, fnCrossover = None):
		# Functions
		self.OBJECTIVE_FUNCTION	 = fnGetFitness
		self.DISPLAY_FUNCTION	 = fnDisplay

		# Custom operators
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
		self.FASTMODE			 = fastmode

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
		self.CANDIDATE_SIZE		 = candidate_size
		self.T_DOM				 = math.floor(prc_tournament_size * self.POPULATION_SIZE)
		self.NICHE_RADIUS		 = niche_radius

		# Statistic parameters
		self.Statistics = Statistics(self.NUMBER_OBJECTIVE, self.FASTMODE)

		self.population = []
		self.history = {'Genes' : [], 'Fitness' : []}

	def __Evaluate(self):
		for chromosome in self.population:
			if chromosome.Genes in self.history['Genes']:
				index = self.history['Genes'].index(chromosome.Genes)
				chromosome.Fitness = self.history['Fitness'][index]
			else:
				# call objective_function
				chromosome.Fitness = self.OBJECTIVE_FUNCTION(chromosome.Genes)
				chromosome.Fitness = np.asarray(chromosome.Fitness, dtype = np.float64)
				self.history['Genes'].append(chromosome.Genes)
				self.history['Fitness'].append(chromosome.Fitness)
				if np.all(chromosome.Fitness <= self.OPTIMAL_FITNESS):
					return ''.join(chromosome.Genes), chromosome.Fitness, True

		EDGenes, EDfitness = self.Statistics.Update(self.population, self.history)
		self.DISPLAY_FUNCTION(self.population, self.Statistics)
		return EDGenes, EDfitness, False

	def __ParetoDominationTournments(self):
		# Few candidate chromosomes and a comparison set, of size T_DOM, of
		# chromosomes are chosen for selection at random from the population.
		compareindexset = random.sample(range(self.POPULATION_SIZE), k = self.CANDIDATE_SIZE + self.T_DOM)

		# Each of candidates are then compared against each individual
		# in the comparison set.
		nonDominatedable = [True] * self.CANDIDATE_SIZE
		for e, i in enumerate(compareindexset[:self.CANDIDATE_SIZE]):
			fitnesses = []
			fitnesses.append(self.population[i].Fitness)

			for j in compareindexset[self.CANDIDATE_SIZE:]:
				fitnesses.append(self.population[j].Fitness)

			costs = np.asarray(fitnesses, dtype = np.float64)
			# I want to know if first add fitness is nonDominatedable
			if self.FASTMODE:
				nonDominatedable[e] = IsNonDominatedableFast(costs)[0]
			else:
				nonDominatedable[e] = IsNonDominatedable(costs)[0]

		# If one candidate is dominate by the comparison set, and the other
		# is not, the latter is selected for reproduction. If neither or both
		# are dominated by the comparison set, then we must use sharing to
		# choose a winner.
		if nonDominatedable.count(True) == 1:
			candidateindex = compareindexset[nonDominatedable.index(True)]
			return self.population[candidateindex], False, []
		else:
			return None, True, compareindexset[:self.CANDIDATE_SIZE]

	def __FitnessSharing(self, candidateindexes):
		distances = []
		for i in candidateindexes:
			distances.append(self.__NichedCount(self.population[i]))
			# If we want to maintain useful diversity, it would be best to
			# choose the candidate that has the smaller niche count.
			candidateindex = candidateindexes[distances.index(min(distances))]
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
		return Chromosome(genlen, child, -1, "Generation")

	def __Reproduction(self, parentA, parentB):
		# The reproduction operator makes duplicate of a string,
		# according to its fitness value.
		parentA.Age = parentA.Age + 1
		parentB.Age = parentB.Age + 1
		parentA.Strategy = "Reproduction"
		parentB.Strategy = "Reproduction"
		return parentA, parentB

	def __ShrinkMutation(self, parent):
		# Erase genes.
		child = []

		lengthset = [i for i in self.LENGTH_SET if i < parent.Length]
		mutationLenght = random.sample(lengthset, 1)[0]
		mutationquantity = parent.Length - mutationLenght

		# It erases genes at the end
		child.extend(parent.Genes[:-mutationquantity])

		return Chromosome(parent.Length - mutationquantity, child, -1, "ShrinkMutation")

	def __GrowthMutation(self, parent):
		child = []

		lengthset = [i for i in self.LENGTH_SET if i > parent.Length]
		mutationLenght = random.sample(lengthset, 1)[0]
		mutationquantity = mutationLenght - parent.Length

		# Insert random correct genes at the end of chromosome.
		child.extend(parent.Genes)
		# It grows at the end
		child.extend(random.choices(self.CHROMOSOME_SET, k = mutationquantity))

		return Chromosome(parent.Length + mutationquantity, child, -1, "GrowthMutation")

	def __LengthMutation(self, parent):
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

		return Chromosome(parent.Length, child, -1, "Mutation")

	def __Crossover(self, parentA, parentB):
		# Two points on both parents' chromosomes is picked randomly, and
		# designated 'crossover points'. This results in
		# two offspring, each carrying some genetic information from both parents.
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

		return	Chromosome(parentB.Length, childA, -1, "Crossover"), \
				Chromosome(parentA.Length, childB, -1, "Crossover")

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
				# Crossover probability says how often will be crossover performed.
				# If there is no crossover, offspring is exact copy of parents.
				if FlipCoin(self.CROSSOVER_RATE):
					childA, childB = self.CROSSOVER_FUNCTION(parentA, parentB)
				else:
					childA, childB = self.__Reproduction(parentA, parentB)

				childA = self.MUTATION_FUNCTION(childA)
				childB = self.MUTATION_FUNCTION(childB)

				# Length Operator
				if self.LENGTH_MUTATION_RATE != 0:
					if FlipCoin(self.LENGTH_MUTATION_RATE):
						childA = self.__LengthMutation(childA)
					if FlipCoin(self.LENGTH_MUTATION_RATE):
						childB = self.__LengthMutation(childB)

				# Add to current population
				newpopulation.append(childA)
				newpopulation.append(childB)

			# Replace Population
			self.population = newpopulation

		return chromosome, fitness
