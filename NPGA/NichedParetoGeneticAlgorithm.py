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

from .Statistics import Statistics
from .Chromosome import Chromosome
from .util import FlipCoin, EuclideanDistance, EuclideanDistanceFast, IsNonDominatedable, IsNonDominatedableFast


class NichedParetoGeneticAlgorithm:
	def __init__(self, fnGetFitness, fnDisplay, optimal_fitness, chromosome_set,
		chromosome_length_set, population_size = 30, max_generation = 100,
		crossover_rate = 0.7, mutation_rate = 0.05, length_mutation_rate = 0,
		growth_rate = 0.5, shrink_rate = 0.5, prc_tournament_size = 0.1,
		candidate_size = 2, niche_radius = 1, fastmode = False, multithreadmode = False,
		fnMutation = None, fnCrossover = None, historyrecoverfitness = False):

		assert(isinstance(chromosome_set, str)), "[TYPE ERROR] chromosome_set must be a string."
		assert(isinstance(chromosome_length_set, list)), "[TYPE ERROR] chromosome_length_set must be expressed as a list of lengths."
		assert(isinstance(population_size, int)), "[TYPE ERROR] Population Size must be an integer."
		assert(isinstance(candidate_size, int)), "[TYPE ERROR] The number of candidates must be an integer."
		assert(isinstance(max_generation, int)), "[TYPE ERROR] Maximum number of generation must be an integer."
		assert(isinstance(crossover_rate, (float, int))), "[TYPE ERROR] Crossover Rate must take values between 0 and 1."
		assert(isinstance(mutation_rate, (float, int))), "[TYPE ERROR] Mutation Rate must take values between 0 and 1."
		assert(isinstance(length_mutation_rate, (float, int))), "[TYPE ERROR] Length Mutation Rate must take values between 0 and 1."
		assert(isinstance(prc_tournament_size, (float, int))), "[TYPE ERROR] Length Mutation Rate must take values between 0 and 1."
		assert(isinstance(niche_radius, (float, int))), "[TYPE ERROR] NICHE RADIUS wrong value."
		assert(isinstance(growth_rate, (float, int))), "[TYPE ERROR] Growth Rate wrong value."
		assert(isinstance(shrink_rate, (float, int))), "[TYPE ERROR] Shrink Rate wrong value."
		assert(isinstance(multithreadmode, bool)), "[TYPE ERROR] multithreadmode must be a boolean."
		assert(isinstance(fastmode, bool)), "[TYPE ERROR] fastmode must be a boolean."
		assert(isinstance(historyrecoverfitness, bool)), "[TYPE ERROR] historyrecoverfitness must be a boolean."
		assert(crossover_rate >= 0 and crossover_rate <= 1), "[RANGE ERROR] Crossover Rate must take values between 0 and 1."
		assert(mutation_rate >= 0 and mutation_rate <= 1), "[RANGE ERROR] Mutation Rate must take values between 0 and 1."
		assert(length_mutation_rate >= 0 and length_mutation_rate <= 1), "[RANGE ERROR] Length Mutation Rate must take values between 0 and 1."
		assert(prc_tournament_size >= 0 and prc_tournament_size <= 1), "[RANGE ERROR] The percentage of tournament size must take values between 0 and 1."
		assert(population_size >= 4), "[RANGE ERROR] Population size is very small."
		assert(candidate_size >= 2), "[RANGE ERROR] Candidate must be at least 2."
		assert(max_generation >= 1), "[RANGE ERROR] Generation must be positive."

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

	# Only dynamic structure in the code (I HOPE)
	def __AddToHistory(self, genes, fitness, fitnessfominimise, problemtype):
		self.Statistics.Combination += 1
		if self.HISTORYRECOVERFITNESS:
			self.Statistics.History.append(Chromosome(genes, len(genes), fitness, fitnessfominimise, problemtype))

	def __AlreadySeen(self, genes):
		if self.HISTORYRECOVERFITNESS:# and (genes in self.history['Genes']):
			entry = next((item for item in self.Statistics.History if item.Genes == genes), False)
			if entry:
				return entry, True
			else:
				return None, False
		else:
			return None, False

	def __ThreadObjectiveFunction(self, genes, queue):
		entry, historyfound = self.__AlreadySeen(genes)
		if historyfound:
			queue.put((genes, entry.Fitness, entry.ProblemType, historyfound))
		else:
			# call objective_function
			fitness = np.zeros((self.NUMBER_OBJECTIVE,), dtype = np.float64)
			# Declaration of list of strings. I hope static variable is faster than dynamic
			problemtypes = ['        ' for s in range(self.NUMBER_OBJECTIVE)]
			for i, (singlefitness, problemtype) in enumerate(self.OBJECTIVE_FUNCTION(genes)):
				fitness[i] = singlefitness
				problemtypes[i] = problemtype
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
			tmp.append(Chromosome(genes, len(genes), fitness, fitnessToMinimise, problemtypes))
			# Store chromosome in already seen list
			if not historyfound:
				self.__AddToHistory(genes, fitness, fitnessToMinimise, problemtypes)

			solutionfound = self.__CheckSolution(fitness, problemtypes)

		self.population = tmp

		ParetoSolutions = self.Statistics.Update(self.population)
		self.DISPLAY_FUNCTION(self.Statistics)

		return ParetoSolutions, solutionfound

	def __Evaluate(self):
		if self.MULTITHREADMODE:
			return self.__MultiThreadEvaluate()

		for chromosome in self.population:
			entry, historyfound = self.__AlreadySeen(chromosome.Genes)
			if historyfound:
				chromosome.Fitness = entry.Fitness
				chromosome.FitnessToMinimise = entry.FitnessToMinimise
				chromosome.ProblemType = entry.ProblemType
			else:
				chromosome.Fitness = np.zeros((self.NUMBER_OBJECTIVE,), dtype = np.float64)
				# Declaration of list of strings. I hope static variable is faster than dynamic
				problemtypes = ["        " for s in range(self.NUMBER_OBJECTIVE)]

				# call objective function
				for i, (singlefitness, problemtype) in enumerate(self.OBJECTIVE_FUNCTION(chromosome.Genes)):
					chromosome.Fitness[i] = singlefitness
					problemtypes[i] = problemtype
				chromosome.FitnessToMinimise = self.__ConvertMaximizeToMinimize(chromosome.Fitness, problemtypes)

				# Store chromosome in already seen list
				self.__AddToHistory(chromosome.Genes, chromosome.Fitness, chromosome.FitnessToMinimise, problemtypes)

				solutionfound = self.__CheckSolution(chromosome.Fitness, problemtypes)

		ParetoSolutions = self.Statistics.Update(self.population)
		self.DISPLAY_FUNCTION(self.Statistics)

		return ParetoSolutions, solutionfound

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
		return Chromosome(child, genlen)

	def __ShrinkMutation(self, parent):
		# Erase genes.
		child = []

		lengthset = [i for i in self.LENGTH_SET if i < parent.Length]
		mutationLenght = random.sample(lengthset, 1)[0]
		mutationquantity = parent.Length - mutationLenght

		# It erases genes at the end
		child.extend(parent.Genes[:-mutationquantity])

		return Chromosome(child, parent.Length - mutationquantity)

	def __GrowthMutation(self, parent):
		child = []

		lengthset = [i for i in self.LENGTH_SET if i > parent.Length]
		mutationLenght = random.sample(lengthset, 1)[0]
		mutationquantity = mutationLenght - parent.Length

		# Insert random correct genes at the end of chromosome.
		child.extend(parent.Genes)
		# It grows at the end
		child.extend(random.choices(self.CHROMOSOME_SET, k = mutationquantity))

		return Chromosome(child, parent.Length + mutationquantity)

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
		return Chromosome(child, parent.Length)

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

			return	Chromosome(childA, parentB.Length), \
					Chromosome(childB, parentA.Length)
		else:
			return parentA, parentB


	def Evolution(self):
		for _ in range(self.POPULATION_SIZE):
			child = self.__Generation()
			self.population.append(child)

		for _ in range(self.MAX_GENERATIONS):
			# Calculation of fitness
			ParetoSolutions, found = self.__Evaluate()
			if found:
				return ParetoSolutions

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

		return ParetoSolutions
