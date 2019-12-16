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

import numpy as np
from .Chromosome import Chromosome
from .util import EuclideanDistance, EuclideanDistanceFast, IsNonDominatedable, IsNonDominatedableFast

class Statistics:
	def __init__(self, optimal_fitness, fastmode):
		self.NUMBER_OBJECTIVE	 = len(optimal_fitness)
		self.FASTMODE			 = fastmode
		self.OPTIMAL_FITNESS	 = optimal_fitness
		self.Combination		 = 0
		self.population 		 = []
		self.__population_size	 = 0
		# BUG OF PYTHON3 (SAME ADDRESS POINTER ASSIGNMENT)
		#self.BEST = [{'Genes': [], 'Value': 999999, 'Fitness' : []}] * self.NUMBER_OBJECTIVE

		self.Species = []
		for _ in range(self.NUMBER_OBJECTIVE):
			self.Species.append(Chromosome('', 0))
		self.__speciation_value = [999999] * self.NUMBER_OBJECTIVE

		self.EuclideanBetter = Chromosome('', 0)
		self.__distance = 999999
		self.__sum_fitness		 = np.zeros((self.NUMBER_OBJECTIVE,), dtype = np.float64)
		self.avg				 = np.zeros((self.NUMBER_OBJECTIVE,), dtype = np.float64)
		self.History			 = []
		self.ParetoSet			 = []

	def __FoundSpecies(self, obj_index, chromosome):
		if self.__speciation_value[obj_index]  > chromosome.FitnessToMinimise[obj_index]:
			self.Species[obj_index].Genes = chromosome.Genes
			self.Species[obj_index].Length = chromosome.Length
			self.Species[obj_index].Fitness = chromosome.Fitness
			self.Species[obj_index].FitnessToMinimise = chromosome.FitnessToMinimise
			self.Species[obj_index].ProblemType = chromosome.ProblemType

			self.__speciation_value[obj_index] = chromosome.FitnessToMinimise[obj_index]

	def __MinDistanceFromSolutionCheck(self, chromosome):
		if self.FASTMODE:
			distance = EuclideanDistanceFast(self.OPTIMAL_FITNESS, chromosome.Fitness)
		else:
			distance = EuclideanDistance(self.OPTIMAL_FITNESS, chromosome.Fitness)

		if self.__distance > distance:
			self.__distance = distance
			self.EuclideanBetter.Genes = chromosome.Genes
			self.EuclideanBetter.Length = chromosome.Length
			self.EuclideanBetter.Fitness = chromosome.Fitness
			self.EuclideanBetter.FitnessToMinimise = chromosome.FitnessToMinimise
			self.EuclideanBetter.ProblemType = chromosome.ProblemType

	def __FoundParetoPoints(self, population):
		CompareSet = []
		for chromosome in population:
			CompareSet.append(chromosome)
		for chromosome in self.ParetoSet:
			CompareSet.append(chromosome)

		# Python3 and pointer (EVERY TRICK IS A BUG)
		#CompareSet = population # current population
		#CompareSet += self.ParetoSet # old Pareto Set

		fitness = [item.FitnessToMinimise for item in CompareSet]
		fitness = np.asarray(fitness, dtype = np.float64)

		if self.FASTMODE:
			nonDominatedable = IsNonDominatedableFast(fitness)
		else:
			nonDominatedable = IsNonDominatedable(fitness)

		# Insert new points
		self.ParetoSet = []
		for i, (singleNonDominatedable, chromosome) in enumerate(zip(nonDominatedable, CompareSet)):
			if singleNonDominatedable:
				self.ParetoSet.append(chromosome)

	def __UpdateCurrentPopulation(self, population):
		self.population = population

	def Update(self, population):
		self.__UpdateCurrentPopulation(population)
		#print(len(self.population))
		# Set variables to calculate the average of each objective in current population
		self.__sum_fitness = np.zeros((self.NUMBER_OBJECTIVE,), dtype = np.float64)
		self.__population_size = 0

		for individual in self.population:

			self.__MinDistanceFromSolutionCheck(individual)

			for i, (fitness, fitnessToMinimise) in enumerate(zip(individual.Fitness, individual.FitnessToMinimise)):
				self.__sum_fitness[i] = self.__sum_fitness[i] + fitness

				self.__FoundSpecies(i, individual)

			# The population can be variable. It is better to prevent
			self.__population_size = self.__population_size + 1

		for i, sum in enumerate(self.__sum_fitness):
			self.avg[i] = sum / self.__population_size

		self.__FoundParetoPoints(population)

		return self.ParetoSet
