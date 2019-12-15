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
from .util import FlipCoin, EuclideanDistance, EuclideanDistanceFast, IsNonDominatedable, IsNonDominatedableFast

class Statistics:
	def __init__(self, optimal_fitness, fastmode):
		self.NUMBER_OBJECTIVE	 = len(optimal_fitness)
		self.FASTMODE			 = fastmode
		self.COMPARE_FITNESS	 = optimal_fitness
		self.Combination		 = 0
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
		self.History			 = []
		self.ParetoSet			 = []

	def Update(self, population):
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

			for i, (fitness, fitnessToMinimise) in enumerate(zip(individual.Fitness, individual.FitnessToMinimise)):
				self.sum_fitness[i] = self.sum_fitness[i] + fitness

				if self.best[i]['Value'] > fitnessToMinimise:
					self.best[i]['Value'] = fitness
					self.best[i]['Genes'] = ''.join(individual.Genes)
					self.best[i]['Fitness'] = individual.Fitness

			self.population_size = self.population_size + 1

		for i, sum in enumerate(self.sum_fitness):
			self.avg[i] = sum / self.population_size

		CompareSet = population
		CompareSet += self.ParetoSet
		fitness = [item.FitnessToMinimise for item in CompareSet]
		fitness = np.asarray(fitness, dtype = np.float64)
		if self.FASTMODE:
			nonDominatedable = IsNonDominatedableFast(fitness)
		else:
			nonDominatedable = IsNonDominatedable(fitness)

		self.ParetoSet = []
		for i, (singleNonDominatedable, chromosome) in enumerate(zip(nonDominatedable, CompareSet)):
			if singleNonDominatedable:
				self.ParetoSet.append(chromosome)

		return self.ParetoSet
