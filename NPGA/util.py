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
import random
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
