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

class Chromosome:
	def __init__(self, genes, dimention, fitness = -1, fitnessToMinimise = -1, problemtype = '        '):
		self.Genes			 = ''.join(genes)
		self.Length			 = dimention
		self.Fitness		 = fitness
		# maximization problem is the negation of minimazion problem
		self.FitnessToMinimise  = fitnessToMinimise
		# There is the possibility that in case we want to switch
		# the problem type in runtime
		self.ProblemType  = problemtype
