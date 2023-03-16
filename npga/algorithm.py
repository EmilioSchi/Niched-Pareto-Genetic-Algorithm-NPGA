# File: algorithm.py
# Niched Pareto Genetic Algorithm (npga)
#
# Author: Emilio Schinina' <emilioschi@gmail.com>
# Copyright (C) 2019, 2023 Emilio Schinina'
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import math
import random
import queue
import numpy as np
import concurrent.futures
from typing import Callable, List, Optional

class _Utility:

    @staticmethod
    def _is_non_dominatedable(costs, return_mask = True):
        """
        Find the pareto-efficient points

        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.

        Parameters
        ----------
        costs
            An (n_points, n_costs) array
        return_mask
            True to return a mask
        Returns
        -------
        non_dominatedable
            An array of indices of pareto-efficient points.
        """

        non_dominatedable = np.arange(costs.shape[0])
        points_number = costs.shape[0]

        # Next index in the non_dominatedable array to search for
        next_point_index = 0
        while next_point_index < len(costs):
            non_dominated_point_mask = np.any(costs < costs[next_point_index], axis = 1)
            non_dominated_point_mask[next_point_index] = True
            # Remove dominated points
            non_dominatedable = non_dominatedable[non_dominated_point_mask]
            costs = costs[non_dominated_point_mask]
            next_point_index = np.sum(non_dominated_point_mask[:next_point_index]) + 1
        if return_mask:
            non_dominatedable_mask = np.zeros(points_number, dtype = bool)
            non_dominatedable_mask[non_dominatedable] = True
            return non_dominatedable_mask
        else:
            return non_dominatedable

    @staticmethod
    def _euclidean_distance(point_1, point_2):
        """
        Calculate the Euclidean distance between two points.

        Parameters
        ----------
        point_1 : numpy.ndarray
            A 1D array representing the first point.
        point_2 : numpy.ndarray
            A 1D array representing the second point.

        Returns
        -------
        float
            The Euclidean distance between the two points.
        """
        diff = point_1 - point_2
        distance = np.linalg.norm(diff)
        return distance

    @staticmethod
    def _flip_coin(probability):
        """
        Simulate a coin flip with the specified probability 
        of landing heads.

        Parameters
        ----------
        probability : float
            The probability of landing heads on a single coin flip.

        Returns
        -------
        bool
            True if the coin lands heads, False if it lands tails.
        """
        return random.random() < probability

class Statistics:
    def __init__(self, optimal_fitness):
        self.current_population = []
        self.pareto_front = []

        self._optimal_fitness = optimal_fitness

        self.best_euclidean_chromosome = Chromosome('', 0)

    def __str__(self):
        return (f"\nBest Euclidean solution:\n{self.best_euclidean_chromosome}")

    def _found_solution_with_min_distance(self):

        # Loop through each chromosome in the current population.
        for chromosome in self.pareto_front:

            # Calculate the Euclidean distance between the 
            # chromosome's fitness and the optimal fitness.
            distance_calculated = _Utility._euclidean_distance(
                self._optimal_fitness, chromosome.fitness)

            # Initialize the best distance to the highest number.
            best_distance = float('inf')

            # If the current distance is better than the best distance found 
            # so far, update the best distance and set the best Euclidean 
            # chromosome to be the current chromosome.
            if best_distance > distance_calculated:
                best_distance = distance_calculated
                self.best_euclidean_chromosome = chromosome

    def _found_pareto_points(self, population):
        """
        Add non-dominated solutions in the given population to the Pareto set.

        Parameters
        ----------
        population : list
            A list of chromosomes, each representing a 
            solution to the optimization problem.
        """
        compare = []
        compare.extend(population)
        compare.extend(self.pareto_front)

        fitness = np.asarray([item.fitness_to_minimize for item in compare] \
                             , dtype = np.float64)

        non_dominatedable = _Utility._is_non_dominatedable(fitness)

        # Update the pareto front adding new points
        self.pareto_front = [chromosome \
                             for single_non_dominatedable, chromosome \
                             in zip(non_dominatedable, compare) \
                             if single_non_dominatedable]

    def update(self, population):
        # Update current population
        self.current_population = population

        self._found_pareto_points(population)

        self._found_solution_with_min_distance()

        return self.pareto_front

class Chromosome:
    def __init__(self, genes: list, dimension: int, fitness = -1, 
                 fitness_to_minimize = -1, problem_type: str = ''):
        """
        Initialize a new Chromosome object.

        Each `Chromosome` object contains a set of genes that represent the 
        solution candidate, along with its fitness values, dimension, and 
        problem type. The fitness value is a measure of how good the solution 
        candidate is in solving the problem, and can be used to select 
        chromosomes for the next generation.

        Parameters
        ----------
        genes : list
            A list of genes that represent the genetic code of the chromosome.
        dimension : int
            The length of the chromosome.
        fitness
            The fitness value of the chromosome. Defaults to -1.
        fitness_to_minimize
            The fitness value of the chromosome in minimizing the problem.
            Defaults to -1.
        problem_type : str, optional
        The type of problem that the chromosome is solving. Defaults to ''.
        """

        # Store the genetic code as a string, 
        # the length, the fitness value of the chromosome
        self.genes = ''.join(genes)
        self.length = dimension
        self.fitness = fitness

        # If the problem is a maximization problem, store the negation of 
        # the fitness value as the fitness to minimize
        # If the problem is a minimization problem, store the 
        # fitness value as the fitness to minimize
        self.fitness_to_minimize = fitness_to_minimize

        # There is the possibility that in case we want to switch
        # the problem type in runtime
        self.problem_type = problem_type

    def __str__(self):
        """
        Return a string representation of the Chromosome object.

        Returns
        -------
        str: A string that summarizes the chromosome's attributes.
        """
        return (f"Chromosome with genes: {self.genes}\n"
                f"Dimension: {self.length}\n"
                f"Fitness: {self.fitness}\n"
                f"Fitness to minimize: {self.fitness_to_minimize}\n"
                f"Problem type: {self.problem_type}")

class Algorithm:
    def __init__(
        self,
        objective_function: Callable,
        optimal_fitness: float,
        chromosome_length_set: List[int],
        chromosome_set: str = '01',
        display_function: Optional[Callable] = None,
        population_size: int = 30,
        max_generation: int = 100,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.05,
        length_mutation_rate: float = 0,
        growth_rate: float = 0.5,
        shrink_rate: float = 0.5,
        prc_tournament_size: float = 0.1,
        candidate_size: int = 2,
        niche_radius: float = 1,
        multithread_mode: bool = False,
        history_recover_fitness: bool = False
    ):
        """
        Initializes the Niched Pareto Genetic Algorithm.

        Parameters
        ----------
        objective_function : function
            The fitness function to evaluate the solution domain.
        display_function : function
            At the end of each generation, it is possible to call the display 
            function to see the state of the algorithm and useful statistics.
        optimal_fitness : float
            The solution that wants to be reached.
        chromosome_set : str
            A set of characters used in the chromosome.
        chromosome_length_set : list
            A list of sizes that the chromosome can be assumed.
        population_size : int, optional
            The number of individuals present in each generation, by default 30.
        max_generation : int, optional
            A maximum number of generations, by default 100.
        crossover_rate : float, optional
            Crossover probability, a float between 0 and 1, says how often 
            crossover will be performed. If there is a crossover, 
            offspring is made from parts of the parents' chromosome, otherwise, 
            if there is no crossover, offspring is an exact copy of the parents.
            Crossover is made in hope that new chromosomes will have good parts 
            of old chromosomes and maybe the new chromosomes will be better, 
            by default 0.7.
        mutation_rate : float, optional
            Mutation probability, a float between 0 and 1, says how often parts 
            of the chromosome will be mutated. If mutation is performed, part 
            of the chromosome is changed. Mutation is made to prevent the GA 
            from falling into a local extreme, but it should not occur very 
            often, because then the GA will in fact change to a random search, 
            by default 0.05.
        length_mutation_rate : float, optional
            Length Mutation probability, a float between 0 and 1, says how 
            often a change in size of chromosome will occur. The lengths of 
            both the parent chromosomes are checked and the chromosome whose 
            length is smaller is taken as parent 1. If the lengths of both the 
            chromosomes are the same, the exchange doesn't happen. Then, two 
            crossover points are picked randomly for the parent 1. 
            The bits in between the two points are swapped between the parent, 
            by default 0.
        growth_rate : float, optional
            In growth mutation, the chromosome is enlarged, by default 0.5.
        shrink_rate : float, optional
            The purpose of shrink mutation is to reduce the length of the 
            chromosome, by default 0.5.
        prc_tournament_size : float, optional
            The percentage of the population that will form a comparison set in 
            tournament selection, a float between 0 and 1, by default 0.1.
        candidate_size : int, optional
            The number of candidate chromosomes that can be selected as 
            parents, by default 2.
        niche_radius : float, optional
            Niche Radius is the distance threshold below which two individuals 
            are considered similar enough to affect the niche count. 
            The concept of Niche was introduced to ensure the diversity of 
            individuals and prevent individuals converging into a narrow region 
            of solution space, the range of niche is a spherical area. It is 
            fixed by the user at some estimate of the minimal separation 
            expected between the goal solutions, by default 1.
        multithread_mode : bool, optional
            By default False.
        history_recover_fitness : bool, optional
            If a solution is already seen, the algorithm takes the old value 
            without computing the objective function, by default False.
        """

        if not 0 <= crossover_rate <= 1:
            raise ValueError("Crossover Rate must take values between 0 and 1.")

        if not 0 <= mutation_rate <= 1:
            raise ValueError("Mutation Rate must take values between 0 and 1.")

        if not 0 <= length_mutation_rate <= 1:
            raise ValueError("Length Mutation Rate must take values between 0 and 1.")

        if not 0 <= prc_tournament_size <= 1:
            raise ValueError("The percentage of tournament size must take values "
                             "between 0 and 1.")

        if population_size < 4:
            raise ValueError("Population size is very small.")

        if candidate_size < 2:
            raise ValueError("Candidate must be at least 2.")

        if max_generation < 1:
            raise ValueError("Generation must be positive.")

        self.population = []

        # Chromosome already seen
        self._history = []

        # Functions
        self._objective_function = objective_function
        self._display_function = display_function

        # Parameter of classic Genetic Algorithm
        self._chromosome_set = chromosome_set
        self._length_set = chromosome_length_set
        p = population_size # even number of population size
        self._population_size = p if p % 2 == 0 else p + 1
        self._max_generation = max_generation
        self._crossover_rate = crossover_rate
        self._mutation_rate = mutation_rate

        # MultiObjective variables
        self._optimal_fitness = np.asarray(optimal_fitness, dtype = np.float64)
        self._number_objective = len(optimal_fitness)

        # Variable length chromosome variable
        if len(self._length_set) == 1:
            self._length_mutation_rate = 0
        else:
            self._length_mutation_rate = length_mutation_rate
        self._max_length_set = max(chromosome_length_set)
        self._min_length_set = min(chromosome_length_set)
        self._growth_rate = growth_rate
        self._shrink_rate = shrink_rate

        # Pareto Niched Selection Tournament parameters
        self._candidate_size = candidate_size
        self._t_dom = math.floor(prc_tournament_size * self._population_size)
        self._niched_radius = niche_radius

        self._multithread_mode = multithread_mode
        self._history_recover_fitness = history_recover_fitness

        # Statistic parameters
        self.statistics = Statistics(self._optimal_fitness)

    def _add_solution_to_history_list(self, genes, fitness, 
                                      fitnessfominimise, problem_type):
        if self._history_recover_fitness:
            self._history.append(Chromosome(genes, len(genes), fitness, 
                                            fitnessfominimise, problem_type))

    def _already_seen_solution(self, genes):
        if self._history_recover_fitness:# and (genes in self.history['genes']):
            entry = next((item for item in self._history if item.genes == genes), False)
            if entry:
                return entry, True
            else:
                return None, False
        else:
            return None, False

    def _objective_function_thread(self, genes, queue):

        entry, history_found = self._already_seen_solution(genes)

        # Check if the solution are previously calculated
        if history_found:
            queue.put((genes, entry.fitness, entry.problem_type, history_found))
        else:

            # Unpack the results of the objective functions 
            # into separate arrays for fitness and problem types
            fitness, problem_types = zip(*self._objective_function(genes))

            # Convert the fitness array to a NumPy array of type np.float64
            fitness = np.array(fitness, dtype=np.float64)

            # Convert the problem types array to a NumPy array
            problem_types = np.array(problem_types)

            # Put the solutions on atomic queue
            queue.put((genes, fitness, problem_types, history_found))

    def _check_solution(self, values, problem_types):
        found = True
        for i, (value, problem_type) in enumerate(zip(values, problem_types)):
            if problem_type == 'minimize':
                found = found and (value <= self._optimal_fitness[i])
            elif problem_type == 'maximize':
                found = found and (value >= self._optimal_fitness[i])
        return found

    def _convert_maximize_to_minimize(self, values, problem_types):

        fitness_to_minimize = np.zeros((self._number_objective,), dtype = np.float64)

        # I want to convert all functions which are to
        # be minimized into a form which allows their
        # maximization.
        for i, (fitness, problem_type) in enumerate(zip(values, problem_types)):

            if problem_type == 'minimize':
                fitness_to_minimize[i] = fitness
            # Maximization problem is the negation of Minimazion problem
            elif problem_type == 'maximize':
                fitness_to_minimize[i] = -fitness
            else:
                raise ValueError("Problem type can be minimize or maximize.")

        return fitness_to_minimize

    def _multithread_evaluation(self):
        '''
        Submit a new task to the thread pool for each 
        chromosome in the population
        '''

        # Queue to store objective function results
        queued_request = queue.Queue()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for chromosome in self.population:
                executor.submit(self._objective_function_thread, 
                                chromosome.genes, queued_request)

        tmp = []
        for _ in range(queued_request.qsize()):
            genes, fitness, problem_types, history_found = queued_request.get()

            fitness_to_minimize = self._convert_maximize_to_minimize(fitness,
                                                                     problem_types)

            tmp.append(Chromosome(genes, len(genes), fitness, 
                                  fitness_to_minimize, problem_types))

            # Store chromosome in already seen list
            if not history_found:
                self._add_solution_to_history_list(genes, fitness, 
                                                   fitness_to_minimize, problem_types)

            found = self._check_solution(fitness, problem_types)

        self.population = tmp

        pareto_solutions = self.statistics.update(self.population)

        if self._display_function is not None:
            self._display_function(self.statistics)

        return pareto_solutions, found

    def _evaluation(self):
        """
        Evaluate the fitness of each candidate solution based on 
        the objective functions and any constraints.
        """

        if self._multithread_mode:
            return self._multithread_evaluation()

        for chromosome in self.population:
            entry, history_found = self._already_seen_solution(chromosome.genes)
            if history_found:
                chromosome.fitness = entry.fitness
                chromosome.fitness_to_minimize = entry.fitness_to_minimize
                chromosome.problem_type = entry.problem_type
            else:

                # Unpack the results of the objective functions 
                # into separate arrays for fitness and problem types
                fitness, problem_types = zip(*self._objective_function(chromosome.genes))

                # Convert the fitness array to a NumPy array of type np.float64
                chromosome.fitness = np.array(fitness, dtype=np.float64)

                # Convert the problem types array to a NumPy array
                problem_types = np.array(problem_types)

                chromosome.fitness_to_minimize = self._convert_maximize_to_minimize(
                    chromosome.fitness, problem_types)

                # Store chromosome in already seen list
                self._add_solution_to_history_list(chromosome.genes, 
                                                    chromosome.fitness,
                                                    chromosome.fitness_to_minimize, 
                                                    problem_types)

                found = self._check_solution(chromosome.fitness, problem_types)

        pareto_solutions = self.statistics.update(self.population)
        self._display_function(self.statistics)

        return pareto_solutions, found

    def _pareto_domination_tournment(self):
        """
        Few candidate chromosomes and a comparison set, of size t_dom, of
        chromosomes are chosen for selection at random from the population.
        """

        compareindexset = random.sample(range(self._population_size),
                                        k = self._candidate_size + self._t_dom)

        # Each of candidates are then compared against each individual
        # in the comparison set.
        non_dominatedable = np.ones((self._candidate_size,), dtype=bool)
        for e, i in enumerate(compareindexset[:self._candidate_size]):
            costs = np.zeros((1 + self._t_dom, self._number_objective),
                                dtype = np.float64)
            costs[0] = self.population[i].fitness_to_minimize

            for z, j in enumerate(compareindexset[self._candidate_size:]):
                costs[z + 1] = self.population[j].fitness_to_minimize

            # I want to know if first add fitness is non dominatedable
            non_dominatedable[e] = _Utility._is_non_dominatedable(costs)[0]

        """
        If one candidate is dominate by the comparison set, and the other
        is not, the latter is selected for reproduction. If neither or both
        are dominated by the comparison set, then we must use sharing to
        choose a winner.
        """
        if np.count_nonzero(non_dominatedable == True) == 1:
            itemindex = np.where(non_dominatedable == True)[0][0]
            candidateindex = compareindexset[itemindex]
            return self.population[candidateindex], False, []
        else:
            return None, True, compareindexset[:self._candidate_size]

    def _fitness_sharing(self, candidate_indexes):
        """
        Applies fitness sharing to the candidates in `candidate_indexes`, 
        which are assumed to belong to the same niche.
    
        Fitness sharing is a mechanism that encourages diversity by 
        penalizing the fitness of similar solutions. In this
        implementation, the fitness of each candidate is divided by 
        the sum of the niche counts of all candidates that are
        within a certain distance from it.
    

        Parameters
        ----------
        candidate_indexes (array-like): The indices of the candidates to be evaluated.
    
        Returns
        -------
        Chromosome: The selected candidate, which has the smallest niche count.
        """

        distances = np.zeros((self._candidate_size,), dtype = np.float64)
        for e, i in enumerate(candidate_indexes):
            distances[e] = self._niched_count_calculation(self.population[i])

        # If we want to maintain useful diversity, it would be best to
        # choose the candidate that has the smaller niche count.
        itemindex = np.where(distances == distances.min())[0][0]
        candidateindex = candidate_indexes[itemindex]

        return self.population[candidateindex]

    def _niched_count_calculation(self, cadidate):
        """
        The niche count is initialized to zero, and is updated 
        foreach near neighbor.
        """

        niche_count = 0
        for individual in self.population:
            """
            The formula increments the niche count by 
            (1 - (neighbor_distance / niched_radius)) 
            if the neighbor distance is less then niched radius.
            """

            neighbor_distance = _Utility._euclidean_distance(
                cadidate.fitness, individual.fitness)
            
            if neighbor_distance <= self._niched_radius:
                sh = 1.0 - (neighbor_distance / self._niched_radius)
            else:
                sh = 0

            niche_count = niche_count + sh

        return niche_count

    def _selection_operation(self):
        """
        Selection maintains diversity in the population and 
        ensures that the best chromosomes are passed on to the next generation.

        A small number of chromosomes are selected at random
        from the population, and the best chromosome among 
        the selected chromosomes is chosen for the next generation.
        """

        for _ in range(self._population_size // 2):

            # Pareto Domination Tournment Selection
            parent_1, dominated, candidate_indexes = self._pareto_domination_tournment()

            # Fitness Sharing if applicable
            if dominated:
                parent_1 = self._fitness_sharing(candidate_indexes)

            # Pareto Domination Tournment Selection
            parent_2, dominated, candidate_indexes = self._pareto_domination_tournment()

            # Fitness Sharing if applicable
            if dominated:
                parent_2 = self._fitness_sharing(candidate_indexes)

            yield (parent_1, parent_2)

    def _generation_initial_population(self):
        # It generates random string
        genlen = random.choices(self._length_set, k = 1)[0]
        child = random.choices(self._chromosome_set, k = genlen)
        return Chromosome(child, genlen)

    def _shrink_mutation_operation(self, parent):
        """
        Shrink mutation is a type of mutation operation that 
        reduces the magnitude of the gene values. 
        
        The operation is performed by subtracting a random 
        value from the gene value, with the magnitude of the 
        random value proportional to the magnitude of the gene value. 

        This operation can be used to reduce the size of 
        large gene values and to prevent the search from 
        getting stuck in large, but suboptimal, regions of the search space.
        """

        child = []

        lengthset = [i for i in self._length_set if i < parent.length]
        mutationLenght = random.sample(lengthset, 1)[0]
        mutationquantity = parent.length - mutationLenght

        # It erases genes at the end
        child.extend(parent.genes[:-mutationquantity])

        # Erase genes length
        return Chromosome(child, parent.length - mutationquantity)

    def _growth_mutation_operation(self, parent):
        """
        Growth mutation is a type of mutation operation that 
        increases the magnitude of the gene values. 
        
        The operation is performed by adding a random value 
        to the gene value, with the magnitude of the random 
        value proportional to the magnitude of the gene value. 

        This operation can be used to increase the size of 
        small gene values and to allow the search to explore 
        new regions of the search space.
        """

        child = []

        lengthset = [i for i in self._length_set if i > parent.length]
        mutationLenght = random.sample(lengthset, 1)[0]
        mutationquantity = mutationLenght - parent.length

        # Insert random correct genes at the end of chromosome.
        child.extend(parent.genes)

        # It grows at the end
        child.extend(random.choices(self._chromosome_set, k = mutationquantity))

        # Add genes length
        return Chromosome(child, parent.length + mutationquantity)

    def _length_mutation_operation(self, parent):
        """
        This operation can be used to increase or to reduce the size of 
        gene values to allow the search to explore new regions of the search 
        space.
        """

        if _Utility._flip_coin(self._length_mutation_rate):

            # Check if the length of the parent's chromosome matches the 
            # maximum or minimum length set
            if self._max_length_set == parent.length:
                return self._shrink_mutation_operation(parent)

            if self._min_length_set == parent.length:
                return self._growth_mutation_operation(parent)

            # Create a dictionary that maps integers to the two functions
            switcher = { 
                0 : self._growth_mutation_operation, 
                1 : self._shrink_mutation_operation
            }

            # Store the probabilities in a list
            probabilities = [self._growth_rate, self._shrink_rate]

            # Normalize the probabilities so they sum up to 1
            normalized_probabilities = [p / sum(probabilities) for p in probabilities]

            # Calculate the cumulative probability for each function
            chunks = [sum(normalized_probabilities[:i+1]) for i in range(2)]

            # Generate a random number between 0 and 1
            arrow = random.random()

            # Loop over the cumulative probabilities and select the function
            # corresponding to the first cumulative probability that the arrow
            # is less than or equal to
            for index, interval_end in enumerate(chunks):
                if arrow <= interval_end:
                    # Get the selected function from the dictionary
                    function = switcher.get(index, lambda :'Invalid')

                    # Return and execute the selected function
                    return function(parent)
        else:
            return parent

    def _mutation_operation(self, parent):
        """ 
        Mutation consists of the variation of a randomly chosen bit, 
        belonging to a randomly selected string.

        If there is no mutation, offspring is taken without any change.
        """

        child = []
        for i in range(parent.length):

            # Mutation probability says how often will be parts of chromosome mutated.
            if _Utility._flip_coin(self._mutation_rate):
                new_gene, new_gene_alternate = random.sample(self._chromosome_set, 2)
                child.extend(new_gene_alternate 
                             if new_gene == parent.genes[i] 
                             else new_gene)
            else:
                child.extend(parent.genes[i])

        return Chromosome(child, parent.length)

    def _crossover_operation(self, parent_1, parent_2):
        """
        Two points on both parents' chromosomes is picked randomly, and
        designated 'crossover points'. This results in
        two offspring, each carrying some genetic information from both parents.

        Crossover probability says how often will be crossover performed.
        If there is no crossover, offspring is exact copy of parents.
        """

        if _Utility._flip_coin(self._crossover_rate):
            child_1, child_2 = [], []

            # The lengths of both the parent chromosomes are checked and the
            # chromosome whose length is smaller is taken as parent A.
            len_min = min(parent_1.length, parent_2.length)
            if parent_1.length != len_min:
                # Swap the two strings
                parent_1, parent_2 = parent_2, parent_1

            # Crossover points are randomly chosen for parent A
            startpoint, endpoint = random.sample(range(1, len_min + 1), 2)
            startpoint, endpoint = min(startpoint, endpoint), max(startpoint, endpoint)
                
            # The genetic material in the two parents is exchanged 
            # between the two crossover points of the parents
            child_1.extend(parent_2.genes[0:startpoint])
            child_1.extend(parent_1.genes[startpoint:endpoint])
            child_1.extend(parent_2.genes[endpoint:])

            child_2.extend(parent_1.genes[0:startpoint])
            child_2.extend(parent_2.genes[startpoint:endpoint])
            child_2.extend(parent_1.genes[endpoint:])

            return Chromosome(child_1, parent_2.length), \
                Chromosome(child_2, parent_1.length)
        else:
            return parent_1, parent_2

    def run(self):
        """
        The main function of the genetic algorithm that 
        initializes the population and uses selection,
        crossover, and mutation to find optimal solutions.

        Returns
        -------
        pareto_solutions
            A list of solutions belonging to the Pareto front.
        """

        # Initialize population
        for _ in range(self._population_size):
            child = self._generation_initial_population()
            self.population.append(child)

        # Main loop of the genetic algorithm
        for _ in range(self._max_generation):
            # Calculation of fitness
            pareto_solutions, found = self._evaluation()
            if found:
                return pareto_solutions

            # Operators
            new_population = []
            for parent_a, parent_b in self._selection_operation():
                child_1, child_2 = self._crossover_operation(parent_a, parent_b)
                child_1 = self._mutation_operation(child_1)
                child_2 = self._mutation_operation(child_2)

                # Length Operator
                if self._length_mutation_rate != 0:
                    child_1 = self._length_mutation_operation(child_1)
                    child_2 = self._length_mutation_operation(child_2)

                # Add to current population
                new_population.append(child_1)
                new_population.append(child_2)

            # Replace Population
            self.population = new_population

        return pareto_solutions
