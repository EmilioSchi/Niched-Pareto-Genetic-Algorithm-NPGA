"""
npga
====

Niched Pareto Genetic Algorithm (npga)

NPGA is used to optimize problems with multiple objectives to 
provide a set of non-dominated solutions, also known as the 
Pareto-optimal set, which represents the best possible trade-offs between 
the multiple objectives in the optimization problem.

Background information
----------------------

The algorithm can be summarized in the following steps:

1. Initialization: Create an initial population of candidate solutions randomly.

2. Evaluate fitness: Evaluate the fitness of each candidate solution based on 
                     the objective functions and any constraints.

3. Selection: Select the best individuals in the population for reproduction 
              using a selection method, such as tournament selection.

4. Crossover: Use crossover to create offspring solutions from the selected 
              individuals. The crossover operator combines the genetic material 
              of two individuals to create a new one.
              
5. Mutation: Apply a mutation operator to introduce random changes to the 
             offspring solutions.

6. Evaluate fitness: Evaluate the fitness of the offspring solutions.

7. Niching: Apply a niching mechanism to encourage the population to spread out 
            in the search space and avoid clustering in a few regions.

8. Replacement: Replace the least fit individuals in the population with the 
                new offspring solutions.

9. Termination: Check the termination criterion to determine if the algorithm 
                should stop. If not, go back to step 3.
                
10. Pareto front: After the algorithm has terminated, the final result is a set 
                  of non-dominated solutions that represent the Pareto front of 
                  the optimization problem.

The steps of NPGA are similar to those of other genetic algorithms, but with 
the addition of the niching mechanism to maintain diversity in the population. 
By following these steps, developers can use NPGA to find a set of 
non-dominated solutions that provide trade-offs between multiple objectives in 
an optimization problem.

License
------
Author: Emilio Schinina' <emilioschi@gmail.com>
Copyright (C) 2019, 2023 Emilio Schinina'

Licensed under the Apache Lic ense, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

from .algorithm import Algorithm

# used for setup.py
__version__ = '0.3.1'
