<p><img align="left" height="98" src="img/logo.png">
<b><font size="20">Niched Pareto Genetic Algorithm</font></b></p>

Genetic Algorithm (GA) for a  Multi-objective Optimization Problem (MOP)
<br /><br /><br />
# Introduction
To maintain multiple Pareto optimal solutions, Horn et all [1] have altered tournament selection. NPGA uses a tournament selection scheme based on Pareto dominance. Many (conventionally 2 candidates at once) individuals randomly chosen are compared against a subset from the entire population. When both competitors are either dominated or non dominated, the result of the tournament is decided through fitness sharing in the objective domain.

## Pareto Domination Tournments
The initially procedure is based on the random sampling of two groups of individuals from the entire population. *c_{dom}* candidate chromosomes (conventionally 2 candidates at once), that are the candidates for selection as parents, are chosen at random from the population. A comparison set, of size *t_{dom}*, of chromosomes is also chosen randomly from the population. Each of the candidate chromosomes is then compared against the chromosomes of the comparison set, and a non-inferior candidate chromosomes is selected for reproduction. If there is a tie, means neither or both of the candidate chromosomes are non-inferior, then sharing is used to decide the winner. This process continues until the number of solutions that is chosen reaches the initial size of the population. Horn and Nafpliotis [2] found that algorithm was fairly robust with respect to  *t_{dom}*, they found significantly behaviour once *t_{dom}* exceeded this large range of value.
- *t_{dom}* ≈ 1% of N; result in too many dominated solutions (a very fuzzy front).  
- *t_{dom}* ≈ 10% of N; yields a tight and complete distribution.
- *t_{dom}* >> 20% of N; cause the algorithm to prematurely converge to a small portion of the front. Alternative tradeoffs were never even found.
\end{itemize}

##  Fitness Sharing
Goldberg and Richardson defined a sharing function [3]. They describe the idea of fitness sharing in a GA as a way of promoting stable sub-population, or species. The focus of fitness sharing is to distribute the population in search space over a number of different peaks, which are possible Pareto-optimal solutions. So, fitness sharing helps the algorithm to maintain the population diversity. Goldberg and Richardson say that when the candidates are either both dominated or both non-dominated, it is likely that they are in the same equivalance class. We are interested in maintaining diversity along the front, and most of the individuals in these equivalence classes can be labeled “equally” fit, so, the “best fit” candidate is determined to be that candidate which has the least number of individuals in its niche. If we wish to maintain useful diversity on population, it is apparent that it would be best to choose the candidate that has the smaller **niche count** *m_i*. The competitor with lowest niche count won the tournament.

[1] N. Nafploitis J. Horn and D. E. Goldberg.  A niched pareto genetic algorithmfor multiobjective optimization.Proceedings of the First IEEE Conference on Evolu-tionary Computation. Z. Michalewicz, Ed. Piscataway, NJ: IEEE Press, page82–87,1994

[2] N. Nafploitis J. Horn.  Multiobjective optimization using the niche pareto ge-netic algorithm.IlliGAL Report No.93005. Illinois Genetic Algorithm Laboratory.University of Illinois at Urbana-Champaign,1993.

[3] J.Richardson D.E.Goldberg.   Genetic algorithms with sharing for multimodalfunction  optimization.In:  Proceedings  of  the  second  international  conference  ongenetic algorithms, Lawrence Erlbaum Associates, Hillsdale, NJ, pages41–49,1987.

### Installation
When NumPy has been installed, NPGA can be installed using pip as follows:
```bash
 pip3 install git+https://github.com/EmilioSchi/Niched-Pareto-Genetic-Algorithm-NPGA
```

### Importing
```python
import NPGA
```
### Define Fitness calculation function
```python
 def getfitness(candidate):
 	x = decode_chromosome_function(candidate)
 	return [[F1(x), 'maximize'], [F2(x), 'minimize'], ..., [Fn(x), 'minimize']]

 def fnGetFitness(genes): return getfitness(genes)
```

### Define display function over generation

```python
 def display(statistics):
 	print(statistic)
 	...

 def fnDisplay(statistic): display(statistic)
```

### Set parameters
```python
 geneset = '01'
 genelen = [64] # or genelen = [10, 12, 15] if there are more choromosome lenght
 optimalFitness = [0, 0]
 GA = NPGA.NichedParetoGeneticAlgorithm(
 	fnGetFitness, fnDisplay, optimalFitness,
 	geneset, genelen, population_size = 20,
 	max_generation = 30, crossover_rate = 0.7,
 	mutation_rate = 0.05, niche_radius = 0.2,
 	candidate_size = 2, prc_tournament_size = 0.2)
```
### Run
```python
best, fitness = GA.Evolution()
```
