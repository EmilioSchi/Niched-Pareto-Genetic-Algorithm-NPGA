<img align="left" height="100" src="img/logo.png">
# Niched Pareto Genetic Algorithm
Genetic Algorithm (GA) for a  Multi-objective Optimization Problem (MOP)

## Introduction
To maintain multiple Pareto optimal solutions, Horn et all [1] have altered tournament selection. NPGA uses a tournament selection scheme based on Pareto dominance. Many (conventionally 2 candidates at once) individuals randomly chosen are compared against a subset from the entire population. When both competitors are either dominated or non dominated, the result of the tournament is decided through fitness sharing in the objective domain.

[1] N. Nafploitis J. Horn and D. E. Goldberg.  A niched pareto genetic algorithmfor multiobjective optimization.Proceedings of the First IEEE Conference on Evolu-tionary Computation. Z. Michalewicz, Ed. Piscataway, NJ: IEEE Press, page82–87,1994

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
	return [f1(x), f2(x), f3(x), ..., fn(x)]

def fnGetFitness(genes): return getfitness(genes)
```

### Define display function over generation

```python
def display(candidates, statistics):
	...

def fnDisplay(candidate, statistic): display(candidate, statistic)
```

### Set parameters
```python
geneset = '01'
genelen = [64]
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
