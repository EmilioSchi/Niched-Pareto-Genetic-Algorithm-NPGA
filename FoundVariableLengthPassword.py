import NPGA
import numpy as np

class StaticVariables:
	Generation = 0

def display(candidates, statistics):

	print(statistics.EuclideanBetter['Genes'], end='\t')
	print(statistics.EuclideanBetter['Fitness'], end='\t')
	print(statistics.EuclideanBetter['Distance'], end='\n')

	StaticVariables.Generation = StaticVariables.Generation + 1

def getfitness(candidate):
	password = "mypassword"
	#candidate = ''.join(candidate)
	lenght_error = abs(len(candidate) - len(password))

	character_error = 0
	for charPassword, charCandidate in zip(password, candidate):
		character_error = character_error + abs(int(ord(charPassword)) - int(ord(charCandidate)))
	character_error = character_error + lenght_error * 26

	return [character_error, lenght_error]

def test():
	geneset = 'abcdefghijklmnopqrstuvwxyz'
	genelen = list(range(6, 13))

	def fnDisplay(candidate, statistic):
		display(candidate, statistic)

	def fnGetFitness(genes):
		return getfitness(genes)

	optimalFitness = [0, 0]

	GA = NPGA.NichedParetoGeneticAlgorithm(
							fnGetFitness, fnDisplay, optimalFitness,
							geneset, genelen, population_size = 100,
							max_generation = 1000, crossover_rate = 0.7,
							mutation_rate = 0.04, niche_radius = 2,
							length_mutation_rate = 0.05, fastmode = True,
							candidate_size = 4, prc_tournament_size = 0.1)
	best, fitness = GA.Evolution()

	return best

print(test())
