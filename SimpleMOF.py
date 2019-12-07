import math
import NPGA
import matplotlib.pyplot as plt
import numpy as np



def scaleMinMax(x, xmin, xmax, mindesired, maxdesired):
	x = (x - xmin) / (xmax - xmin) * (maxdesired - mindesired) + mindesired
	return x

def graycode(n):
	# Return n-bit Gray code in a list
	if n == 0:
		return ['']
	first_half = graycode(n - 1)
	second_half = first_half.copy()
	first_half = ['0' + code for code in first_half]
	second_half = ['1' + code for code in reversed(second_half)]
	return first_half + second_half

class StaticCode:
	g = []

def graytodec(bits):
	return int(StaticCode.g.index(''.join(bits)))

def decodechromosome(bits):
	dec = graytodec(bits)
	max_current = math.pow(2, len(bits)) - 1
	value = scaleMinMax(dec, 0, max_current, -10, 10)
	return value

class StaticGen:
	Generation = 1

def display(candidates, statistics):
	xpop = []
	ypop = []
	for candidate in candidates:
		xpop.append(decodechromosome(candidate.Genes))
		ypop.append(candidate.Fitness)

	xbest = []
	ybest = []
	for candidate in statistics.best:
		xbest.append(decodechromosome(candidate['Genes']))
		ybest.append(candidate['Fitness'])

	xEUbest = [decodechromosome(statistics.EuclideanBetter['Genes'])]
	yEUbest = [statistics.EuclideanBetter['Fitness']]

	x = np.linspace(-10,10,100)
	y21 = [F1(i) for i in x if True]
	y22 = [F2(i) for i in x if True]

	plt.figure(1)
	plt.clf()
	plt.axis([-10, 10, -20, 180])
	#plt.legend(['Generation'], loc=1)
	plt.plot(x, y21, 'k')
	plt.plot(x, y22, 'k')
	plt.plot(xpop, ypop, 'ko')
	plt.plot(xbest, ybest, 'go')
	plt.plot(xEUbest, yEUbest, 'ro')
	plt.title('Simple MO problem, GENERATION: ' + str(StaticGen.Generation))
	#plt.text(4.3, 28, 'F1()')
	#plt.text(4.6, 13, 'F2()')
	plt.grid()
	plt.draw()
	plt.pause(0.4)
	plt.show(block=False)

	print(statistics.EuclideanBetter['Genes'], end='\t')
	print(statistics.EuclideanBetter['Fitness'], end='\t')
	print(statistics.EuclideanBetter['Distance'], end='\n')

	StaticGen.Generation = StaticGen.Generation + 1

def F1(x):
	return (x + 2) * (x + 2) - 10

def F2(x):
	return (x - 2) * (x - 2) +20

def getfitness(candidate):
	x = decodechromosome(candidate)
	return [F1(x), F2(x)]

def test():
	geneset = '01'
	genelen = [20]
	StaticCode.g = graycode(20)

	def fnDisplay(candidate, statistic): display(candidate, statistic)
	def fnGetFitness(genes): return getfitness(genes)

	optimalFitness = [0, 0]

	GA = NPGA.NichedParetoGeneticAlgorithm(
							fnGetFitness, fnDisplay, optimalFitness,
							geneset, genelen, population_size = 20,
							max_generation = 30, crossover_rate = 0.7,
							mutation_rate = 0.05, niche_radius = 0.3,
							candidate_size = 2, prc_tournament_size = 0.2)
	best, fitness = GA.Evolution()

	return best

print(test())
plt.show()
