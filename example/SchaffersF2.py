import math
import NPGA
import matplotlib.pyplot as plt
import numpy as np

def scaleMinMax(x, xmin, xmax, mindesired, maxdesired):
	return ((x - xmin) / (xmax - xmin) * (maxdesired - mindesired) + mindesired)

def graytodec(bin_list):
	"""
	Convert from Gray coding to binary coding.
	We assume big endian encoding.
	"""
	b = bin_list[0]
	d = int(b) * (2**(len(bin_list)-1))
	for i, e in enumerate(range(len(bin_list) - 2, -1, -1)):
		b = str(int(b != bin_list[i + 1]))
		d += int(b) * (2**e)
	return d

def decodechromosome(bits):
	dec = graytodec(bits)
	max_current = math.pow(2, len(bits)) - 1
	value = scaleMinMax(dec, 0, max_current, -6, 6)
	return value

class StaticGen:
	Generation = 1

def display(statistics):
	xpop = []
	ypop = []
	#print(len(statistics.population))

	for candidate in statistics.population:
		xpop.append(decodechromosome(candidate.Genes))
		ypop.append(candidate.Fitness)

	xbest = []
	ybest = []
	for specie in statistics.Species:
		xbest.append(decodechromosome(specie.Genes))
		ybest.append(specie.Fitness)

	xEUbest = [decodechromosome(statistics.EuclideanBetter.Genes)]
	yEUbest = [statistics.EuclideanBetter.Fitness]

	x = np.linspace(-6,6,100)
	y21 = [f21(i) for i in x if True]
	y22 = [f22(i) for i in x if True]

	plt.figure(1, figsize=(10,4))
	plt.clf()

	plt.subplot(1, 2, 1)
	plt.axis([-6, 6, 0, 40])

	plt.plot(x, y21, 'k')
	plt.plot(x, y22, 'k')
	plt.plot(xpop, ypop, 'ko')
	plt.plot(xbest, ybest, 'go')
	plt.plot(xEUbest, yEUbest, 'ro')

	plt.title('Schaffer\'s function F2, GENERATION: ' + str(StaticGen.Generation))
	plt.text(4.2, 28, 'f21()')
	plt.text(4.5, 13, 'f22()')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.grid()


	f1x = []
	f2x = []
	for point in statistics.ParetoSet:
		f1x.append(point.Fitness[0])
		f2x.append(point.Fitness[1])

	plt.subplot(1, 2, 2)
	plt.axis([0, 4.5, 0, 4.5])

	plt.plot(f1x, f2x, 'ko')
	plt.title('Pareto Front')
	plt.xlabel('F21(x)')
	plt.ylabel('F22(x)')
	plt.grid()

	plt.draw()
	plt.pause(0.1)
	plt.show(block=False)

	print(statistics.EuclideanBetter.Genes, end='\t')
	print(statistics.EuclideanBetter.Fitness)

	StaticGen.Generation = StaticGen.Generation + 1

def f21(x):
	return x * x

def f22(x):
	return (x - 2) * (x - 2)

def getfitness(candidate):
	x = decodechromosome(candidate)
	return [[f21(x), 'minimize'], [f22(x), 'minimize']]

def test():
	geneset = '01'
	genelen = [128]

	def fnDisplay(statistics): display(statistics)
	def fnGetFitness(genes): return getfitness(genes)

	optimalFitness = [0, 0]

	GA = NPGA.NichedParetoGeneticAlgorithm(
							fnGetFitness, fnDisplay, optimalFitness,
							geneset, genelen, population_size = 30,
							max_generation = 100, crossover_rate = 0.65,
							mutation_rate = 1/128, niche_radius = 0.08,
							candidate_size = 2, prc_tournament_size = 0.2,
							fastmode = True, multithreadmode = True)

	paretosolution = GA.Evolution()

test()
plt.show()
