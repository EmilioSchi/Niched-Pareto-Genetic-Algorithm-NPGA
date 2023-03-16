import math
import npga as ga
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

def decodechromosome(bits, BitsForEachNumber, vector_size):
	x = np.zeros((vector_size,), dtype = np.float64)
	for i in range(vector_size):
		dec = graytodec(bits[(i * BitsForEachNumber) : (i * BitsForEachNumber + BitsForEachNumber)])
		max_current = math.pow(2, BitsForEachNumber) - 1
		x[i] = scaleMinMax(dec, 0, max_current, 0, 1)
	return x

def ZDT1(x):
	f1 = x[0]
	g = 1 + 9 * (np.sum(x[1:]) / (len(x)- 1))
	f2 = g * (1 - np.sqrt(x[0] / g))
	return f1, f2

def getfitness(candidate, BitsForEachNumber, vector_size):
	x = decodechromosome(candidate, BitsForEachNumber, vector_size)
	F1, F2 = ZDT1(x)
	return [[F1, 'minimize'], [F2, 'minimize']]

def display(statistics):
	f1x = [obj.fitness[0] for obj in statistics.pareto_front]
	f2x = [obj.fitness[1] for obj in statistics.pareto_front]

	xpop = [obj.fitness[0] for obj in statistics.current_population]
	ypop = [obj.fitness[1] for obj in statistics.current_population]

	plt.figure(1); plt.clf()
	plt.title('Zitzler-Deb-Thiele\'s function 1 ')
	plt.xlabel('F1(x)'); plt.ylabel('F2(x)')
	plt.plot(xpop, ypop, 'ko', label='individuals')
	plt.plot(f1x, f2x, 'ro', label='pareto front')
	plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), ncol=2)
	plt.axis([0, 1, -1, 3]); plt.grid()
	plt.draw(); plt.pause(0.00001); plt.show(block=False)

gene_set = '01'
bits_foreach_number = 16
vector_size = 30
gene_len = [bits_foreach_number * vector_size]

def fnGetFitness(genes): return getfitness(genes, bits_foreach_number, vector_size)

algorithm = ga.Algorithm(fnGetFitness, [0, 0], 
                gene_len,
                chromosome_set = gene_set,
                display_function = display,
                population_size = 200,
                max_generation = 4000, crossover_rate = 0.65,
                mutation_rate = 1/170, niche_radius = 0.02,
                candidate_size = 4, prc_tournament_size = 0.13,
                multithread_mode = True)

algorithm.run()

plt.show()