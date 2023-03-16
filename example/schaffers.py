import npga as ga
import matplotlib.pyplot as plt
import numpy as np
import math

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

def display(statistics):
	print(statistics)

	xpop = [obj.fitness[0] for obj in statistics.current_population]
	ypop = [obj.fitness[1] for obj in statistics.current_population]

	f1x = [obj.fitness[0] for obj in statistics.pareto_front]
	f2x = [obj.fitness[1] for obj in statistics.pareto_front]

	plt.figure(1); plt.clf()
	plt.title('Schaffer')
	plt.xlabel('F21(x)'); plt.ylabel('F22(x)')
	plt.plot(xpop, ypop, 'ko', label='individuals')
	plt.plot(f1x, f2x, 'ro', label='pareto front')
	plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), ncol=2)
	plt.axis([0, 5, 0, 5]); plt.grid()
	plt.draw(); plt.pause(0.00001); plt.show(block=False)

def f21(x):
	return x * x

def f22(x):
	return (x - 2) * (x - 2)

def getfitness(candidate):
	x = decodechromosome(candidate)
	return [[f21(x), 'minimize'], [f22(x), 'minimize']]

geneset = '01'
genelen = [128]
optimalFitness = [0, 0]

algorithm = ga.Algorithm(getfitness, [0, 0], 
							genelen,
							chromosome_set = geneset,
							display_function = display,
							multithread_mode = True,
							max_generation = 100)
algorithm.run()

plt.show()
