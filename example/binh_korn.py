import npga as ga
import matplotlib.pyplot as plt

def display(statistics):
	f1x = [obj.fitness[0] for obj in statistics.pareto_front]
	f2x = [obj.fitness[1] for obj in statistics.pareto_front]

	xpop = [obj.fitness[0] for obj in statistics.current_population]
	ypop = [obj.fitness[1] for obj in statistics.current_population]

	plt.figure(1); plt.clf()
	plt.title('Binh and Korn')
	plt.xlabel('F1(x)'); plt.ylabel('F2(x)')
	plt.plot(xpop, ypop, 'ko', label='individuals')
	plt.plot(f1x, f2x, 'ro', label='pareto front')
	plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), ncol=2)
	plt.axis([0, 140, 0, 50]); plt.grid()
	plt.draw(); plt.pause(0.00001); plt.show(block=False)

def scale(x, xmin, xmax, mindesired, maxdesired):
	return ((x - xmin) / (xmax - xmin) * (maxdesired - mindesired) + mindesired)

def binary_to_decimal(arr):
    return sum(int(arr[i]) * (2**i) for i in range(len(arr)))

def binh_korn(genes):
    x = scale(binary_to_decimal(genes[0:32]), 0, (2**32)-1, 0, 5)
    y = scale(binary_to_decimal(genes[32:64]), 0, (2**32)-1, 0, 5)

    f1 = 4 * x ** 2 + 4 * y ** 2
    f2 = (x - 5) ** 2 + (y - 5) ** 2

    return [[f1, 'minimize'], [f2, 'minimize']]

gene_set = '01'
gene_len = [64]

algorithm = ga.Algorithm(binh_korn, [0, 0], 
                gene_len,
                chromosome_set = gene_set,
                display_function = display,
                multithread_mode = True,
                max_generation = 100)
algorithm.run()

plt.show()