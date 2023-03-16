import npga as ga

class StaticVariables:
	Generation = 0

def display(statistics):

	print(statistics)

	StaticVariables.Generation = StaticVariables.Generation + 1

def getfitness(candidate):
	password = "mypassword"
	lenght_error = abs(len(candidate) - len(password))

	character_error = 0
	for charPassword, charCandidate in zip(password, candidate):
		character_error = character_error + abs(int(ord(charPassword)) - int(ord(charCandidate)))
	character_error = character_error + lenght_error * 26

	return [[character_error, 'minimize'], [lenght_error, 'minimize']]

def test():
	geneset = 'abcdefghijklmnopqrstuvwxyz'
	gene_len = list(range(6, 13))

	optimalFitness = [0, 0]

	algorithm = ga.Algorithm(getfitness, [0, 0], 
					gene_len, chromosome_set = geneset,
					display_function = display,
					population_size = 200,
					max_generation = 1000, crossover_rate = 0.8,
					mutation_rate = 0.08, niche_radius = 1.5,
					length_mutation_rate = 0.05,
					candidate_size = 3, prc_tournament_size = 0.1)

	algorithm.run()

test()
