from setuptools import setup
from NPGA import __version__

setup(
	author = 'Emilio Schinina',
	maintainer = 'Emilio Schinina',
	author_email = 'emilioschi@gmail.com',
	description = 'Multi-Objective Genetic Algorithm',
	long_description = '''
	An implementation of Niched Pareto Genetic Algorith, a Multi-Objective
	Genetic Algorithm invented by N. Nafploitis, J. Horn and D. E. Goldberg.
	''',
	license = 'Apache',
	url = 'https://github.com/EmilioSchi/Niched-Pareto-Genetic-Algorithm-NPGA',
	name = 'NPGA',
	version = __version__,
	packages=['NPGA'],
	install_requires=['numpy'],
	classifiers=("Programming Language :: Python :: 3"),
)
