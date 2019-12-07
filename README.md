# Niched-Pareto-Genetic-Algorithm-NPGA
Genetic Algorithm (GA) for a  Multi-objective Optimization Problem (MOP)

To maintain multiple Pareto optimal solutions, Horn et all [1] have altered tournament selection. NPGA uses a tournament selection scheme based on Pareto dominance. Many (conventionally 2 candidates at once) individuals randomly chosen are compared against a subset from the entire population. When both competitors are either dominated or non dominated, the result of the tournament is decided through fitness sharing in the objective domain.

[1] N. Nafploitis J. Horn and D. E. Goldberg.  A niched pareto genetic algorithmfor multiobjective optimization.Proceedings of the First IEEE Conference on Evolu-tionary Computation. Z. Michalewicz, Ed. Piscataway, NJ: IEEE Press, page82–87,1994
