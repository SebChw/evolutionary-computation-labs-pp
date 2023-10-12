import argparse
import json
from dataclasses import asdict
from collections import defaultdict

from data.data_parser import get_data

from algorithms.random import random_hamiltonian
from algorithms.nn import nearest_neighbor_hamiltonian
from algorithms.greedy_cycle import greedy_cycle

from algorithms.utils import calculate_path_cost, Solution


ALGORITHMS = {
    'random': random_hamiltonian,
    'nn': nearest_neighbor_hamiltonian,
    'greedy': greedy_cycle
}


def main():
    parser = argparse.ArgumentParser(
        description="Run Hamiltonian path algorithms.")
    for algorithm_name in ALGORITHMS.keys():
        parser.add_argument(f'--{algorithm_name}', type=int,
                            help=f'Number of runs for {algorithm_name} algorithm')

    args = parser.parse_args()

    data = get_data()

    solutions = defaultdict(lambda: defaultdict(list))

    for algorithm_name, algorithm_function in ALGORITHMS.items():
        runs = getattr(args, algorithm_name)
        if runs:
            for problem, adj_matrix in data.items():
                for i in range(runs):
                    nodes = algorithm_function(adj_matrix, i)
                    solution = Solution(
                        nodes=nodes, cost=calculate_path_cost(nodes, adj_matrix))
                    solutions[problem][algorithm_name].append(asdict(solution))

    with open('solutions.json', 'w') as file:
        json.dump(dict(solutions), file, indent=4)

    print("Results have been saved to solutions.json")


if __name__ == '__main__':
    main()
