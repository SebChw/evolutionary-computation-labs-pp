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
    for algo_name in ALGORITHMS.keys():
        parser.add_argument(f'--{algo_name}', type=int,
                            help=f'Number of runs for {algo_name} algorithm')

    args = parser.parse_args()

    data = get_data()

    solutions = defaultdict(lambda: defaultdict(list))

    for algo_name, algo_func in ALGORITHMS.items():
        runs = getattr(args, algo_name)
        if runs:
            for problem, adj_matrix in data.items():
                for _ in range(runs):
                    nodes = algo_func(adj_matrix)
                    solution = Solution(
                        nodes=nodes, cost=calculate_path_cost(nodes, adj_matrix))
                    solutions[problem][algo_name].append(asdict(solution))

    with open('solutions.json', 'w') as file:
        json.dump(dict(solutions), file, indent=4)

    print("Results have been saved to solutions.json")


if __name__ == '__main__':
    main()
