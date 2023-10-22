import argparse
import json
from collections import defaultdict
from dataclasses import asdict

from joblib import Parallel, delayed

from algorithms.greedy_2_regret import Greedy2Regret
from algorithms.greedy_cycle import greedy_cycle
from algorithms.nn import nearest_neighbor_hamiltonian
from algorithms.random import random_hamiltonian
from algorithms.utils import Solution, calculate_path_cost
from data.data_parser import get_data

ALGORITHMS = {
    "random": random_hamiltonian,
    "nn": nearest_neighbor_hamiltonian,
    "greedy": greedy_cycle,
    "2_regret": Greedy2Regret(),
    "W_2_regret": Greedy2Regret(alpha=0.5),
}


def main():
    parser = argparse.ArgumentParser(description="Run Hamiltonian path algorithms.")
    for algorithm_name in ALGORITHMS.keys():
        parser.add_argument(
            f"--{algorithm_name}",
            type=int,
            help=f"Number of runs for {algorithm_name} algorithm",
        )

    args = parser.parse_args()

    data = get_data()

    solutions = defaultdict(lambda: defaultdict(list))

    for algorithm_name, algorithm_function in ALGORITHMS.items():
        runs = getattr(args, algorithm_name)
        if runs:
            for problem, instance in data.items():
                print(f"Running {algorithm_name} on {problem}...")
                distance_matrix = instance["dist_matrix"]
                nodes_cost = instance["nodes_cost"]

                parallel = Parallel(n_jobs=-1)
                all_nodes = parallel(
                    delayed(algorithm_function)(distance_matrix, nodes_cost, i)
                    for i in range(runs)
                )

                for nodes in all_nodes:
                    solution = Solution(
                        nodes=nodes,
                        cost=calculate_path_cost(nodes, distance_matrix, nodes_cost),
                    )
                    solutions[problem][algorithm_name].append(asdict(solution))

    with open("solutions.json", "w") as file:
        json.dump(dict(solutions), file, indent=4)

    print("Results have been saved to solutions.json")


if __name__ == "__main__":
    main()
