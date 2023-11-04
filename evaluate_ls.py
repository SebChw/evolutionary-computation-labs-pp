from data.data_parser import get_data
from algorithms.random import random_hamiltonian
from algorithms.local_search import LocalSearch
from algorithms.greedy_2_regret import Greedy2Regret
from algorithms.utils import Solution, calculate_path_cost
from copy import copy
from collections import defaultdict
from dataclasses import asdict
import json

from joblib import Parallel, delayed

from copy import deepcopy

import numpy as np

np.random.seed(42)


data = get_data()
results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

for problem, instance in data.items():
    distance_matrix = instance["dist_matrix"]
    nodes_cost = instance["nodes_cost"]
    for i in range(200):
        random_solution = random_hamiltonian(distance_matrix, nodes_cost)
        greedy_solution = Greedy2Regret(alpha=0.5)(
            distance_matrix, nodes_cost, starting_node=i)
        solutions = [random_solution, greedy_solution]

        for greedy in [True, False]:
            for exchange_nodes in [True, False]:
                for starting_solution in solutions:
                    result = LocalSearch(greedy=greedy, exchange_nodes=exchange_nodes)(
                        distance_matrix, nodes_cost, copy(starting_solution)
                    )
                    results[problem][greedy][exchange_nodes].append(
                        asdict(Solution(result, calculate_path_cost(
                            result, distance_matrix, nodes_cost)))
                    )

    with open("solutions.json", "w") as file:
        json.dump(dict(solutions), file, indent=4)
        print("Results have been saved to solutions.json")
