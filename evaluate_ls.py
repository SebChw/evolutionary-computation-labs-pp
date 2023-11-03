from copy import copy

import numpy as np

np.random.seed(42)
from algorithms.local_search import LocalSearch
from algorithms.random import random_hamiltonian
from data.data_parser import get_data

data = get_data()

for problem, instance in data.items():
    distance_matrix = instance["dist_matrix"]
    nodes_cost = instance["nodes_cost"]
    # TODO implement start from best solution found by greedy
    initial_solution = random_hamiltonian(distance_matrix, nodes_cost, 0)
    # Greedy, exchange nodes
    LocalSearch(greedy=True, exchange_nodes=True)(
        distance_matrix, nodes_cost, copy(initial_solution)
    )

    # Greedy, exchange edges
    LocalSearch(greedy=True, exchange_nodes=False)(
        distance_matrix, nodes_cost, copy(initial_solution)
    )

    # Steepest, exchange nodes
    LocalSearch(greedy=False, exchange_nodes=True)(
        distance_matrix, nodes_cost, copy(initial_solution)
    )

    # Steepest, exchange edges
    LocalSearch(greedy=False, exchange_nodes=False)(
        distance_matrix, nodes_cost, copy(initial_solution)
    )

    break
