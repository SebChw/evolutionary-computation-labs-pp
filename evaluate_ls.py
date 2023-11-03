import numpy as np

from algorithms.local_search import LocalSearch
from algorithms.random import random_hamiltonian
from data.data_parser import get_data

data = get_data()
np.random.seed(42)
for problem, instance in data.items():
    distance_matrix = instance["dist_matrix"]
    nodes_cost = instance["nodes_cost"]

    local_search = LocalSearch(greedy=True, exchange_nodes=True)
    initial_solution = random_hamiltonian(distance_matrix, nodes_cost, 0)
    solution = local_search(distance_matrix, nodes_cost, initial_solution)
