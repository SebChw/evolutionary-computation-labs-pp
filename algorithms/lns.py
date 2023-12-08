import random
import time
from copy import deepcopy
from typing import Dict

import numpy as np

from algorithms.greedy_2_regret import Greedy2Regret
from algorithms.local_search import LocalSearch, LSStrategy
from algorithms.random import random_hamiltonian
from algorithms.utils import calculate_path_cost


class LNS:
    def __init__(self, destroy_rate=0.2, do_LS=False, max_time: int = 7 * 200):
        """
        Initialize the LNS algorithm parameters.
        """
        self.destroy_rate = destroy_rate
        self.strategy = LSStrategy.STEEPEST
        self.do_LS = do_LS
        self.max_time = max_time
        self.exchange_nodes = False

    def destroy(self, solution):
        """
        Destroy operator: removes a fraction of nodes/edges from the solution.
        """
        num_elements_to_remove = int(len(solution) * self.destroy_rate)
        indices_to_remove = set(
            random.sample(range(len(solution)), num_elements_to_remove)
        )
        return [
            element for i, element in enumerate(solution) if i not in indices_to_remove
        ]

    def repair(self, partial_solution, adj_matrix, nodes_cost):
        """
        Repair operator: rebuilds the solution, potentially finding a better one.
        Use best greedy heuristic here.
        """
        greedy = Greedy2Regret()
        repaired_solution = greedy(
            adj_matrix, nodes_cost, starting_solution=partial_solution
        )

        return repaired_solution

    def __call__(self, adj_matrix: np.ndarray, nodes_cost: np.ndarray) -> Dict:
        self.adj_matrix = adj_matrix
        self.nodes_cost = nodes_cost

        initial_solution = random_hamiltonian(adj_matrix, nodes_cost)

        start_time = time.perf_counter()
        new_solution = deepcopy(initial_solution)
        new_cost = calculate_path_cost(new_solution, self.adj_matrix, self.nodes_cost)
        n_iterations = 0
        total_time = 0
        while total_time < self.max_time:
            partial_solution = self.destroy(new_solution)
            new_solution = self.repair(
                partial_solution, self.adj_matrix, self.nodes_cost
            )
            new_cost = calculate_path_cost(
                new_solution, self.adj_matrix, self.nodes_cost
            )
            if self.do_LS:
                local_search = LocalSearch(self.strategy, self.exchange_nodes)
                solution_LS = local_search(
                    self.adj_matrix, self.nodes_cost, deepcopy(new_solution)
                )
                solution_LS_cost = calculate_path_cost(
                    solution_LS, self.adj_matrix, self.nodes_cost
                )
                if solution_LS_cost > new_cost:
                    new_solution = solution_LS
                    new_cost = solution_LS_cost
            n_iterations += 1
            total_time = time.perf_counter() - start_time

        return {
            "solution": new_solution,
            "cost": new_cost,
            "n_iterations": n_iterations,
        }
