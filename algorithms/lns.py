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

    def destroy(self, solution, nodes_cost):
        """
        Destroy operator: removes a fraction of nodes/edges from the solution.
        """
        num_elements_to_remove = int(len(solution) * self.destroy_rate)
        weights = nodes_cost[solution]
        weights = weights / np.sum(weights)
        indices_to_remove = set(
            np.random.choice(
                range(len(solution)), num_elements_to_remove, replace=False, p=weights
            )
        )

        return [
            element for i, element in enumerate(solution) if i not in indices_to_remove
        ]

    def repair(self, partial_solution, adj_matrix, nodes_cost):
        """
        Repair operator: rebuilds the solution, potentially finding a better one.
        Use best greedy heuristic here.
        """
        greedy = Greedy2Regret(alpha=0.5)
        repaired_solution = greedy(
            adj_matrix, nodes_cost, starting_solution=partial_solution
        )

        return repaired_solution

    def __call__(self, adj_matrix: np.ndarray, nodes_cost: np.ndarray) -> Dict:
        print("call")
        self.adj_matrix = adj_matrix
        self.nodes_cost = nodes_cost

        local_search = LocalSearch(self.strategy, self.exchange_nodes)
        # initialize solution and perform local search
        best_solution = local_search(
            self.adj_matrix, self.nodes_cost, random_hamiltonian(adj_matrix, nodes_cost)
        )
        best_cost = calculate_path_cost(best_solution, self.adj_matrix, self.nodes_cost)

        start_time = time.perf_counter()
        n_iterations = 0
        total_time = 0
        while total_time < self.max_time:
            # Start from copy of the best solution
            new_solution = deepcopy(best_solution)

            # Destroy it
            new_solution = self.destroy(new_solution, nodes_cost)

            # repair with weighted 2 regert
            new_solution = self.repair(new_solution, self.adj_matrix, self.nodes_cost)

            # Apply LS if needed
            if self.do_LS:
                local_search = LocalSearch(self.strategy, self.exchange_nodes)
                new_solution = local_search(
                    self.adj_matrix, self.nodes_cost, new_solution
                )

            new_cost = calculate_path_cost(
                new_solution, self.adj_matrix, self.nodes_cost
            )

            # If we observe improvement, update best solution
            if best_cost > new_cost:
                best_solution = new_solution
                best_cost = new_cost

            n_iterations += 1
            total_time = time.perf_counter() - start_time

        return {
            "solution": best_solution,
            "cost": best_cost,
            "n_iterations": n_iterations,
        }
