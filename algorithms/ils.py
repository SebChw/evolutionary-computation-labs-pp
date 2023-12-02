import random
import time
from typing import Dict

import numpy as np

from algorithms.local_search import LocalSearch, LSStrategy
from algorithms.random import random_hamiltonian
from algorithms.utils import calculate_path_cost


class ILS:
    def __init__(self):
        """For simplicity I hard coded the parameters of the MLS algorithm."""
        self.strategy = LSStrategy.STEEPEST
        self.exchange_nodes = False
        self.max_time = 30

    def perform_local_search(self, starting_solution: np.ndarray) -> Dict:
        start_time = time.perf_counter()
        local_search = LocalSearch(self.strategy, self.exchange_nodes)
        solution = local_search(self.adj_matrix, self.nodes_cost, starting_solution)

        return {
            "solution": solution,
            "cost": calculate_path_cost(solution, self.adj_matrix, self.nodes_cost),
            "time": time.perf_counter() - start_time,
        }

    def perturb(self, solution: np.ndarray) -> np.ndarray:
        local_search = LocalSearch(self.strategy, self.exchange_nodes)

        if random.random() < 0.5:
            for _ in range(5):
                if random.random() < 0.5:
                    i = random.randint(0, self.n_in_solution - 1)
                    j = random.randint(0, self.n_in_solution - 1)
                    solution = local_search.two_nodes_exchange(i, j, solution)
                else:
                    i, j = random.sample(range(self.n_in_solution), 2)
                    solution = local_search.two_edges_exchange(i, j, solution)
        else:
            MIN_NODES_PERMUTED = 5
            i = random.randint(0, self.n_in_solution - 1 - MIN_NODES_PERMUTED)
            j = random.randint(i + MIN_NODES_PERMUTED, self.n_in_solution - 1)
            to_shuffle = solution[i:j].copy()
            np.random.shuffle(to_shuffle)
            solution[i:j] = to_shuffle

        return solution

    def __call__(self, adj_matrix: np.ndarray, nodes_cost: np.ndarray) -> Dict:
        self.adj_matrix = adj_matrix
        self.nodes_cost = nodes_cost

        total_time = 0
        self.iterations = 0
        i_for_best_solution = -1
        new_solution = random_hamiltonian(self.adj_matrix, self.nodes_cost)

        self.n_not_in_solution = self.adj_matrix.shape[0] - len(new_solution)
        self.n_in_solution = len(new_solution)
        best_cost = calculate_path_cost(new_solution, self.adj_matrix, self.nodes_cost)
        best_solution = new_solution

        old_cost = best_cost
        while total_time < self.max_time:
            new_solution = self.perform_local_search(new_solution)
            self.iterations += 1
            total_time += new_solution["time"]

            if new_solution["cost"] == old_cost:
                print("Returned to the previous solution!")
            old_cost = new_solution["cost"]

            if new_solution["cost"] < best_cost:
                best_cost = new_solution["cost"]
                best_solution = new_solution["solution"]
                i_for_best_solution = self.iterations

            new_solution = self.perturb(new_solution["solution"])

        return {
            "solution": best_solution,
            "cost": best_cost,
            "n_iterations": self.iterations,
            "best_found_at": i_for_best_solution,
        }
