import time
from typing import Dict

import numpy as np
from joblib import Parallel, delayed

from algorithms.local_search import LocalSearch, LSStrategy
from algorithms.random_solution import random_hamiltonian
from algorithms.utils import calculate_path_cost


class MSLS:
    def __init__(self):
        """For simplicity I hard coded the parameters of the MLS algorithm."""
        self.strategy = LSStrategy.STEEPEST
        self.exchange_nodes = False
        self.n_iterations = 200

    def perform_local_search(
        self,
    ) -> Dict:
        start_time = time.perf_counter()
        local_search = LocalSearch(self.strategy, self.exchange_nodes)
        starting_solution = random_hamiltonian(self.adj_matrix, self.nodes_cost)
        solution = local_search(self.adj_matrix, self.nodes_cost, starting_solution)

        return {
            "solution": solution,
            "cost": calculate_path_cost(solution, self.adj_matrix, self.nodes_cost),
            "time": time.perf_counter() - start_time,
        }

    def __call__(self, adj_matrix: np.ndarray, nodes_cost: np.ndarray) -> Dict:
        self.adj_matrix = adj_matrix
        self.nodes_cost = nodes_cost

        tasks = []
        for _ in range(self.n_iterations):
            tasks.append(delayed(self.perform_local_search)())

        n_jobs = -1  # Use all available cores
        parallel_results = Parallel(n_jobs=n_jobs)(tasks)

        best_cost = np.inf
        best_solution = None
        times = []
        for result in parallel_results:
            curr_cost = result["cost"]
            if curr_cost < best_cost:
                best_cost = curr_cost
                best_solution = result["solution"]
            times.append(result["time"])

        return {
            "solution": best_solution,
            "cost": best_cost,
            "time": np.mean(times),
        }
