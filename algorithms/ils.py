import random
import time
from typing import Dict

import numpy as np

from algorithms.local_search import LocalSearch, LSStrategy
from algorithms.random import random_hamiltonian
from algorithms.utils import calculate_path_cost

from copy import deepcopy


class ILS:
    def __init__(self):
        """For simplicity I hard coded the parameters of the MLS algorithm."""
        self.strategy = LSStrategy.STEEPEST
        self.exchange_nodes = False

        # TODO: THIS MAX TIME SHOULD BE SET TO 200 * average time obtained for MSLS
        self.max_time = 200*7

    def perform_local_search(self, starting_solution: np.ndarray) -> Dict:
        start_time = time.perf_counter()
        local_search = LocalSearch(self.strategy, self.exchange_nodes)
        solution = local_search(self.adj_matrix, self.nodes_cost, starting_solution)

        return {
            "solution": solution,
            "cost": calculate_path_cost(solution, self.adj_matrix, self.nodes_cost),
            "time": time.perf_counter() - start_time,
        }

    def get_n_perturbations(self) -> int:
        # TODO try to adjust this if have time
        if self.iterations < 100:
            return 10
        elif self.iterations < 300:
            return 20
        elif self.iterations < 500:
            return 25
        elif self.iterations < 700:
            return 30
        else:
            return 40

    def get_min_max_nodes_permuted(self) -> (int, int):
        # TODO try to adjust this if have time
        if self.iterations < 100:
            return (5, 20)
        elif self.iterations < 300:
            return (5, 30)
        elif self.iterations < 500:
            return (10, 30)
        elif self.iterations < 700:
            return (10, 40)
        else:
            return (20, 50)

    def perturb(self, solution: np.ndarray) -> np.ndarray:
        local_search = LocalSearch(self.strategy, self.exchange_nodes)

        if random.random() < 0.5:
            n_perm = self.get_n_perturbations()
            # print(n_perm)
            for _ in range(n_perm):
                if random.random() < 0.5:
                    i = random.randint(0, self.n_in_solution - 1)
                    j = random.randint(0, self.n_in_solution - 1)
                    solution = local_search.two_nodes_exchange(i, j, solution)
                else:
                    i, j = random.sample(range(self.n_in_solution), 2)
                    solution = local_search.two_edges_exchange(i, j, solution)
        else:
            MIN_NODES_PERMUTED, MAX_NODES_PERMUTED = self.get_min_max_nodes_permuted()
            # print(MIN_NODES_PERMUTED, MAX_NODES_PERMUTED)
            i = random.randint(0, self.n_in_solution - 1 - MAX_NODES_PERMUTED)
            j = random.randint(i + MIN_NODES_PERMUTED, i + MAX_NODES_PERMUTED - 1)
            to_shuffle = solution[i:j].copy()
            np.random.shuffle(to_shuffle)
            solution[i:j] = to_shuffle

        return solution

    def __call__(self, adj_matrix: np.ndarray, nodes_cost: np.ndarray) -> Dict:
        print("call")
        self.adj_matrix = adj_matrix
        self.nodes_cost = nodes_cost

        total_time = 0
        self.iterations = 0
        i_for_best_solution = -1
        best_solution = random_hamiltonian(self.adj_matrix, self.nodes_cost)
        best_cost = calculate_path_cost(best_solution, self.adj_matrix, self.nodes_cost)

        self.n_not_in_solution = self.adj_matrix.shape[0] - len(best_solution)
        self.n_in_solution = len(best_solution)

        old_cost = best_cost
        while total_time < self.max_time:
            new_solution = self.perform_local_search(self.perturb(best_solution))
            self.iterations += 1
            total_time += new_solution["time"]

            # TODO You must balance between how many times you come back to the solution and how many times you improve it using size of changes
            # if new_solution["cost"] == old_cost:
                # print("Returned to the previous solution!")
            old_cost = new_solution["cost"]

            if new_solution["cost"] < best_cost:
                # print("improvement")
                best_cost = deepcopy(new_solution["cost"])
                best_solution = deepcopy(new_solution["solution"])
                print(best_solution, best_cost, calculate_path_cost(best_solution, self.adj_matrix, self.nodes_cost))
                i_for_best_solution = self.iterations

        return {
            "solution": best_solution,
            "cost": best_cost,
            "n_iterations": self.iterations,
            "best_found_at": i_for_best_solution,
        }
