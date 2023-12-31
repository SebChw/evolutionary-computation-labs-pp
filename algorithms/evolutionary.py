import random
import time
from collections import namedtuple
from copy import deepcopy
from typing import Dict

import numpy as np

from algorithms.greedy_2_regret import Greedy2Regret
from algorithms.local_search import LocalSearch, LSStrategy
from algorithms.random import random_hamiltonian
from algorithms.utils import calculate_path_cost

TSPSolution = namedtuple("TSPSolution", ["solution", "cost"])
NOT_YET_FOUND = -1


class LNS:
    def __init__(self, elite_population_size=20, do_LS=True, max_time: int = 7 * 200):
        """
        Initialize the LNS algorithm parameters.
        """
        self.elite_population_size = elite_population_size
        self.do_LS = do_LS
        self.max_time = max_time

        self.ls_strategy = LSStrategy.STEEPEST
        self.ls_exchange_nodes = False
        self.local_search = LocalSearch(self.ls_strategy, self.ls_exchange_nodes)

        self.recombination_operators = [self.common_crossover, self.repair_crossover]

    def get_edges(self, solution):
        return [(solution[i], solution[i + 1]) for i in range(len(solution) - 1)] + [
            (solution[-1], solution[0])
        ]

    def common_crossover(self, parentA, parentB):
        # get common nodes
        common_nodes = set(parentA).intersection(set(parentB))

        # First create all separate common subpaths. It is enough to iterate over one parent - no matter which
        subpaths = []
        i = 0
        while i < len(parentA):
            if parentA[i] in common_nodes:
                subpath = [parentA[i]]
                i += 1
                while parentA[i] in common_nodes:
                    subpath.append(parentA[i])
                    i += 1
                if len(subpath) > 1:
                    subpaths.append(subpath)
            else:
                i += 1

        # Take care of the last connection
        if subpaths[-1][-1] == parentA[-1] and parentA[0] in common_nodes:
            # At first check if we can merge last and first subpath, if not append first node to last subpath
            if subpaths[0][0] == parentA[0]:
                subpaths[0] = subpaths.pop() + subpaths[0]
            else:
                subpaths[-1].append(parentA[0])

        n_nodes_in_subpaths = sum([len(subpath) for subpath in subpaths])

        offspring = []
        # Place subpaths in somehow random order. Total random would be very computationally expensive
        random.shuffle(subpaths)
        p = 1 / len(subpaths)
        indices_to_be_filled = []
        while len(subpaths) > 0:
            if len(parentA) - len(offspring) == n_nodes_in_subpaths:
                subpath = subpaths.pop()
                offspring.extend(subpath)
                n_nodes_in_subpaths -= len(subpath)
            elif random.random() < p:
                subpath = subpaths.pop()
                offspring.extend(subpath)
                n_nodes_in_subpaths -= len(subpath)
            else:
                # Special places to which we will later add nodes
                offspring.append(NOT_YET_FOUND)
                indices_to_be_filled.append(len(offspring) - 1)

        # At this point length should math
        assert len(offspring) == len(parentA)

        used_nodes = set()
        for path in subpaths:
            used_nodes.update(path)

        # We cannot use node that consitutes a common edge as we will have a duplicate
        common_nodes_to_use = list(common_nodes.difference(used_nodes))
        n_possible_nodes = self.adj_matrix.shape[0]
        uncommon_nodes = list(set(range(n_possible_nodes)).difference(common_nodes))
        random.shuffle(common_nodes_to_use)

        while len(indices_to_be_filled) > 0:
            index = indices_to_be_filled.pop()

            # Find a node that is not yet in the offspring
            if len(common_nodes_to_use) > 0 and (
                random.random() < 0.5
                or len(indices_to_be_filled) - 1 == len(common_nodes_to_use)
            ):
                offspring[index] = common_nodes_to_use.pop()
            else:
                offspring[index] = uncommon_nodes.pop()

    def generate_initial_population(self):
        self.population = []
        self.population_costs = set()

        while len(self.population) < self.elite_population_size:
            solution = self.local_search(
                self.adj_matrix,
                self.nodes_cost,
                random_hamiltonian(self.adj_matrix, self.nodes_cost),
            )
            cost = calculate_path_cost(solution, self.adj_matrix, self.nodes_cost)
            if cost not in self.population_costs:
                self.population_costs.add(cost)
                solution = TSPSolution(solution=solution, cost=cost)
                self.population.append(solution)

    def __call__(self, adj_matrix: np.ndarray, nodes_cost: np.ndarray) -> Dict:
        self.adj_matrix = adj_matrix
        self.nodes_cost = nodes_cost

        population = self.generate_initial_population()

        start_time = time.perf_counter()
        n_iterations = 0
        while time.perf_counter() - start_time < self.max_time:
            # Select parents
            parentA, parentB = np.random.choice(self.population, size=2, replace=False)

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

        return {
            "solution": best_solution,
            "cost": best_cost,
            "n_iterations": n_iterations,
        }
