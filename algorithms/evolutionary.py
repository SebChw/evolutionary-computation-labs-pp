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
        n_possible_nodes = self.adj_matrix.shape[0]

        # get common edges
        edgesA = self.get_edges(parentA)
        edgesB = self.get_edges(parentB)

        common_edges = list(set(edgesA).intersection(set(edgesB)))
        # Incoming and outcoming map will be used to reconstruct paths
        incoming_map = {edge[0]: edge[1] for edge in common_edges}

        #! First create all separate subpaths
        used_edges = set()
        subpaths = []
        for edge in common_edges[1:]:
            if edge not in used_edges:
                subpath = [edge[0], edge[1]]
                used_edges.add(edge)
                while subpath[-1] in incoming_map:
                    # we have to reconstruct the path
                    subpath.append(incoming_map[subpath[-1]])
                    edge = (subpath[-2], subpath[-1])
                    used_edges.add(edge)

                subpaths.append(subpath)

        #! Now we have to connect subpaths into offspring
        used_nodes = set()
        for edge in common_edges:
            used_nodes.add(edge[0])
            used_nodes.add(edge[1])

        # We cannot use node that consitutes a common edge as we will have a duplicate
        common_nodes_to_use = list(common_nodes.difference(used_nodes))
        uncommon_nodes = list(set(range(n_possible_nodes)).difference(common_nodes))
        random.shuffle(common_nodes_to_use)
        random.shuffle(uncommon_nodes)
        offspring = []

        # this actually biases the algorithm as we always create connect subbpaths just by single node
        for subpath in subpaths:
            offspring.extend(subpath)
            if len(common_nodes_to_use) > 1:
                offspring.append(common_nodes_to_use.pop())
            else:
                offspring.append(uncommon_nodes.pop())

        while len(offspring) < len(parentA):
            if len(common_nodes_to_use) > 1:
                offspring.append(common_nodes_to_use.pop())
            else:
                offspring.append(uncommon_nodes.pop())

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
