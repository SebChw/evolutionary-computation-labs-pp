import logging
import random
import time
from collections import namedtuple
from datetime import datetime
from typing import Dict, List, Set, Tuple

import numpy as np

from algorithms.greedy_2_regret import Greedy2Regret
from algorithms.local_search import LocalSearch, LSStrategy
from algorithms.random import random_hamiltonian
from algorithms.utils import calculate_path_cost

LOG_FILENAME = datetime.now().strftime("logfile_%H_%M_%S_%d_%m_%Y.log")
logging.basicConfig(
    filename=LOG_FILENAME,
    encoding="utf-8",
    level=logging.DEBUG,
)

TSPSolution = namedtuple("TSPSolution", ["solution", "cost"])
NOT_YET_FOUND = -1


class HybridEvolutionary:
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

    def common_crossover(self, parentA, parentB) -> List[int]:
        # get common nodes
        common_nodes = set(parentA).intersection(set(parentB))

        # Get common subpaths
        edgesA = self.get_edges(parentA)
        edgesB = self.get_edges(parentB)
        common_edges = list(set(edgesA).intersection(set(edgesB)))
        # Incoming and outcoming map will be used to reconstruct paths
        incoming_map = {edge[0]: edge[1] for edge in common_edges}
        outcoming_map = {edge[1]: edge[0] for edge in common_edges}
        used_edges = set()
        subpaths = []
        for edge in common_edges:
            if edge not in used_edges:
                subpath = [edge[0], edge[1]]
                used_edges.add(edge)
                while True:
                    if subpath[-1] in incoming_map:
                        # we have to reconstruct the path
                        subpath.append(incoming_map[subpath[-1]])
                        edge = (subpath[-2], subpath[-1])
                        used_edges.add(edge)
                    elif subpath[0] in outcoming_map:
                        subpath.insert(0, outcoming_map[subpath[0]])
                        edge = (subpath[0], subpath[1])
                        used_edges.add(edge)
                    else:
                        break

                subpaths.append(subpath)

        logging.debug(
            f"parentA: {parentA}\n, parentB: {parentB}\n, subpaths: {subpaths}\n, common_nodes: {common_nodes}"
        )

        n_nodes_in_subpaths = sum([len(subpath) for subpath in subpaths])
        if n_nodes_in_subpaths > len(parentA):
            logging.error(
                f"Subpaths: {subpaths}\n ParentA: {parentA}\n ParentB: {parentB} \n Common nodes: {common_nodes}"
            )
            raise Exception("Subpaths are calculated incorrectly!")

        offspring = []
        # Place subpaths in somehow random order. Total random would be very computationally expensive
        random.shuffle(subpaths)
        indices_to_be_filled = []
        for path in subpaths:
            n_nodes_in_subpaths -= len(path)
            offspring.extend(path)
            max_gap_size = len(parentA) - len(offspring) - n_nodes_in_subpaths
            gap = random.randint(0, max_gap_size)

            for _ in range(gap):
                # Special places to which we will later add nodes
                offspring.append(NOT_YET_FOUND)
                indices_to_be_filled.append(len(offspring) - 1)

        while len(offspring) < len(parentA):
            offspring.append(NOT_YET_FOUND)
            indices_to_be_filled.append(len(offspring) - 1)

        already_used_nodes = set()
        for path in subpaths:
            already_used_nodes.update(path)

        # We cannot use node that consitutes a common edge as we will have a duplicate
        common_nodes_to_use = list(common_nodes.difference(already_used_nodes))
        n_possible_nodes = self.adj_matrix.shape[0]
        uncommon_nodes = list(set(range(n_possible_nodes)).difference(common_nodes))
        random.shuffle(common_nodes_to_use)
        random.shuffle(uncommon_nodes)

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

        if len(set(offspring)) != len(offspring):
            logging.error(
                f"Offspring: {offspring}\n ParentA: {parentA}\n ParentB: {parentB} \n Common nodes: {common_nodes}\n Common nodes to use: {list(common_nodes.difference(already_used_nodes))}\n Uncommon nodes: {list(set(range(n_possible_nodes)).difference(common_nodes))} "
            )
            raise Exception("Duplicate nodes in offspring!")
        return offspring

    def repair_crossover(self, parentA, parentB) -> List[int]:
        def repair(partial_solution, adj_matrix, nodes_cost):
            """
            Repair operator: rebuilds the solution, potentially finding a better one.
            Use best greedy heuristic here.
            """
            greedy = Greedy2Regret(alpha=0.5)
            repaired_solution = greedy(
                adj_matrix, nodes_cost, starting_solution=partial_solution
            )

            return repaired_solution

        edgesA = self.get_edges(parentA)
        edgesB = self.get_edges(parentB)
        common_edges = list(set(edgesA).intersection(set(edgesB)))

        if random.random() < 0.5:
            base_edges = edgesA
        else:
            base_edges = edgesB

        offspring = [base_edges[0][0], base_edges[0][1]]
        for edge in base_edges[1:]:
            if edge in common_edges:
                if offspring[-1] != edge[0]:
                    offspring.append(edge[0])
                if offspring[0] != edge[1]:
                    offspring.append(edge[1])

        logging.debug(
            f"Crossover Repair: parentA: {parentA}\n, parentB: {parentB}\n, offspring: {offspring}"
        )

        if len(set(offspring)) != len(offspring):
            logging.error(
                f"Offspring: {offspring}\n ParentA: {parentA}\n ParentB: {parentB})"
            )
            raise Exception("Duplicate nodes in offspring!")

        return repair(offspring, self.adj_matrix, self.nodes_cost)

    def generate_initial_population(self) -> Tuple[List[TSPSolution], Set[int]]:
        population = []
        population_costs = set()

        while len(population) < self.elite_population_size:
            solution = self.local_search(
                self.adj_matrix,
                self.nodes_cost,
                random_hamiltonian(self.adj_matrix, self.nodes_cost),
            )
            cost = calculate_path_cost(solution, self.adj_matrix, self.nodes_cost)
            if cost not in population_costs:
                population_costs.add(cost)
                solution = TSPSolution(solution=solution, cost=cost)
                population.append(solution)

        return population, population_costs

    def __call__(self, adj_matrix: np.ndarray, nodes_cost: np.ndarray) -> Dict:
        start_time = time.perf_counter()

        self.adj_matrix = adj_matrix
        self.nodes_cost = nodes_cost

        population, population_cost = self.generate_initial_population()
        population = sorted(population, key=lambda x: x.cost)

        n_iterations = 0
        while time.perf_counter() - start_time < self.max_time:
            # Select parents
            idx = np.random.choice(len(population), size=2, replace=False)
            parentA, parentB = population[idx[0]].solution, population[idx[1]].solution

            # Generate offspring
            offspring = random.choice(self.recombination_operators)(parentA, parentB)

            if self.do_LS:
                offspring = self.local_search(
                    self.adj_matrix, self.nodes_cost, offspring
                )

            offspring_cost = calculate_path_cost(
                offspring, self.adj_matrix, self.nodes_cost
            )

            if (
                offspring_cost < population[-1].cost
                and offspring_cost not in population_cost
            ):
                population_cost.remove(population[-1].cost)
                population_cost.add(offspring_cost)
                population[-1] = TSPSolution(solution=offspring, cost=offspring_cost)
                population = sorted(population, key=lambda x: x.cost)

            logging.debug(
                f"iteration {n_iterations}: population_cost: {population_cost}"
            )
        return {
            "solution": population[0].solution,
            "cost": population[0].cost,
            "n_iterations": n_iterations,
        }
