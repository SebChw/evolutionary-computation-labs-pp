import logging
import pickle
import random
import time
from collections import namedtuple
from datetime import datetime
from typing import Dict, List, Set, Tuple

import numpy as np

from algorithms.greedy_2_regret import Greedy2Regret
from algorithms.local_search import LocalSearch, LSStrategy
from algorithms.random_solution import random_hamiltonian
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
    def __init__(
        self,
        elite_population_size=20,
        do_LS=True,
        max_time: int = 7 * 200,
    ):
        """
        Initialize the LNS algorithm parameters.
        """
        self.elite_population_size = elite_population_size
        self.do_LS = do_LS
        self.max_time = max_time
        self.destroy_rate = 0.2

        self.use_mutation = False
        self.mutation_probability = 0.2

        self.ls_strategy = LSStrategy.STEEPEST
        self.ls_exchange_nodes = False
        self.local_search = LocalSearch(self.ls_strategy, self.ls_exchange_nodes)

        self.recombination_operators = [self.common_crossover, self.repair_crossover]

        self.X = []
        self.y = []
        with open("classifiers/TSPA_big.pkl", "rb") as f:
            self.clf = pickle.load(f)

    def get_edges(self, solution, reversed=False) -> List[Tuple[int, int]]:
        if not reversed:
            edges = [
                (solution[i], solution[i + 1]) for i in range(len(solution) - 1)
            ] + [(solution[-1], solution[0])]
        else:
            edges = [
                (solution[i + 1], solution[i]) for i in range(len(solution) - 1)
            ] + [(solution[0], solution[-1])]

        return edges

    def get_common_edges(self, parentA, parentB) -> Set[Tuple[int, int]]:
        edgesA = set(self.get_edges(parentA))
        # If we make one reversed, we will also catch situation where in one soltuon its 1->2 and in the other 2->1
        edgesB = set(self.get_edges(parentB)).union(
            set(self.get_edges(parentB, reversed=True))
        )

        return edgesA.intersection(edgesB)

    def build_base_offspring(self, parentA, parentB):
        # For both crossover we basically can start from the same point. We copy common edges
        # This is more general than copying common nodes. As is we have 2 nodes it doesn't mean that we have a common edge.
        # TODO: If you want you can implement strategy that copies common nodes, ans use it here with some predefined probability.
        common_edges = self.get_common_edges(parentA, parentB)

        if random.random() < 0.5:
            base_edges = self.get_edges(parentA)
        else:
            base_edges = self.get_edges(parentB)

        offspring = []
        for edge in base_edges[1:]:
            if edge in common_edges:
                if not offspring:
                    offspring.append(edge[0])
                    offspring.append(edge[1])
                else:
                    # Special case if we have 2 consecutive common edges or the last and first edge is common
                    if offspring[-1] != edge[0]:
                        offspring.append(edge[0])
                    if offspring[0] != edge[1]:
                        offspring.append(edge[1])

        return offspring

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

    def common_crossover(self, parentA, parentB) -> List[int]:
        # get common nodes
        common_nodes = set(parentA).intersection(set(parentB))

        offspring = self.build_base_offspring(parentA, parentB)

        # logging.debug(
        #     f"parentA: {parentA}\n, parentB: {parentB}\n, offspring: {offspring}"
        # )

        already_used_nodes = set(offspring)
        # We cannot use node that consitutes a common edge as we will have a duplicate
        common_nodes_to_use = list(common_nodes.difference(already_used_nodes))
        offspring.extend(common_nodes_to_use)

        n_possible_nodes = self.adj_matrix.shape[0]
        uncommon_nodes = list(set(range(n_possible_nodes)).difference(common_nodes))
        random.shuffle(uncommon_nodes)

        while len(offspring) < len(parentA):
            offspring.append(uncommon_nodes.pop())

        if len(offspring) != len(parentA):
            raise Exception("Offspring has not the same length as parents!")

        if len(set(offspring)) != len(offspring):
            # logging.error(
            #     f"counts: {np.unique(offspring, return_counts=True)} \nOffspring: {offspring}\n ParentA: {parentA}\n ParentB: {parentB} \n Common nodes: {common_nodes}\n Common nodes to use: {list(common_nodes.difference(already_used_nodes))}\n Uncommon nodes: {list(set(range(n_possible_nodes)).difference(common_nodes))} "
            # )
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

        offspring = self.build_base_offspring(parentA, parentB)

        # logging.debug(
        #     f"Crossover Repair: parentA: {parentA}\n, parentB: {parentB}\n, offspring: {offspring}"
        # )

        if len(set(offspring)) != len(offspring):
            # logging.error(
            #     f"counts: {np.unique(offspring, return_counts=True)} \nOffspring: {offspring}\n ParentA: {parentA}\n ParentB: {parentB}"
            # )
            raise Exception("Duplicate nodes in offspring!")

        if len(offspring) == 0:
            # TODO if still exceptoion is raised catch it here, I have no idea what is going on. After this fix I didn't get any error
            if random.random() < 0.5:
                offspring = [parentB[0], parentB[1]]
            else:
                offspring = [parentA[0], parentA[1]]

            # logging.debug(f"New offspring: {offspring}")

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
        logging.debug(f"Initial population: {[x.cost for x in population]}")

        n_iterations = 0
        iterations_without_improvement = 0
        iterations_no_imp = []
        iterations_without_best_improvement = 0
        iterations_no_best_imp = []
        while time.perf_counter() - start_time < self.max_time:
            # Select parents
            idx = np.random.choice(len(population), size=2, replace=False)
            parentA, parentB = population[idx[0]].solution, population[idx[1]].solution

            # Generate offspring
            offspring = random.choice(self.recombination_operators)(parentA, parentB)

            if random.random() < self.mutation_probability and self.use_mutation:
                logging.debug("Mutation is applied")
                offspring = self.destroy(offspring, nodes_cost)
                offspring = self.repair(offspring, self.adj_matrix, self.nodes_cost)

            # self.X.append(copy.deepcopy(offspring))
            cost_pred = self.clf.predict([offspring])[0]
            if not cost_pred:
                logging.debug("skip LS")
                continue
            if self.do_LS:
                offspring = self.local_search(
                    self.adj_matrix, self.nodes_cost, offspring
                )

            offspring_cost = calculate_path_cost(
                offspring, self.adj_matrix, self.nodes_cost
            )
            # self.y.append(offspring_cost)

            if (
                offspring_cost < population[-1].cost
                and offspring_cost not in population_cost
            ):
                logging.debug(
                    f"offspring_cost: {offspring_cost} better than {population[-1].cost}"
                )
                if offspring_cost < population[0].cost:
                    iterations_no_best_imp.append(iterations_without_best_improvement)
                    logging.debug(
                        f"After: {iterations_no_best_imp} New best solution with cost: {offspring_cost} was found"
                    )
                    iterations_without_best_improvement = 0
                else:
                    iterations_no_imp.append(iterations_without_improvement)
                    logging.debug(
                        f"After: {iterations_no_imp} New solution that can enter population with the cost: {offspring_cost} was found"
                    )

                    iterations_without_improvement = 0
                    iterations_without_best_improvement += 1

                population_cost.remove(population[-1].cost)
                population_cost.add(offspring_cost)
                population[-1] = TSPSolution(solution=offspring, cost=offspring_cost)
                population = sorted(population, key=lambda x: x.cost)

                logging.debug(f"New population: {[x.cost for x in population]}")
            else:
                iterations_without_improvement += 1
                iterations_without_best_improvement += 1
                # logging.debug(
                #     f"offspring_cost: {offspring_cost} worse than {population[-1].cost}"
                # )

            logging.debug(
                f"iteration {n_iterations}: population_cost: {population_cost}"
            )

            n_iterations += 1

            if iterations_without_best_improvement > 200:
                logging.debug(
                    f"iterations_without_best_improvement: {iterations_without_best_improvement}, finish the algorithm"
                )
                break
            if (iterations_without_best_improvement + 1) % 50 == 0:
                logging.debug(
                    f"iterations_without_best_improvement: {iterations_without_best_improvement}, add completely new solution to the population"
                )
                offspring = self.local_search(
                    self.adj_matrix,
                    self.nodes_cost,
                    random_hamiltonian(self.adj_matrix, self.nodes_cost),
                )
                offspring_cost = calculate_path_cost(
                    offspring, self.adj_matrix, self.nodes_cost
                )
                population_cost.remove(population[-1].cost)
                population_cost.add(offspring_cost)
                population[-1] = TSPSolution(solution=offspring, cost=offspring_cost)
                population = sorted(population, key=lambda x: x.cost)
            if iterations_without_best_improvement > 50 and not self.use_mutation:
                logging.debug(
                    f"iterations_without_best_improvement: {iterations_without_best_improvement}, turn on mutation"
                )
                self.use_mutation = True

        return {
            "solution": population[0].solution,
            "cost": population[0].cost,
            "n_iterations": n_iterations,
            "time": time.perf_counter() - start_time,
            "max_time": self.max_time,
            "X": self.X,
            "y": self.y,
        }
