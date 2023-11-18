import json
import time
from collections import defaultdict
from copy import copy, deepcopy
from dataclasses import asdict

import numpy as np
from joblib import Parallel, delayed

from algorithms.greedy_2_regret import Greedy2Regret
from algorithms.local_search import LocalSearch, LSStrategy
from algorithms.random import random_hamiltonian
from algorithms.utils import Solution, calculate_path_cost
from data.data_parser import get_data


def perform_local_search(
    greedy,
    exchange_nodes,
    starting_solution_name,
    starting_solution,
    distance_matrix,
    nodes_cost,
    candidate_moves,
):
    local_search_start_time = time.perf_counter()
    local_search = LocalSearch(
        greedy=greedy, exchange_nodes=exchange_nodes, candidate_moves=candidate_moves
    )
    result = local_search(distance_matrix, nodes_cost, deepcopy(starting_solution))
    solution_dict = asdict(
        Solution(result, calculate_path_cost(result, distance_matrix, nodes_cost))
    )
    local_search_time = time.perf_counter() - local_search_start_time
    return (
        greedy,
        exchange_nodes,
        starting_solution_name,
        solution_dict,
        local_search_time,
        candidate_moves,
    )


data = get_data()
results = defaultdict(
    lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
)

for problem, instance in data.items():
    print(problem)
    distance_matrix = instance["dist_matrix"]
    nodes_cost = instance["nodes_cost"]
    tasks = []
    for i in range(200):
        starting_solutions = {}
        starting_solution_times = {}
        start_time_random = time.perf_counter()
        starting_solutions["random"] = random_hamiltonian(distance_matrix, nodes_cost)
        starting_solution_times["random"] = time.perf_counter() - start_time_random

        start_time_greedy = time.perf_counter()
        starting_solutions["greedy"] = Greedy2Regret(alpha=0.5)(
            distance_matrix, nodes_cost, starting_node=i
        )
        starting_solution_times["greedy"] = time.perf_counter() - start_time_greedy

        local_search = LocalSearch(LSStrategy.STEEPEST_MOVE_EVAL, exchange_nodes=False)
        result = local_search(
            distance_matrix, nodes_cost, deepcopy(starting_solutions["random"])
        )
        print("LS pure steepest")
        local_search = LocalSearch(LSStrategy.STEEPEST, exchange_nodes=False)
        result = local_search(
            distance_matrix, nodes_cost, deepcopy(starting_solutions["random"])
        )
        raise Exception("Finito.")

    n_jobs = -1  # Use all available cores
    parallel_results = Parallel(n_jobs=n_jobs)(tasks)

    # Process the results
    for result in parallel_results:
        (
            greedy,
            exchange_nodes,
            starting_solution_name,
            solution_dict,
            local_search_time,
            candidate_moves,
        ) = result
        greedy_name = "greedy" if greedy else "steepest"
        exchange_nodes_name = "nodes" if exchange_nodes else "edges"
        for x in range(len(solution_dict["nodes"])):
            solution_dict["nodes"][x] = int(solution_dict["nodes"][x])
        results[problem][greedy_name][exchange_nodes_name][
            starting_solution_name
        ].append(
            {
                "solution": solution_dict,
                "local_search_time": local_search_time,
                "starting_solution_time": starting_solution_times[
                    starting_solution_name
                ],
                "total_time": local_search_time
                + starting_solution_times[starting_solution_name],
                "candidate_moves": candidate_moves,
            }
        )


# Save the results to a JSON file
with open("solutions.json", "w") as file:
    json.dump(dict(results), file, indent=4)
print("Results have been saved to solutions.json")
