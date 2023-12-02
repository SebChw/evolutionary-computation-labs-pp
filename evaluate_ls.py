import json
import time
from collections import defaultdict
from copy import copy, deepcopy
from dataclasses import asdict

import numpy as np
from joblib import Parallel, delayed

from algorithms.ils import ILS
from algorithms.local_search import LocalSearch
from algorithms.msls import MSLS
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
results = defaultdict(list)


def evaluate_msls(problem: str, instance: dict):
    N_ITERATIONS = 5
    distance_matrix = instance["dist_matrix"]
    nodes_cost = instance["nodes_cost"]

    for _ in range(N_ITERATIONS):
        msls = MSLS()
        result = msls(distance_matrix, nodes_cost)
        solution_dict = asdict(Solution(result["solution"], result["cost"]))
        results[problem].append(
            {
                "method": "msls",
                "solution": solution_dict,
                "time": result["time"],
            }
        )


def evaluate_ils(problem: str, instance: dict):
    N_ITERATIONS = 20
    distance_matrix = instance["dist_matrix"]
    nodes_cost = instance["nodes_cost"]

    for _ in range(N_ITERATIONS):
        ils = ILS()
        result = ils(distance_matrix, nodes_cost)
        solution_dict = asdict(Solution(result["solution"], result["cost"]))
        results[problem].append(
            {
                "method": "ils",
                "solution": solution_dict,
                "time": result["time"],
            }
        )


for problem, instance in data.items():
    print(f"Problem: {problem}")
    evaluate_msls(problem, instance)
    # evaluate_ils(problem, instance)


# Save the results to a JSON file
with open("solutions.json", "w") as file:
    json.dump(dict(results), file, indent=4)
print("Results have been saved to solutions.json")
