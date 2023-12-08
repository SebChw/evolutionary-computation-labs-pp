import json
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict

from joblib import Parallel, delayed

from algorithms.ils import ILS
from algorithms.lns import LNS
from algorithms.local_search import LocalSearch
from algorithms.msls import MSLS
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
    N_ITERATIONS = 19
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

    tasks = []
    for _ in range(N_ITERATIONS):
        ils = ILS()
        # TODO: after setting optimal parameters remove this!
        # ils(distance_matrix, nodes_cost)

        tasks.append(delayed(ils)(distance_matrix, nodes_cost))

    n_jobs = len(tasks)
    parallel_results = Parallel(n_jobs=n_jobs)(tasks)

    for result in parallel_results:
        solution_dict = asdict(Solution(result["solution"], result["cost"]))
        results[problem].append(
            {
                "method": "ils",
                "solution": solution_dict,
                "n_iterations": result["n_iterations"],
                "best_found_at": result["best_found_at"],
            }
        )


def evaluate_lns(problem: str, instance: dict):
    N_ITERATIONS = 20
    distance_matrix = instance["dist_matrix"]
    nodes_cost = instance["nodes_cost"]

    tasks = []
    for _ in range(N_ITERATIONS):
        ils = LNS()
        tasks.append(delayed(ils)(distance_matrix, nodes_cost))

    n_jobs = len(tasks)
    parallel_results = Parallel(n_jobs=n_jobs)(tasks)

    for result in parallel_results:
        solution_dict = asdict(Solution(result["solution"], result["cost"]))
        results[problem].append(
            {
                "method": "lns",
                "solution": solution_dict,
                "n_iterations": result["n_iterations"],
            }
        )


for problem, instance in data.items():
    print(f"Problem: {problem}")
    # evaluate_msls(problem, instance)
    # evaluate_ils(problem, instance)
    evaluate_lns(problem, instance)


# Save the results to a JSON file
with open("solutionsLNS.json", "w") as file:
    json.dump(dict(results), file, indent=4)
print("Results have been saved to solutionsLNS.json")
