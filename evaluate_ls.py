import json
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict

from joblib import Parallel, delayed

from algorithms.evolutionary import HybridEvolutionary
from algorithms.ils import ILS
from algorithms.lns import LNS
from algorithms.local_search import LocalSearch, LSStrategy
from algorithms.msls import MSLS
from algorithms.random_solution import random_hamiltonian
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
        ils = LNS(do_LS=False)
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


def perform_n_greedy(n: int, instance: dict):
    tasks = []
    distance_matrix = instance["dist_matrix"]
    nodes_cost = instance["nodes_cost"]

    for _ in range(n):
        starting_solution = random_hamiltonian(distance_matrix, nodes_cost)
        local_search = LocalSearch(strategy=LSStrategy.GREEDY, exchange_nodes=False)

        tasks.append(
            delayed(local_search)(distance_matrix, nodes_cost, starting_solution)
        )

    parallel_results = Parallel(n_jobs=-1)(tasks)
    for result in parallel_results:
        solution_dict = asdict(
            Solution(result, calculate_path_cost(result, distance_matrix, nodes_cost))
        )
        results[problem].append(
            {
                "method": "greedy",
                "solution": solution_dict,
            }
        )


def hybridevolutionary(instance: dict):
    distance_matrix = instance["dist_matrix"]
    nodes_cost = instance["nodes_cost"]

    evo = HybridEvolutionary(elite_population_size=3, do_LS=False, max_time=7 * 2)
    result = evo(distance_matrix, nodes_cost)
    return result


def max_time_hybrid_evolutionary(instance: dict):
    distance_matrix = instance["dist_matrix"]
    nodes_cost = instance["nodes_cost"]

    current_time = time.perf_counter()
    results = []

    while time.perf_counter() - current_time < 7 * 10:
        evo = HybridEvolutionary(
            elite_population_size=3,
            do_LS=True,
            max_time=7 * 10,
        )
        result = evo(distance_matrix, nodes_cost)
        results.append(result)

    return results


results = defaultdict(list)

for problem, instance in data.items():
    print(f"Problem: {problem}")

    # # Parallel execution of the hybridevolutionary function for each instance
    # results[problem] = Parallel(n_jobs=-1)(
    #     delayed(hybridevolutionary)(instance) for _ in range(200)
    # )
    results[problem] = Parallel(n_jobs=-1)(
        delayed(max_time_hybrid_evolutionary)(instance) for _ in range(2)
    )

for problem in results:
    partial_results = []
    for result_list in results[problem]:
        for result in result_list:
            partial_results.append(result)
    results[problem] = partial_results


# Save the results to a JSON file
with open("solutions_evo.json", "w") as file:
    json.dump(results, file, indent=4)
print("Results have been saved to solutions_evo.json")
