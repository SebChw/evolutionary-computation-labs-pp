import json
from collections import defaultdict

from data.data_parser import get_data
from visualize import plot_solution


def get_result(data: dict, key: str = "cost"):
    costs = defaultdict(float)
    results = {"best": costs.copy(), "worst": costs.copy(), "avg": costs.copy()}
    best_nodes = defaultdict(list)

    for problem, problem_results in data.items():
        for result in problem_results:
            cost = result[key]
            nodes = result["solution"]
            if results["best"][problem] == 0:
                results["best"][problem] = cost
                results["worst"][problem] = cost
                best_nodes[problem] = nodes
            if cost < results["best"][problem]:
                results["best"][problem] = cost
                best_nodes[problem] = nodes
            if cost > results["worst"][problem]:
                results["worst"][problem] = cost
            results["avg"][problem] += cost / 20

    return results, best_nodes


def get_time_results(data):
    x = defaultdict(float)
    results = {"best": x.copy(), "worst": x.copy(), "avg": x.copy()}
    for problem, problem_results in data.items():
        for result in problem_results:
            cost = result["n_iterations"]
            results["best"][problem] = min(
                cost, results["best"].get(problem, float("inf"))
            )
            results["worst"][problem] = max(
                cost, results["worst"].get(problem, float("-inf"))
            )
            results["avg"][problem] += cost / 20
    return results


with open("solutions_evo.json", "r") as f:
    data = json.load(f)


cost_results, nodes = get_result(data)
time_results = get_time_results(data)
# print(nodes['TSPA']['steepest']['edges']['random'])

for problem, problem_results in nodes.items():
    matrix = get_data()[problem]["original_data"]
    print(nodes, cost_results["best"][problem])
    plot_solution(
        matrix,
        (problem_results, cost_results["best"][problem]),
        problem,
        "evo",
    )

print(json.dumps(cost_results, indent=4))
print(json.dumps(time_results, indent=4))
