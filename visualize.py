import json

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from data.data_parser import get_data


def get_best_values(solutions: dict, problem: str):
    best_values = {}
    for algorithm_name, algorithm_solutions in solutions[problem].items():
        best_values[algorithm_name] = min(
            algorithm_solutions, key=lambda solution: solution["cost"]
        )
    return best_values


def plot_solution(
    matrix: np.ndarray,
    best_solutions: dict,
    problem_name: str,
    algorithm_name: str,
):
    plt.figure(figsize=(50, 50))

    # solution = best_solutions[problem_name][algorithm_name]
    # nodes = solution["nodes"]
    # nodes += [nodes[0]]
    nodes = best_solutions[0]
    nodes += [nodes[0]]

    x_coords = [matrix[node][0] for node in nodes]
    y_coords = [matrix[node][1] for node in nodes]
    costs = [matrix[node][2] for node in nodes]

    normalized_costs = (costs - np.min(costs)) / \
        (np.max(costs) - np.min(costs))

    plt.scatter(
        x_coords, y_coords, s=1200, c=normalized_costs, cmap="coolwarm", alpha=0.6
    )

    for i in range(len(nodes) - 1):
        x1, y1 = matrix[nodes[i]][:2]
        x2, y2 = matrix[nodes[i + 1]][:2]
        plt.plot([x1, x2], [y1, y2], color="gray", alpha=0.6)

    other_nodes = [node for node in range(len(matrix)) if node not in nodes]
    other_costs = [matrix[node][2] for node in other_nodes]
    other_x_coords = [matrix[node][0] for node in other_nodes]
    other_y_coords = [matrix[node][1] for node in other_nodes]
    other_normalized_costs = (other_costs - np.min(other_costs)) / (
        np.max(other_costs) - np.min(other_costs)
    )

    plt.scatter(
        other_x_coords,
        other_y_coords,
        s=1200,
        c=other_normalized_costs,
        cmap="coolwarm",
        alpha=0.6,
    )

    plt.title(
        f"Visualization for {problem_name} using {algorithm_name}. Value={best_solutions[1]}",
        fontsize=60,
    )
    plt.xticks(fontsize=36)
    plt.yticks(fontsize=36)
    plt.colorbar().ax.tick_params(labelsize=36)

    plt.savefig(f"visualizations/{problem_name}_{algorithm_name}.png")


def compute_best_solution(algorithm_solutions: list) -> dict:
    best = min(algorithm_solutions, key=lambda solution: solution["cost"])
    return {"cost": best["cost"], "nodes": best["nodes"]}


def compute_worst_solution(algorithm_solutions: list) -> dict:
    worst = max(algorithm_solutions, key=lambda solution: solution["cost"])
    return {"cost": worst["cost"], "nodes": worst["nodes"]}


def compute_average_cost(algorithm_solutions: list) -> dict:
    average = sum(solution["cost"] for solution in algorithm_solutions) / len(
        algorithm_solutions
    )
    return {"cost": average, "nodes": None}


def get_extremes(solutions: dict):
    best_solutions = {}
    worst_solutions = {}
    average_solutions = {}

    for problem, algorithms in solutions.items():
        best_solutions[problem] = {}
        worst_solutions[problem] = {}
        average_solutions[problem] = {}

        for algorithm_name, algorithm_solutions in algorithms.items():
            best_solutions[problem][algorithm_name] = compute_best_solution(
                algorithm_solutions
            )
            worst_solutions[problem][algorithm_name] = compute_worst_solution(
                algorithm_solutions
            )
            average_solutions[problem][algorithm_name] = compute_average_cost(
                algorithm_solutions
            )

    return best_solutions, worst_solutions, average_solutions


def create_table_data(best_solutions, worst_solutions, average_solutions):
    tables = {}

    algorithm_names = [
        x for x in best_solutions[list(best_solutions.keys())[0]].keys()]

    for problem_name in best_solutions.keys():
        rows = [["best"], ["worst"], ["average"]]
        for algorithm_name in algorithm_names:
            best_cost = best_solutions[problem_name][algorithm_name]["cost"]
            worst_cost = worst_solutions[problem_name][algorithm_name]["cost"]
            average_cost = average_solutions[problem_name][algorithm_name]["cost"]
            rows[0].append(best_cost)
            rows[1].append(worst_cost)
            rows[2].append(average_cost)

        tables[problem_name] = rows

    return [""] + algorithm_names, tables


def save_results(best_solutions, worst_solutions, average_solutions):
    headers, data = create_table_data(
        best_solutions, worst_solutions, average_solutions
    )
    for problem_name, table in data.items():
        plt.figure(figsize=(15, 5))
        plt.axis("off")
        plt.table(cellText=table, colLabels=headers, loc="center")
        plt.savefig(f"visualizations/{problem_name}_table.png")


def main():
    with open("solutions.json", "r") as file:
        solutions = json.load(file)

    best_solutions, worst_solutions, average_solutions = get_extremes(
        solutions)

    data = get_data()
    for problem_name in data.keys():
        for algorithm_name in solutions[problem_name].keys():
            plot_solution(
                data[problem_name]["original_data"],
                best_solutions,
                problem_name,
                algorithm_name,
            )

    save_results(best_solutions, worst_solutions, average_solutions)


if __name__ == "__main__":
    main()
