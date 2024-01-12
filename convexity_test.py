import json
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from algorithms.solution_similarity import SolutionSimilarity


def save_scatterplot(x, y, xlabel, ylabel, title, filename):
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(f"visualizations/8/{filename}")
    plt.clf()


def summary(
    costs: List,
    similarities_edges: List,
    similarities_nodes: List,
    instance_name: str,
    plot_type: str,
):
    corr_edges = np.corrcoef(costs, similarities_edges)[0, 1]
    corr_nodes = np.corrcoef(costs, similarities_nodes)[0, 1]

    print(f"Correlation coefficient (edges): {corr_edges}")
    print(f"Correlation coefficient (nodes): {corr_nodes}")

    save_scatterplot(
        costs,
        similarities_edges,
        "Cost",
        "Number of common edges",
        f"Cost vs. number of common edges {instance_name} {plot_type}",
        f"cost_vs_common_edges_{instance_name}_{plot_type}.png",
    )

    save_scatterplot(
        costs,
        similarities_nodes,
        "Cost",
        "Number of common nodes",
        f"Cost vs. number of common nodes {instance_name} {plot_type}",
        f"cost_vs_common_nodes_{instance_name}_{plot_type}.png",
    )


def best_solution_similarity(solutions: List[Dict], instance_name: str):
    solutions_graphs = [s["solution"]["nodes"] for s in solutions]
    costs = [s["solution"]["cost"] for s in solutions]

    best_solution_idx = np.argmin(costs)
    best_solution = solutions_graphs[best_solution_idx]

    solution_similarity = SolutionSimilarity(best_solution)

    similarities_edges = []
    similarities_nodes = []
    for solution in solutions_graphs:
        similarities_edges.append(solution_similarity.num_common_edges(solution))
        similarities_nodes.append(solution_similarity.num_common_nodes(solution))

    summary(costs, similarities_edges, similarities_nodes, instance_name, "best")


def average_solution_similarity(solutions: List[Dict], instane_name: str):
    solutions_graphs = [s["solution"]["nodes"] for s in solutions]
    costs = [s["solution"]["cost"] for s in solutions]

    solution_similarity = SolutionSimilarity(solutions_graphs[0])

    similarities_edges = [[] for _ in solutions_graphs]
    similarities_nodes = [[] for _ in solutions_graphs]

    for anchor_solution in solutions_graphs:
        solution_similarity = SolutionSimilarity(anchor_solution)
        for i, solution in enumerate(solutions_graphs):
            similarities_edges[i].append(solution_similarity.num_common_edges(solution))
            similarities_nodes[i].append(solution_similarity.num_common_nodes(solution))

    similarities_edges = np.mean(similarities_edges, axis=1)
    similarities_nodes = np.mean(similarities_nodes, axis=1)

    summary(costs, similarities_edges, similarities_nodes, instane_name, "average")


with open("solutions_greedy.json") as f:
    results = json.load(f)

for instance_name, solutions in results.items():
    print(f"Instance: {instance_name}")
    best_solution_similarity(solutions, instance_name)
    average_solution_similarity(solutions, instance_name)
