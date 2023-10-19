from dataclasses import dataclass

import numpy as np


@dataclass
class Solution:
    nodes: list
    cost: float


def calculate_path_cost(
    nodes: list[int], adj_matrix: np.ndarray, nodes_cost: np.ndarray
):
    total_cost = np.sum(nodes_cost[nodes])
    total_cost += np.sum(adj_matrix[nodes, nodes[1:] + [nodes[0]]])

    return total_cost


def find_two_smallest(values):
    smallest = float("inf")
    second_smallest = float("inf")
    for i, x in enumerate(values):
        if x <= smallest:
            second_smallest = smallest
            smallest = x
        elif x < second_smallest:
            second_smallest = x

    return smallest, second_smallest
