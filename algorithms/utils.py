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
