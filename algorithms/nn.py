import random

import numpy as np


def nearest_neighbor_hamiltonian(
    adj_matrix: np.ndarray, nodes_cost: np.ndarray, starting_node: int = None
):
    num_nodes = len(adj_matrix)
    num_selected = (num_nodes + 1) // 2

    starting_node = (
        random.randint(0, num_nodes - 1) if starting_node is None else starting_node
    )

    selected_nodes = [starting_node]
    selected_set = {starting_node}
    for _ in range(num_selected - 1):
        last_node = selected_nodes[-1]
        costs = adj_matrix[last_node] + nodes_cost
        min_distance = float("inf")
        min_node = None
        for j in range(num_nodes):
            if j not in selected_set:
                if costs[j] < min_distance:
                    min_distance = costs[j]
                    min_node = j
        selected_nodes.append(min_node)
        selected_set.add(min_node)

    return selected_nodes
