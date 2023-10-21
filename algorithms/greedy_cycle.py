import random

import numpy as np


def greedy_cycle(
    adj_matrix: np.ndarray, nodes_cost: np.ndarray, starting_node: int = None
):
    num_nodes = len(adj_matrix)

    starting_node = (
        random.randint(0, num_nodes - 1) if starting_node is None else starting_node
    )

    first, second = np.argpartition(adj_matrix[starting_node] + nodes_cost, 1)[:2]
    cheapest_node = int(first) if first != starting_node else int(second)

    cycle = [starting_node, cheapest_node, starting_node]
    in_cycle = set(cycle)

    while len(in_cycle) < num_nodes // 2:
        best_increase = float("inf")
        best_position = None
        best_vertex = None

        for candidate_id in range(num_nodes):
            if candidate_id not in in_cycle:
                for insert_at, (from_id, to_id) in enumerate(
                    zip(cycle[:-1], cycle[1:])
                ):
                    increase = (
                        adj_matrix[from_id, candidate_id]
                        + adj_matrix[candidate_id, to_id]
                        - adj_matrix[from_id, to_id]
                        + nodes_cost[candidate_id]
                    )
                    if increase < best_increase:
                        best_increase = increase
                        best_position = insert_at
                        best_vertex = candidate_id

        cycle.insert(best_position + 1, best_vertex)

        in_cycle.add(best_vertex)

    return cycle[:-1]
