import random
import numpy as np
from algorithms.utils import find_two_smallest


def greedy_2_regret_weighted(
    adj_matrix: np.ndarray, nodes_cost: np.ndarray, starting_node: int = None, alpha: float = 0.5
):
    num_nodes = len(adj_matrix)

    starting_node = (
        random.randint(
            0, num_nodes - 1) if starting_node is None else starting_node
    )

    first, second = np.argpartition(
        adj_matrix[starting_node] + nodes_cost, 2)[:2]
    cheapest_node = int(first) if first != starting_node else int(second)

    cycle = [starting_node, cheapest_node, starting_node]
    in_cycle = set(cycle)

    while len(in_cycle) < num_nodes // 2:
        best_score = float("-inf")
        best_position = None
        best_vertex = None

        for candidate_id in range(num_nodes):
            if candidate_id not in in_cycle:
                insertion_costs = [
                    adj_matrix[from_id, candidate_id]
                    + adj_matrix[candidate_id, to_id]
                    - adj_matrix[from_id, to_id]
                    + nodes_cost[candidate_id]
                    for (from_id, to_id) in zip(cycle[:-1], cycle[1:])
                ]
                if len(insertion_costs) > 1:
                    smallest, second_smallest = find_two_smallest(
                        insertion_costs)
                    regret = second_smallest - smallest
                    score = alpha * regret - \
                        (1 - alpha) * min(insertion_costs)
                    if score > best_score:
                        best_score = score
                        best_position = np.argmin(insertion_costs)
                        best_vertex = candidate_id

        cycle.insert(best_position + 1, best_vertex)
        in_cycle.add(best_vertex)

    return cycle[:-1]
