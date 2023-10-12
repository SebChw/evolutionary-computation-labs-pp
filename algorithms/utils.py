from dataclasses import dataclass


@dataclass
class Solution:
    nodes: list
    cost: float


def calculate_path_cost(nodes: list[int], adj_matrix: list[list[int]]):
    total_cost = 0
    for i in range(len(nodes) - 1):
        start_node = nodes[i]
        end_node = nodes[i+1]
        total_cost += adj_matrix[start_node][end_node]
    return total_cost
