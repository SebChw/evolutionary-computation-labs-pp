import random

def greedy_cycle(adj_matrix: list[list[int]], starting_node: int = None):
    num_nodes = len(adj_matrix)

    starting_node = random.randint(0, num_nodes - 1) if starting_node is None else starting_node

    min_distance = float('inf')
    nearest_node = None
    for j in range(num_nodes):
        if j != starting_node:
            if adj_matrix[starting_node][j] < min_distance:
                min_distance = adj_matrix[starting_node][j]
                nearest_node = j

    cycle = [starting_node, nearest_node, starting_node]
    in_cycle = set(cycle)

    while len(in_cycle) < num_nodes:
        best_increase = float('inf')
        best_position = None
        best_vertex = None

        for j in range(num_nodes):
            if j not in in_cycle:
                for pos in range(1, len(cycle)):
                    increase = adj_matrix[cycle[pos-1]][j] + adj_matrix[j][cycle[pos]] - adj_matrix[cycle[pos-1]][cycle[pos]]
                    if increase < best_increase:
                        best_increase, best_position, best_vertex = increase, pos, j

        cycle.insert(best_position, best_vertex)
        in_cycle.add(best_vertex)

    return cycle
