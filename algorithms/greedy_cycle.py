import random

def greedy_cycle(adj_matrix: list[list[int]]):
    num_nodes = len(adj_matrix)

    # Select a random starting vertex.
    start_vertex = random.randint(0, num_nodes - 1)

    # Find the nearest vertex to the starting one.
    min_distance = float('inf')
    nearest_vertex = None
    for j in range(num_nodes):
        if j != start_vertex:
            if adj_matrix[start_vertex][j] < min_distance:
                min_distance = adj_matrix[start_vertex][j]
                nearest_vertex = j

    # Start an initial cycle containing the starting vertex, its nearest neighbor, and back to the starting vertex.
    cycle = [start_vertex, nearest_vertex, start_vertex]
    in_cycle = set(cycle)

    # While not all vertices have been added to the cycle:
    while len(in_cycle) < num_nodes:
        best_increase = float('inf')
        best_position = None
        best_vertex = None

        # For each vertex not in the cycle, find the position where inserting it would result in the smallest increase in cycle length.
        for j in range(num_nodes):
            if j not in in_cycle:
                for pos in range(1, len(cycle)):
                    increase = adj_matrix[cycle[pos-1]][j] + adj_matrix[j][cycle[pos]] - adj_matrix[cycle[pos-1]][cycle[pos]]
                    if increase < best_increase:
                        best_increase, best_position, best_vertex = increase, pos, j

        # Insert the vertex in that position.
        cycle.insert(best_position, best_vertex)
        in_cycle.add(best_vertex)

    return cycle
