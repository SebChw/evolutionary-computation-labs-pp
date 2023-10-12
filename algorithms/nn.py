import random


def nearest_neighbor_hamiltonian(adj_matrix: list[list[int]]):
    num_nodes = len(adj_matrix)
    num_selected = (num_nodes + 1) // 2

    starting_node = random.randint(0, num_nodes - 1)

    selected_nodes = [starting_node]
    selected_set = {starting_node}
    for _ in range(num_selected - 1):
        last_node = selected_nodes[-1]
        min_distance = float('inf')
        min_node = None
        for j in range(num_nodes):
            if j not in selected_set:
                if adj_matrix[last_node][j] < min_distance:
                    min_distance = adj_matrix[last_node][j]
                    min_node = j
        selected_nodes.append(min_node)
        selected_set.add(min_node)
    selected_nodes.append(selected_nodes[0])

    return selected_nodes
