import random


def random_hamiltonian(adj_matrix: list[list[int]]):
    num_nodes = len(adj_matrix)
    num_selected = (num_nodes + 1) // 2

    selected_nodes = random.sample(range(num_nodes), num_selected)
    selected_nodes.append(selected_nodes[0])

    return selected_nodes
