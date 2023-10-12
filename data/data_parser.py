import numpy as np


def load_matrix_from_csv(file_path):
    matrix = np.loadtxt(file_path, delimiter=';')
    return matrix


def create_adj_matrix(data):
    num_nodes = data.shape[0]
    adj_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                distance = np.sqrt(
                    (data[i, 0] - data[j, 0])**2 + (data[i, 1] - data[j, 1])**2)

                adj_matrix[i, j] = round(distance + data[j, 2])

    return adj_matrix


def get_data():
    problems = [f'TSP{letter}' for letter in 'ABCD']
    data = {problem: None for problem in problems}
    for problem in data.keys():
        matrix = load_matrix_from_csv(f'data/{problem}.csv')
        data[problem] = create_adj_matrix(matrix)
    return data
