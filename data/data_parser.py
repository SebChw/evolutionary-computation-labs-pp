import numpy as np
from scipy.spatial.distance import cdist


def load_matrix_from_csv(file_path):
    matrix = np.loadtxt(file_path, delimiter=";")
    return matrix


def create_adj_matrix(data):
    num_nodes = data.shape[0]
    adj_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                distance = np.sqrt(
                    (data[i, 0] - data[j, 0]) ** 2 + (data[i, 1] - data[j, 1]) ** 2
                )

                #! I'd prefer to keep these costs separately. In some algorithms we can potentially weight them etc.
                #! but maybe to be discussed?
                adj_matrix[i, j] = round(distance)  # + data[j, 2])

    return adj_matrix


def create_adj_matrix_np(data):
    diff = data[None, :, :2] - data[:, None, :2]
    return np.round(np.sqrt(np.einsum("ijk,ijk->ij", diff, diff)))


def create_adj_matrix_scipy(data):
    return np.round(cdist(data[:, :2], data[:, :2]))


def get_data():
    problems = [f"TSP{letter}" for letter in "ABCD"]
    data = {problem: None for problem in problems}
    for problem in data.keys():
        matrix = load_matrix_from_csv(f"data/{problem}.csv")
        data[problem] = {
            "original_data": matrix,
            "dist_matrix": create_adj_matrix(matrix),
            "nodes_cost": matrix[:, 2],
        }
    return data


if __name__ == "__main__":
    for id_ in ["A", "B", "C", "D"]:
        data = load_matrix_from_csv(f"TSP{id_}.csv")
        assert np.array_equal(create_adj_matrix_np(data), create_adj_matrix(data))
        assert np.array_equal(create_adj_matrix(data), create_adj_matrix_scipy(data))
