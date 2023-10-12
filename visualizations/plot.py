import matplotlib.pyplot as plt

def plot_solution(matrix: list[list[int]], best_solutions: dict, problem_name: str, algorithm_name: str):
    solution = best_solutions[problem_name][algorithm_name]
    nodes = solution['nodes']

    x_coords = [matrix[node][0] for node in nodes]
    y_coords = [matrix[node][1] for node in nodes]
    costs = [matrix[node][2] for node in nodes]

    scaled_costs = [cost*10 for cost in costs]

    plt.scatter(x_coords, y_coords, s=scaled_costs, c='blue', alpha=0.6)

    for i in range(len(nodes) - 1):
        x1, y1 = matrix[nodes[i]][:2]
        x2, y2 = matrix[nodes[i+1]][:2]
        plt.plot([x1, x2], [y1, y2], color='gray', alpha=0.6)

    plt.title(f"Visualization for {problem_name} using {algorithm_name}")
    plt.show()

