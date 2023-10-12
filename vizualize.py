import json
from data.data_parser import get_data
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate


def get_best_values(solutions: dict, problem: str):
    best_values = {}
    for algorithm_name, algorithm_solutions in solutions[problem].items():
        best_values[algorithm_name] = min(
            algorithm_solutions, key=lambda solution: solution['cost'])
    return best_values


def plot_solution(matrix: list[list[int]], best_solutions: dict, problem_name: str, algorithm_name: str):
    plt.figure(figsize=(50, 50))

    solution = best_solutions[problem_name][algorithm_name]
    nodes = solution['nodes']

    x_coords = [matrix[node][0] for node in nodes]
    y_coords = [matrix[node][1] for node in nodes]
    costs = [matrix[node][2] for node in nodes]

    normalized_costs = (costs - np.min(costs)) / (np.max(costs) - np.min(costs))

    plt.scatter(x_coords, y_coords, s=300, c=normalized_costs, cmap='Blues_r', alpha=0.6)

    for i in range(len(nodes) - 1):
        x1, y1 = matrix[nodes[i]][:2]
        x2, y2 = matrix[nodes[i+1]][:2]
        plt.plot([x1, x2], [y1, y2], color='gray', alpha=0.6)

    plt.title(f"Visualization for {problem_name} using {algorithm_name}. Value={solution['cost']}", fontsize=48)
    plt.xticks(fontsize=36)
    plt.yticks(fontsize=36)
    plt.colorbar().ax.tick_params(labelsize=36)

    plt.savefig(f"visualizations/{problem_name}_{algorithm_name}.png")


def compute_best_solution(algorithm_solutions: list) -> dict:
    best = min(algorithm_solutions, key=lambda solution: solution['cost'])
    return {'cost': best['cost'], 'nodes': best['nodes']}

def compute_worst_solution(algorithm_solutions: list) -> dict:
    worst = max(algorithm_solutions, key=lambda solution: solution['cost'])
    return {'cost': worst['cost'], 'nodes': worst['nodes']}

def compute_average_cost(algorithm_solutions: list) -> dict:
    average = sum(solution['cost'] for solution in algorithm_solutions) / len(algorithm_solutions)
    return {'cost': average, 'nodes': None}

def get_extremes(solutions: dict):
    best_solutions = {}
    worst_solutions = {}
    average_solutions = {}
    
    for problem, algorithms in solutions.items():
        best_solutions[problem] = {}
        worst_solutions[problem] = {}
        average_solutions[problem] = {}
        
        for algorithm_name, algorithm_solutions in algorithms.items():
            best_solutions[problem][algorithm_name] = compute_best_solution(algorithm_solutions)
            worst_solutions[problem][algorithm_name] = compute_worst_solution(algorithm_solutions)
            average_solutions[problem][algorithm_name] = compute_average_cost(algorithm_solutions)

    return best_solutions, worst_solutions, average_solutions


def create_table_data(best_solutions, worst_solutions, average_solutions):
    headers = ['Problem', 'Algorithm', 'Best Solution', 'Worst Solution', 'Average Solution']
    rows = []
    for problem_name in best_solutions.keys():
        for algorithm_name in best_solutions[problem_name].keys():
            best_cost = best_solutions[problem_name][algorithm_name]['cost']
            worst_cost = worst_solutions[problem_name][algorithm_name]['cost']
            average_cost = average_solutions[problem_name][algorithm_name]['cost']
            rows.append([problem_name, algorithm_name, best_cost, worst_cost, average_cost])
    return headers, rows

def save_results(best_solutions, worst_solutions, average_solutions):
    headers, data = create_table_data(best_solutions, worst_solutions, average_solutions)
    plt.figure(figsize=(10, 5))
    plt.axis('off')
    plt.table(cellText=data, colLabels=headers, loc='center')
    plt.savefig('visualizations/table.png')

def main():
    with open('solutions.json', 'r') as file:
        solutions = json.load(file)

    best_solutions, worst_solutions, average_solutions = get_extremes(solutions)

    data = get_data()
    for problem_name in data.keys():
        for algorithm_name in solutions[problem_name].keys():
            plot_solution(data[problem_name], best_solutions, problem_name, algorithm_name)

    save_results(best_solutions, worst_solutions, average_solutions)


if __name__ == '__main__':
    main()