import json
from collections import defaultdict

from data.data_parser import get_data
from visualize import plot_solution


def get_result(data: dict, key: str = 'cost'):
    costs = defaultdict(lambda: defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))))
    results = {'best': costs.copy(), 'worst': costs.copy(),
               'avg': costs.copy()}
    best_nodes = defaultdict(lambda: defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))))

    for problem, problem_results in data.items():
        for ls_type, ls_results in problem_results.items():
            for exchange_type, exchange_results in ls_results.items():
                for starting_solution, starting_solution_results in exchange_results.items():
                    for result in starting_solution_results:
                        cost = result['solution'][key]
                        nodes = result['solution']['nodes']
                        if results['best'][problem][ls_type][exchange_type][starting_solution] == 0:
                            results['best'][problem][ls_type][exchange_type][starting_solution] = cost
                            results['worst'][problem][ls_type][exchange_type][starting_solution] = cost
                            best_nodes[problem][ls_type][exchange_type][starting_solution] = nodes
                        if cost < results['best'][problem][ls_type][exchange_type][starting_solution]:
                            results['best'][problem][ls_type][exchange_type][starting_solution] = cost
                            best_nodes[problem][ls_type][exchange_type][starting_solution] = nodes
                        if cost > results['worst'][problem][ls_type][exchange_type][starting_solution]:
                            results['worst'][problem][ls_type][exchange_type][starting_solution] = cost
                        results['avg'][problem][ls_type][exchange_type][starting_solution] += cost / 200

    return results, best_nodes


def get_time_results(data):
    x = defaultdict(lambda: defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))))
    results = {'best': x.copy(), 'worst': x.copy(), 'avg': x.copy()}
    for problem, problem_results in data.items():
        for ls_type, ls_results in problem_results.items():
            for exchange_type, exchange_results in ls_results.items():
                for starting_solution, starting_solution_results in exchange_results.items():
                    for result in starting_solution_results:
                        cost = result['total_time']
                        results['best'][problem][ls_type][exchange_type][starting_solution] = min(
                            cost, results['best'][problem][ls_type][exchange_type].get(starting_solution, float('inf')))
                        results['worst'][problem][ls_type][exchange_type][starting_solution] = max(
                            cost, results['worst'][problem][ls_type][exchange_type].get(starting_solution, float('-inf')))
                        results['avg'][problem][ls_type][exchange_type][starting_solution] += cost / 200
    return results


with open('solutions.json', 'r') as f:
    data = json.load(f)


cost_results, nodes = get_result(data)
# time_results = get_time_results(data)
print(nodes['TSPA']['steepest']['edges']['random'])

for problem, problem_results in nodes.items():
    matrix = get_data()[problem]['original_data']
    for ls_type, ls_results in problem_results.items():
        for exchange_type, exchange_results in ls_results.items():
            for starting_solution, nodes in exchange_results.items():
                plot_solution(matrix, (nodes, cost_results['best'][problem][ls_type][exchange_type][starting_solution]), problem,
                              f'{ls_type}_{exchange_type}_{starting_solution}')

# print(json.dumps(cost_results, indent=4))
# print(json.dumps(time_results, indent=4))
