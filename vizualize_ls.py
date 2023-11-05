import json

with open('solutions.json', 'r') as f:
    data = json.load(f)

results = {'best': {}, 'worst': {}, 'average': {}}
for key in results.keys():
    for problem, instance in data.items():
        results[key][problem] = {}
        for greedy_name, greedy in instance.items():
            results[key][problem][greedy_name] = {}
            for exchange_nodes_name, exchange_nodes in greedy.items():
                results[key][problem][greedy_name][exchange_nodes_name] = {}
                for starting_solution_name, starting_solution in exchange_nodes.items():
                    results[key][problem][greedy_name][exchange_nodes_name][starting_solution_name] = {
                    }
                    for solution in starting_solution:
                        if not results[key][problem][greedy_name][exchange_nodes_name][starting_solution_name]:
                            results[key][problem][greedy_name][exchange_nodes_name][starting_solution_name] = solution['solution']['cost']
                        else:
                            if solution['solution']['cost'] < results[key][problem][greedy_name][exchange_nodes_name][starting_solution_name]['solution']['cost']:
                                results[key][problem][greedy_name][exchange_nodes_name][starting_solution_name] = solution['solution']['cost']

for key in results.keys():
    for problem, instance in results[key].items():
        for greedy_name, greedy in instance.items():
            for exchange_nodes_name, exchange_nodes in greedy.items():
                for starting_solution_name, starting_solution in exchange_nodes.items():
                    print(
                        f"{key} {problem} {greedy_name} {exchange_nodes_name} {starting_solution_name} {starting_solution}")
