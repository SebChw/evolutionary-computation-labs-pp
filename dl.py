import json
import random

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import one_hot

from data.data_parser import get_data


class TSPModelConv(nn.Module):
    def __init__(self, num_nodes):
        self.conv1 = nn.Conv1d(
            in_channels=num_nodes, out_channels=64, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1
        )
        self.glob_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(256, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.glob_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_data(with_cost: bool = True):
    with open("solutions_evo.json", "r") as f:
        data = json.load(f)
        data = pd.DataFrame(data)

    ml_data = {}

    for problem, problem_results in data.items():
        ml_data[problem] = []
        dist = get_data()[problem]["dist_matrix"]
        nodes_cost = get_data()[problem]["nodes_cost"]
        matrix = dist + nodes_cost
        matrix[dist == 0] = 0
        for result in problem_results:
            solution = result["solution"]
            solution = one_hot(torch.tensor(solution), num_classes=200)
            vector = solution.sum(dim=0)
            vector = torch.clamp(vector, 0, 1)
            combined_vect_matrix = torch.cat(
                (vector.unsqueeze(0), torch.tensor(matrix)), dim=0
            )
            if with_cost:
                ml_data[problem].append((combined_vect_matrix, result["cost"]))
            else:
                ml_data[problem].append((vector, result["cost"]))

    train_data, test_data = [], []
    for problem, problem_data in ml_data.items():
        for data_point in problem_data:
            combined_vect_matrix, cost = data_point
            if random.random() < 0.8:
                train_data.append((combined_vect_matrix, cost))
            else:
                test_data.append((combined_vect_matrix, cost))

    return train_data, test_data


def run_conv_model():
    train_data, test_data = get_data(with_cost=True)
    model = TSPModelConv(201)
    model.float()
    print(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    batch_size = 32

    losses = []

    for epoch in range(num_epochs):
        random.shuffle(train_data)

        for i in range(0, len(train_data), batch_size):
            batch_data = train_data[i : i + batch_size]
            batch_inputs = torch.stack([data[0] for data in batch_data])
            batch_targets = torch.tensor([data[1] for data in batch_data]).unsqueeze(1)
            batch_inputs = batch_inputs.float()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
        losses.append(loss.item())

    plt.plot(losses)
    plt.show()


train, test = get_data(with_cost=False)
# plo
