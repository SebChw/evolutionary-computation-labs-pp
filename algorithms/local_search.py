from copy import copy
from typing import List

import numpy as np

from algorithms.utils import calculate_path_cost

INTER = "inter"
INTRA = "intra"


# IMPORTANT: HERE I ASSUME i,j to be indices of nodes in the solution and not the nodes themselves, node_i,node_j are the nodes themselves, and neighbor_* is also node not an index
class LocalSearch:
    def __init__(self, greedy: bool = True, exchange_nodes: bool = True):
        """Performs local search algorithm

        Args:
            greedy (bool, optional): If True greedy approach is used otherwise steepest. Defaults to True.
            exchange_nodes (bool, optional): If True exchange nodes is used otherwise exchange edges. Defaults to True.
        """
        self.greedy_ = greedy
        self.exchange_nodes = exchange_nodes

    def __call__(
        self,
        adj_matrix: np.ndarray,
        nodes_cost: np.ndarray,
        initial_solution: List[int],
    ):
        """Performs local search algorithm

        Args:
            adj_matrix (np.ndarray): Adjacency matrix
            nodes_cost (np.ndarray): Cost of each node
            initial_solution (np.ndarray): Initial solution

        Returns:
            np.ndarray: Solution
        """
        self.adj_matrix = adj_matrix
        self.nodes_cost = nodes_cost
        self.solution = initial_solution

        # We generate pair of all indices to be checked just once. These are the same every time. As our solution length is fixed
        self.pairs_to_check_intra = self.generate_pairs_to_check(
            len(self.solution), len(self.solution)
        )
        # n_solution x m_not_solution
        self.pairs_to_check_inter = self.generate_pairs_to_check(
            len(self.solution), len(self.adj_matrix) - len(self.solution)
        )

        # two options for intra neigh exchange - nodes or edges
        if self.exchange_nodes:
            self.intra_neigh_deltas = self.two_nodes_exchange_deltas
            self.intra_neigh_exchange = self.two_nodes_exchange
        else:
            self.intra_neigh_deltas = self.two_edges_exchange_deltas
            self.intra_neigh_exchange = self.two_edges_exchange

        # either greedy or steepest algorithm
        if self.greedy_:
            self.greedy()
        else:
            self.steepest()

        # print(
        #     f"Final solution cost: {calculate_path_cost(self.solution, self.adj_matrix, self.nodes_cost)}"
        # )
        return self.solution

    def generate_pairs_to_check(self, n_indices: int, m_indices: int):
        """Generates all solution indices pairs to check if our solution is [10,20,30] -> [(1,0), (2,0), (2,1)] Right element is greater

        Args:
            n_indices (int): basically length of the solution

        Returns:
            np.ndarray: All indices pairs to check
        """
        # Create triangluar matrix and get it's indices We can omit if inside the loop thanks to it.
        return np.vstack(np.where(np.tri(n_indices, m_indices, k=-1))).T

    def two_nodes_exchange_deltas(self, nodes: List[int]):
        """Calculates deltas for every possible two nodes exchange

        Args:
            nodes (List[int]): Current solution

        Yields:
            Tuple: first_solution_index, second_solution_index, delta of this exchange
        """
        n_indices = len(nodes)
        for i, j in self.pairs_to_check_intra:
            # Get actual nodes and their neighbors
            node_i, node_j = nodes[i], nodes[j]
            neighbor_i_l, neighbor_i_r = nodes[i -
                                               1], nodes[(i + 1) % n_indices]
            neighbor_j_l, neighbor_j_r = nodes[j -
                                               1], nodes[(j + 1) % n_indices]

            # This operation changes 4 edges, we don't need to consider nodes costs
            curr_len = (
                self.adj_matrix[node_i, neighbor_i_l]
                + self.adj_matrix[node_i, neighbor_i_r]
                + self.adj_matrix[node_j, neighbor_j_l]
                + self.adj_matrix[node_j, neighbor_j_r]
            )
            len_after_change = (
                self.adj_matrix[node_i, neighbor_j_l]
                + self.adj_matrix[node_i, neighbor_j_r]
                + self.adj_matrix[node_j, neighbor_i_l]
                + self.adj_matrix[node_j, neighbor_i_r]
            )

            #! If nodes are neighbors. This is a special case!
            if abs(i - j) == 1 or (i == 99 and j == 0):
                curr_len -= self.adj_matrix[node_i, node_j]
                len_after_change += self.adj_matrix[node_i, node_j]

            yield i, j, curr_len - len_after_change

    def two_nodes_exchange(self, i: int, j: int, nodes: List[int]) -> List[int]:
        """Performs two nodes exchange"""
        nodes[i], nodes[j] = nodes[j], nodes[i]
        return nodes

    def two_edges_exchange_deltas(self, nodes: List[int]):
        """Calculates deltas for every possible two edges exchange"""

        n_indices = len(nodes)
        #! This is very important to iterate in this order this assures that i > j
        for j, i in self.pairs_to_check_intra:
            node_i, node_j = nodes[i], nodes[j]
            neighbor_i_r = nodes[(i + 1) % n_indices]
            neighbor_j_r = nodes[(j + 1) % n_indices]
            curr_len = (
                self.adj_matrix[node_i, neighbor_i_r]
                + self.adj_matrix[node_j, neighbor_j_r]
            )
            len_after_change = (
                self.adj_matrix[node_i, node_j]
                + self.adj_matrix[neighbor_i_r, neighbor_j_r]
            )

            yield i, j, curr_len - len_after_change

    def two_edges_exchange(self, i: int, j: int, nodes: List[int]) -> List[int]:
        #! We assume i > j
        # From the start to i (exlusive) it stays the same. Then I connect i with j. Next I need to go to the right neighor of i but in reversed order. Finally I add the rest of the nodes
        return nodes[: i + 1] + [nodes[j]] + nodes[j - 1: i: -1] + nodes[j + 1:]

    def inter_route_exchange_deltas(self, nodes: List[int]):
        # Find non selected nodes
        not_selected = list(set(range(self.adj_matrix.shape[0])) - set(nodes))

        # Evaluate all pairs
        for i, j in self.pairs_to_check_inter:
            node_i, node_j = nodes[i], not_selected[j]
            neighbor_i_l, neighbor_i_r = nodes[i -
                                               1], nodes[(i + 1) % len(nodes)]

            # Now we just compute distances between node_i and if we subtract it with node_j
            curr_len = (
                self.adj_matrix[node_i, neighbor_i_l]
                + self.adj_matrix[node_i, neighbor_i_r]
                + self.nodes_cost[node_i]
            )
            len_after_change = (
                self.adj_matrix[node_j, neighbor_i_l]
                + self.adj_matrix[node_j, neighbor_i_r]
                + self.nodes_cost[node_j]
            )

            yield i, node_j, curr_len - len_after_change

    def inter_route_exchange(self, i, node_j, nodes: List[int]) -> List[int]:
        nodes[i] = node_j
        return nodes

    def greedy(
        self,
    ):
        #! Uncomment for debuggign
        # old_cost = calculate_path_cost(self.solution, self.adj_matrix, self.nodes_cost)
        # print(
        #     f"Current solution cost: {calculate_path_cost(self.solution, self.adj_matrix, self.nodes_cost)}"
        # )
        while True:
            # Assure order of checks is different every time, shuffle is done w.r.t first axis so pairs are not mixed
            np.random.shuffle(self.pairs_to_check_inter)
            np.random.shuffle(self.pairs_to_check_intra)

            # Initialize generators that will yield suggested exchanges
            inter_route_deltas = self.inter_route_exchange_deltas(
                copy(self.solution))
            intra_route_deltas = self.intra_neigh_deltas(copy(self.solution))

            # Select when we will try to perform intra or inter route exchange
            n_intra = len(self.pairs_to_check_intra)
            n_inter = len(self.pairs_to_check_inter)
            choices = np.concatenate([np.ones(n_inter), np.zeros(n_intra)])
            np.random.shuffle(choices)

            # Iterate over choices and perform best exchange
            best_delta = 0
            best_move = None
            for choice in choices:
                if choice == 1:
                    # Perform inter route exchange
                    i, node_j, delta = next(inter_route_deltas)
                    if delta > best_delta:
                        best_delta = delta
                        best_move = i, node_j, INTER
                        break
                else:
                    # Perform intra route exchange
                    i, j, delta = next(intra_route_deltas)
                    if delta > best_delta:
                        best_delta = delta
                        best_move = i, j, INTRA
                        break

            if best_move is None:
                break
            else:
                #! Uncomment this stuff for debugging
                # old_solution = copy(self.solution)
                self.update_solution(best_move)
                # new_cost = calculate_path_cost(
                #     self.solution, self.adj_matrix, self.nodes_cost
                # )
                # print(f"New solution cost: {new_cost}")
                # print(f"Delta: {best_delta}")
                # print(f"Improvement: {old_cost - new_cost}")
                # if new_cost != old_cost - best_delta:
                #     raise Exception("Costs are not equal")
                # old_cost = new_cost

    def steepest(
        self,
    ):
        while True:
            best_delta = 0
            best_move = None
            # At first check all intra exchanges
            for i, node_j, delta in self.inter_route_exchange_deltas(
                copy(self.solution)
            ):
                if delta > best_delta:
                    best_delta = delta
                    best_move = i, node_j, INTER
            # Then all inter
            for i, j, delta in self.intra_neigh_deltas(copy(self.solution)):
                if delta > best_delta:
                    best_delta = delta
                    best_move = i, j, INTRA

            # Update best solution if possible
            if best_move is None:
                break
            else:
                self.update_solution(best_move)

    def update_solution(self, best_move):
        type_ = best_move[2]
        if type_ == INTER:
            i, node_j = best_move[:2]
            self.solution = self.inter_route_exchange(i, node_j, self.solution)
        else:
            i, j = best_move[:2]
            self.solution = self.intra_neigh_exchange(i, j, self.solution)
