import enum
from copy import copy
from itertools import product
from queue import PriorityQueue
from typing import List

import numpy as np

from algorithms.utils import calculate_path_cost

INTER = "inter"
INTRA = "intra"


class LSStrategy(enum.Enum):
    GREEDY = 1
    STEEPEST = 2
    STEEPEST_CANDIDATE = 3
    STEEPEST_MOVE_EVAL = 4


class LMCase(enum.Enum):
    NOT_EXISTING = 0
    REVERSED = 1
    NORMAL = 2


# IMPORTANT: HERE I ASSUME i,j to be indices of nodes in the solution and not the nodes themselves, node_i,node_j are the nodes themselves, and neighbor_* is also node not an index
class LocalSearch:
    def __init__(
        self,
        strategy: LSStrategy = LSStrategy.GREEDY,
        exchange_nodes: bool = True,
    ):
        """Performs local search algorithm

        Args:
            greedy (LSStrategy, optional): What LS strategy to use. Defaults to GREEDY.
            exchange_nodes (bool, optional): If True exchange nodes is used otherwise exchange edges. Defaults to True.
        """
        self.strategy = strategy
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
        self.pairs_to_check_inter = np.array(
            list(
                product(
                    range(len(self.solution)),
                    range(len(self.adj_matrix) - len(self.solution)),
                )
            )
        )

        # two options for intra neigh exchange - nodes or edges
        if self.exchange_nodes:
            self.intra_neigh_deltas = self.two_nodes_exchange_deltas
            self.intra_neigh_exchange = self.two_nodes_exchange
        else:
            self.intra_neigh_deltas = self.two_edges_exchange_deltas
            self.intra_neigh_exchange = self.two_edges_exchange

        self.ALL_INDICES_SET = set(range(len(self.adj_matrix)))

        # either greedy or steepest algorithm
        if self.strategy == LSStrategy.GREEDY:
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
            neighbor_i_l, neighbor_i_r = nodes[i - 1], nodes[(i + 1) % n_indices]
            neighbor_j_l, neighbor_j_r = nodes[j - 1], nodes[(j + 1) % n_indices]

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

    def compute_distances_intra(self, node_i, node_j, neighbor_i_r, neighbor_j_r):
        curr_len = (
            self.adj_matrix[node_i, neighbor_i_r]
            + self.adj_matrix[node_j, neighbor_j_r]
        )
        len_after_change = (
            self.adj_matrix[node_i, node_j]
            + self.adj_matrix[neighbor_i_r, neighbor_j_r]
        )
        return curr_len, len_after_change

    def compute_delta_edges_correct(self, i, j):
        if i < j:
            node_i, node_j = self.solution[i], self.solution[j]
            neighbor_i_r = self.solution[(i + 1) % len(self.solution)]
            neighbor_j_r = self.solution[(j + 1) % len(self.solution)]
            curr_len, len_change = self.compute_distances_intra(
                node_i, node_j, neighbor_i_r, neighbor_j_r
            )

        else:
            node_i, node_j = self.solution[j], self.solution[i]
            neighbor_i_l = self.solution[(j - 1) % len(self.solution)]
            neighbor_j_l = self.solution[(i - 1) % len(self.solution)]
            curr_len, len_change = self.compute_distances_intra(
                node_i, node_j, neighbor_i_l, neighbor_j_l
            )

        return curr_len - len_change

    def two_edges_exchange_deltas(self, nodes: List[int]):
        """Calculates deltas for every possible two edges exchange"""
        for j, i in self.pairs_to_check_intra:
            yield i, j, self.compute_delta_edges_correct(i, j)
            yield j, i, self.compute_delta_edges_correct(j, i)

    def two_edges_exchange(self, i: int, j: int, nodes: List[int]) -> List[int]:
        #! It seems that we must allow i > j to fulfill 2 options
        if i < j:
            # From the start to i (exlusive) it stays the same. Then I connect i with j. Next I need to go to the right neighor of i but in reversed order. Finally I add the rest of the nodes
            # in this case i remove edges to right neighbors
            return nodes[: i + 1] + [nodes[j]] + nodes[j - 1 : i : -1] + nodes[j + 1 :]
        else:
            # This kind of reverse order, i > j
            # From end to i in reversed order -> i to j -> from j to the left neighbour of i in correct order -> from left neighbour of i to the left of j and to the beginning in reversed order
            # Here I remove edges to left neighbors
            solution = (
                nodes[-1 : i - 1 : -1]
                + [nodes[j]]
                + nodes[j + 1 : i]
                + nodes[max(j - 1, 0) : 0 : -1]
            )
            if j != 0:
                solution.append(nodes[0])
            return solution

    def compute_distances_inter(self, node_i, node_j, neighbor_i_l, neighbor_i_r):
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

        return curr_len, len_after_change

    def inter_route_exchange_deltas(self, nodes: List[int]):
        # Find non selected nodes
        not_selected = list(set(range(self.adj_matrix.shape[0])) - set(nodes))

        # Evaluate all pairs
        for i, j in self.pairs_to_check_inter:
            node_i, node_j = nodes[i], not_selected[j]
            neighbor_i_l, neighbor_i_r = nodes[i - 1], nodes[(i + 1) % len(nodes)]

            # Now we just compute distances between node_i and if we subtract it with node_j
            curr_len, len_after_change = self.compute_distances_inter(
                node_i, node_j, neighbor_i_l, neighbor_i_r
            )

            yield i, node_j, curr_len - len_after_change

    def inter_route_exchange(self, i, node_j, nodes: List[int]) -> List[int]:
        nodes[i] = node_j
        return nodes

    def greedy(
        self,
    ):
        while True:
            # Assure order of checks is different every time, shuffle is done w.r.t first axis so pairs are not mixed
            np.random.shuffle(self.pairs_to_check_inter)
            np.random.shuffle(self.pairs_to_check_intra)

            # Initialize generators that will yield suggested exchanges
            inter_route_deltas = self.inter_route_exchange_deltas(copy(self.solution))
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
                self.update_solution(best_move)

    def get_neighbors(self, node_id):
        return (
            self.solution[node_id - 1],
            self.solution[(node_id + 1) % len(self.solution)],
        )

    def add_priority_inter(self, i, node_j, delta):
        left_n, right_n = self.get_neighbors(i)
        #! For every move we need to know exactly what edges would be added and what removed
        #! If removed nodes are no longer in the solution -> We can't apply it
        removed = [(left_n, self.solution[i]), (self.solution[i], right_n)]

        element = (
            -delta,
            (
                {
                    "removed": removed,
                    "move": (i, node_j, INTER),
                    "change_type": INTER,
                }
            ),
        )
        self.prioritized_moves.put(element)

    def add_priority_intra(self, i, j, delta):
        left_i, right_i = self.get_neighbors(i)
        left_j, right_j = self.get_neighbors(j)
        node_i, node_j = self.solution[i], self.solution[j]
        if i > j:
            #
            removed = [(left_i, node_i), (left_j, node_j)]
            # TBD
            # added =
        else:
            removed = [(node_i, right_i), (node_j, right_j)]
            # TBD
            # added = These will actually dynamically change. Just the removed nodes stay the same. But this means delta will change too.

        #! I can't use this "move" I must operate on actual nodes
        element = (
            -delta,
            ({"removed": removed, "move": (i, j, INTRA), "change_type": INTRA}),
        )
        self.prioritized_moves.put(element)

    def update_node_to_position(self):
        # We need to know where each node is in the solution
        self.node_to_position = np.full(len(self.adj_matrix), -1)
        for i, node in enumerate(self.solution):
            self.node_to_position[node] = i

    def prepare_for_move_eval(self):
        self.prioritized_moves = PriorityQueue()
        self.update_node_to_position()

        # At first we need to evaluate all moves. Leter we will just update this list with newest
        #! We can't operate on abstract indices now, we must operate on real nodes and edges
        for i, node_j, delta in self.inter_route_exchange_deltas(self.solution):
            if delta > 0:
                self.add_priority_inter(i, node_j, delta)

        for i, j, delta in self.intra_neigh_deltas(self.solution):
            if delta > 0:
                self.add_priority_intra(i, j, delta)

    def get_removed_case(self, removed, change_type):
        for edge in removed:
            node_i, node_j = edge
            if (
                self.node_to_position[node_i] == -1
                or self.node_to_position[node_j] == -1
            ):
                return LMCase.NOT_EXISTING

            node_i_position = self.node_to_position[node_i]
            node_j_position = self.node_to_position[node_j]
            if abs(node_i_position - node_j_position) > 1:
                return LMCase.NOT_EXISTING

        # We assume that these edges are in the same direction as in the solution
        relative = (True, True)
        for i, edge in enumerate(removed):
            node_i, node_j = edge
            node_i_position = self.node_to_position[node_i]
            node_j_position = self.node_to_position[node_j]
            # We know these are neighbours but which is first
            if node_i_position > node_j_position:
                relative[i] = False

        if sum(relative) == 2:
            return LMCase.NORMAL
        # IF edge order changes in INTRA delta will change also!
        elif relative[0] == relative[1] and change_type == INTER:
            return LMCase.NORMAL
        else:
            return LMCase.REVERSED

    def add_new_to_priorietized_moves(self, best_move):
        change_type = best_move[2]
        not_selected = self.ALL_INDICES_SET - set(self.solution)
        if change_type == INTER:
            i = best_move[0]

            #! ADDING NEW INTER MOVES - but is it actually necessar. No other node exchange would bring better delta back.
            # node_i = self.solution[i]
            # neighbor_i_l, neighbor_i_r = (
            #     self.solution[i - 1],
            #     self.solution[(i + 1) % len(self.solution)],
            # )

            # for node_j in not_selected:
            #     curr_len, len_after_change = self.compute_distances_inter(
            #         node_i, node_j, neighbor_i_l, neighbor_i_r
            #     )
            #     delta = curr_len - len_after_change
            #     if delta > 0:
            #         self.add_priority_inter(i, node_j, delta)
            #! ADDING NEW INTRA MOVES
            for j in range(len(self.solution)):
                if abs(i - j) <= 1 or (i == 99 and j == 0):
                    continue
                delta_i_j = self.compute_delta_edges_correct(i, j)
                if delta_i_j > 0:
                    self.add_priority_intra(i, j, delta_i_j)
                delta_j_i = self.compute_delta_edges_correct(j, i)
                if delta_j_i > 0:
                    self.add_priority_intra(j, i, delta_j_i)
        else:
            #! ADDING NEW INTER MOVES we didn't introduce new node. However deltas have changes
            for node_id in best_move[:2]:
                node_i = self.solution[node_id]
                neighbor_i_l, neighbor_i_r = (
                    self.solution[node_id - 1],
                    self.solution[(node_id + 1) % len(self.solution)],
                )

            for node_j in not_selected:
                curr_len, len_after_change = self.compute_distances_inter(
                    node_i, node_j, neighbor_i_l, neighbor_i_r
                )
                delta = curr_len - len_after_change
                if delta > 0:
                    self.add_priority_inter(i, node_j, delta)

            #! ADDING NEW INTRA MOVES - again not necessary as only new possible option is to reverse what we have done?

    def steepest(
        self,
    ):
        # Preparations before strategy
        if self.strategy == LSStrategy.STEEPEST_CANDIDATE:
            self.construct_candidate_list()

        if self.strategy == LSStrategy.STEEPEST_MOVE_EVAL:
            self.prepare_for_move_eval()

        # Running strategies
        while True:
            best_delta = 0
            best_move = None

            if self.strategy == LSStrategy.STEEPEST_CANDIDATE:
                # Check all inter-route exchanges with candidate moves
                not_selected = self.ALL_INDICES_SET - set(self.solution)

                for i, node_i in enumerate(self.solution):
                    for node_j in self.candidate_list[node_i].intersection(
                        not_selected
                    ):
                        neighbor_i_l, neighbor_i_r = (
                            self.solution[i - 1],
                            self.solution[(i + 1) % len(self.solution)],
                        )
                        neighbor_l_l, neighbor_r_r = (
                            self.solution[i - 2],
                            self.solution[(i + 2) % len(self.solution)],
                        )

                        curr_len_l, len_after_replace_l = self.compute_distances_inter(
                            neighbor_i_l, node_j, neighbor_l_l, node_i
                        )

                        curr_len_r, len_after_replace_r = self.compute_distances_inter(
                            neighbor_i_r, node_j, node_i, neighbor_r_r
                        )

                        delta_l = curr_len_l - len_after_replace_l
                        delta_r = curr_len_r - len_after_replace_r

                        if delta_l > delta_r and delta_l > best_delta:
                            best_delta = delta_l
                            best_move = (i - 1, node_j, INTER)
                        elif delta_r > best_delta:
                            best_delta = delta_r
                            best_move = ((i + 1) % len(self.solution), node_j, INTER)
                # INTRA
                for i, node_i in enumerate(self.solution):
                    for j, node_j in enumerate(self.solution):
                        if node_j in self.candidate_list[node_i]:
                            delta_i_to_j = self.compute_delta_edges_correct(i, j)
                            delta_j_to_i = self.compute_delta_edges_correct(j, i)

                            if (
                                delta_i_to_j > delta_j_to_i
                                and delta_i_to_j > best_delta
                            ):
                                best_delta = delta_i_to_j
                                best_move = i, j, INTRA
                            elif delta_j_to_i > best_delta:
                                best_delta = delta_j_to_i
                                best_move = j, i, INTRA

            elif self.strategy == LSStrategy.STEEPEST_MOVE_EVAL:
                to_put_back = []
                while self.prioritized_moves.empty():
                    delta, move_dict = self.prioritized_moves.get()
                    removed, move = move_dict["removed"], move_dict["move"]
                    change_type = move_dict["change_type"]

                    lm_case = self.get_removed_case(removed, change_type)
                    # Check if move is not applicable
                    if lm_case == LMCase.NOT_EXISTING:
                        continue
                    elif lm_case == LMCase.REVERSED:
                        to_put_back.append((delta, move_dict))
                    # Check if move may be applicable
                    else:
                        best_move = move
                        break

                # Maybe in the next iteration we use them
                for to_put in to_put_back:
                    self.prioritized_moves.put(to_put)

            else:
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

            if self.strategy == LSStrategy.STEEPEST_MOVE_EVAL:
                self.add_new_to_priorietized_moves(best_move)
                self.update_node_to_position()

    def update_solution(self, best_move):
        type_ = best_move[2]
        if type_ == INTER:
            i, node_j = best_move[:2]
            self.solution = self.inter_route_exchange(i, node_j, self.solution)
        else:
            i, j = best_move[:2]
            self.solution = self.intra_neigh_exchange(i, j, self.solution)

    def construct_candidate_list(self):
        self.candidate_list = {}
        for i in range(len(self.adj_matrix)):
            costs = self.adj_matrix[i] + self.nodes_cost
            sorted_indices = np.argsort(costs)

            #! Due to nodes cost there may be nodes where itself has bigger cost than it's neighbour
            candidates = set()
            for idx in sorted_indices:
                if idx != i:
                    candidates.add(idx)
                if len(candidates) == 10:
                    break

            self.candidate_list[i] = candidates
