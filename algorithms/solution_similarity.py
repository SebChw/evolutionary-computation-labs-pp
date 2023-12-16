from typing import List, Set, Tuple


class SolutionSimilarity:
    def __init__(self, anchor: List):
        """
        Initialize the SolutionSimilarity class.

        Parameters:
        anchor (List): The anchor solution to which other solution will be compared
        """
        self.anchor = anchor

        self.anchor_edges = self._get_edges(self.anchor)
        self.anchor_nodes = self._get_nodes(self.anchor)

    def _get_edges(self, solution: List) -> Set[Tuple]:
        """
        Parameters:
        solution (List): The solution to get edges from

        Returns:
        Set: A set of all edges from the solution
        """
        edges = set(zip(solution[:-1], solution[1:]))
        last_edge = (solution[-1], solution[0])
        return edges.union({last_edge})

    def _get_nodes(self, solution: List) -> Set:
        """
        Parameters:
        solution (List): The solution to get nodes from

        Returns:
        Set: A set of all nodes from the solution
        """
        return set(solution)

    def num_common_edges(self, solution: List) -> int:
        """
        Parameters:
        solution (List): The solution to compare with the anchor

        Returns:
        int: The number of common edges between the anchor and the solution
        """
        solution_edges = self._get_edges(solution)
        return len(self.anchor_edges.intersection(solution_edges))

    def num_common_nodes(self, solution: List) -> int:
        """
        Parameters:
        solution (List): The solution to compare with the anchor

        Returns:
        int: The number of common nodes between the anchor and the solution
        """
        solution_nodes = self._get_nodes(solution)
        return len(self.anchor_nodes.intersection(solution_nodes))
