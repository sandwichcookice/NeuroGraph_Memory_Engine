import networkx as nx
import torch
from dataclasses import dataclass

@dataclass
class Transition:
    action: str
    weight: float

class MemoryGraph:
    def __init__(self, embedding_dim: int):
        self.graph = nx.DiGraph()
        self.embedding_dim = embedding_dim
        self.node_counter = 0

    def add_state(self, embedding: torch.Tensor) -> int:
        """新增狀態節點並記錄其年齡，避免新節點被過早刪除。"""
        node_id = self.node_counter
        self.graph.add_node(
            node_id,
            emb=embedding.detach().clone(),
            age=0,
        )
        self.node_counter += 1
        return node_id

    def add_transition(self, src: int, dst: int, action: str, weight: float):
        if self.graph.has_edge(src, dst):
            self.graph[src][dst]['weight'] += weight
        else:
            self.graph.add_edge(src, dst, action=action, weight=weight)

    def decay(self, gamma: float, threshold: float):
        """對邊權重衰減並移除舊的孤立節點。"""
        for u, v, data in list(self.graph.edges(data=True)):
            data['weight'] *= gamma
            if data['weight'] < threshold:
                self.graph.remove_edge(u, v)
        for n, node_data in list(self.graph.nodes(data=True)):
            node_data['age'] += 1
            if self.graph.degree(n) == 0 and node_data['age'] > 0:
                self.graph.remove_node(n)
