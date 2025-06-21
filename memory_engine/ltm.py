from .memory_graph import MemoryGraph

class LongTermMemory(MemoryGraph):
    def __init__(self, embedding_dim: int, gamma: float = 0.99, threshold: float = 0.01):
        super().__init__(embedding_dim)
        self.gamma = gamma
        self.threshold = threshold

    def consolidate(self, stm: "ShortTermMemory", beta: float = 0.1):
        for u, v, data in stm.graph.edges(data=True):
            if not self.graph.has_node(u):
                self.graph.add_node(u, emb=stm.graph.nodes[u]['emb'], age=0)
            if not self.graph.has_node(v):
                self.graph.add_node(v, emb=stm.graph.nodes[v]['emb'], age=0)
            weight = beta * data['weight']
            if self.graph.has_edge(u, v):
                self.graph[u][v]['weight'] += weight
            else:
                self.graph.add_edge(u, v, action=data['action'], weight=weight)

    def decay_and_prune(self):
        super().decay(self.gamma, self.threshold)

    def find_goal(self, item: str):
        """回傳包含指定物品的節點編號。若不存在則回傳 None。"""
        for n, d in self.graph.nodes(data=True):
            if d.get("goal") == item:
                return n
        return None
