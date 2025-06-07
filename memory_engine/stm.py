from .memory_graph import MemoryGraph

class ShortTermMemory(MemoryGraph):
    def __init__(self, embedding_dim: int, w0: float = 1.0, gamma: float = 0.9, threshold: float = 0.1):
        super().__init__(embedding_dim)
        self.w0 = w0
        self.gamma = gamma
        self.threshold = threshold

    def add_transition(self, src: int, dst: int, action: str):
        if self.graph.has_edge(src, dst):
            self.graph[src][dst]['weight'] += 1.0
        else:
            self.graph.add_edge(src, dst, action=action, weight=self.w0)

    def decay_and_prune(self):
        super().decay(self.gamma, self.threshold)
