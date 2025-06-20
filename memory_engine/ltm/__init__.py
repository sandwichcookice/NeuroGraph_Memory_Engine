import logging
from ..memory_graph import MemoryGraph

logger = logging.getLogger(__name__)

class LongTermMemory(MemoryGraph):
    def __init__(self, embedding_dim: int, gamma: float = 0.99, threshold: float = 0.01):
        super().__init__(embedding_dim)
        self.gamma = gamma
        self.threshold = threshold

    def consolidate(self, stm: "ShortTermMemory", beta: float = 0.1):
        try:
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
        except Exception as e:
            logger.exception("LTM 鞏固失敗: %s", e)
            raise

    def decay_and_prune(self):
        super().decay(self.gamma, self.threshold)
