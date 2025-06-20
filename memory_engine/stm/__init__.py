import logging
from ..memory_graph import MemoryGraph

logger = logging.getLogger(__name__)

class ShortTermMemory(MemoryGraph):
    def __init__(self, embedding_dim: int, w0: float = 1.0, gamma: float = 0.9, threshold: float = 0.1):
        super().__init__(embedding_dim)
        self.w0 = w0
        self.gamma = gamma
        self.threshold = threshold

    def add_transition(self, src: int, dst: int, action: str):
        try:
            if self.graph.has_edge(src, dst):
                self.graph[src][dst]['weight'] += 1.0
            else:
                self.graph.add_edge(src, dst, action=action, weight=self.w0)
        except Exception as e:
            logger.exception("STM 新增邊失敗: %s", e)
            raise

    def decay_and_prune(self):
        super().decay(self.gamma, self.threshold)
