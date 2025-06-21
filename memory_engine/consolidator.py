from .stm import ShortTermMemory
from .ltm import LongTermMemory
from .ltm_gnn import GNNLongTermMemory
import torch

class Consolidator:
    def __init__(self, beta: float = 0.1, gamma: float = 0.99, threshold: float = 0.01):
        self.beta = beta
        self.gamma = gamma
        self.threshold = threshold

    def run(self, stm: ShortTermMemory, ltm: LongTermMemory | GNNLongTermMemory):
        if isinstance(ltm, GNNLongTermMemory):
            self._run_gnn(stm, ltm)
        else:
            ltm.consolidate(stm, self.beta)
            stm.decay_and_prune()
            ltm.decay_and_prune()

    def _run_gnn(self, stm: ShortTermMemory, ltm: GNNLongTermMemory):
        """將 STM 內容整合進 GNN LTM。"""
        for n, d in stm.graph.nodes(data=True):
            if not ltm.graph.has_node(n):
                ltm.add_state(d['emb'], node_id=n)
        for u, v, data in stm.graph.edges(data=True):
            ltm.add_edge(u, v, data.get('action', ''))
            key = f"{u}->{v}"
            w = ltm.edge_params[key]
            w.data += self.beta * torch.tensor(data.get('weight', 1.0))
            ltm.graph[u][v]['weight'] = float(w.item())
        stm.decay_and_prune()
        ltm.decay_and_prune(self.gamma, self.threshold)
