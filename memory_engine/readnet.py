import torch
import torch.nn as nn
import torch.nn.functional as F
from .stm import ShortTermMemory
from .ltm import LongTermMemory

class ReadNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def stm_attention(self, x_t: torch.Tensor, stm: ShortTermMemory) -> torch.Tensor:
        if not stm.graph:
            return torch.zeros_like(x_t)
        weights = []
        values = []
        for _, dst, data in stm.graph.out_edges(stm.node_counter - 1, data=True):
            dst_emb = stm.graph.nodes[dst]['emb']
            score = torch.dot(x_t, dst_emb)
            weights.append(score * data['weight'])
            values.append(dst_emb)
        if not weights:
            return torch.zeros_like(x_t)
        w = F.softmax(torch.stack(weights), dim=0)
        v = torch.stack(values)
        return torch.sum(w.unsqueeze(1) * v, dim=0)

    def ltm_message(self, x_t: torch.Tensor, ltm: LongTermMemory) -> torch.Tensor:
        if not ltm.graph:
            return torch.zeros_like(x_t)
        agg = torch.zeros_like(x_t)
        for _, dst, data in ltm.graph.out_edges(ltm.node_counter - 1, data=True):
            dst_emb = ltm.graph.nodes[dst]['emb']
            agg += data['weight'] * dst_emb
        return agg

    def forward(self, x_t: torch.Tensor, stm: ShortTermMemory, ltm: LongTermMemory) -> torch.Tensor:
        r_s = self.stm_attention(x_t, stm)
        r_l = self.ltm_message(x_t, ltm)
        fused = torch.cat([x_t, r_s, r_l], dim=0).unsqueeze(0)
        return self.mlp(fused).squeeze(0)
