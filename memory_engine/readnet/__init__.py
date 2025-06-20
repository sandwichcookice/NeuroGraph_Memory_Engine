import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..stm import ShortTermMemory
from ..ltm import LongTermMemory

logger = logging.getLogger(__name__)

class ReadNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def stm_attention(self, x_t: torch.Tensor, stm: ShortTermMemory) -> torch.Tensor:
        """計算 STM 注意力，採向量化方式提升效率。"""
        if not stm.graph:
            return torch.zeros_like(x_t)
        edges = list(stm.graph.out_edges(stm.node_counter - 1, data=True))
        if not edges:
            return torch.zeros_like(x_t)
        embs = torch.stack([stm.graph.nodes[v]['emb'] for _, v, _ in edges])
        weights = torch.tensor([d['weight'] for _, _, d in edges], dtype=x_t.dtype)
        scores = embs @ x_t
        attn = F.softmax(scores * weights, dim=0)
        return torch.sum(attn.unsqueeze(1) * embs, dim=0)

    def ltm_message(self, x_t: torch.Tensor, ltm: LongTermMemory) -> torch.Tensor:
        """以動態注意力從 LTM 擷取訊息。"""
        if not ltm.graph:
            return torch.zeros_like(x_t)
        edges = list(ltm.graph.out_edges(ltm.node_counter - 1, data=True))
        if not edges:
            return torch.zeros_like(x_t)
        embs = torch.stack([ltm.graph.nodes[v]['emb'] for _, v, _ in edges])
        weights = torch.tensor([d['weight'] for _, _, d in edges], dtype=x_t.dtype)
        scores = embs @ x_t
        attn = F.softmax(scores * weights, dim=0)
        return torch.sum(attn.unsqueeze(1) * embs, dim=0)

    def forward(self, x_t: torch.Tensor, stm: ShortTermMemory, ltm: LongTermMemory) -> torch.Tensor:
        try:
            r_s = self.stm_attention(x_t, stm)
            r_l = self.ltm_message(x_t, ltm)
            fused = torch.cat([x_t, r_s, r_l], dim=0).unsqueeze(0)
            return self.mlp(fused).squeeze(0)
        except Exception as e:
            logger.exception("讀取網路失敗: %s", e)
            raise
