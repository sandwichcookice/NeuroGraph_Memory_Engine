import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class AttentionScheduler(nn.Module):
    """根據查詢狀態在多個記憶單元間分配注意力。"""

    def __init__(self, state_dim: int):
        super().__init__()
        self.W_Q = nn.Linear(state_dim, state_dim)
        self.W_K = nn.Linear(state_dim, state_dim)
        self.W_V = nn.Linear(state_dim, state_dim)

    def forward(self, state: torch.Tensor, units: torch.Tensor):
        """選取最具相關性的記憶單元並回傳其表示與索引。"""
        try:
            if units.ndim != 2 or units.size(1) != state.size(0):
                raise ValueError("units shape mismatch")
            q = self.W_Q(state)
            k = self.W_K(units)
            v = self.W_V(units)
            scores = torch.matmul(q, k.t()) / (state.size(0) ** 0.5)
            attn = F.softmax(scores, dim=-1)
            best_idx = torch.argmax(attn).item()
            return v[best_idx], best_idx
        except Exception as e:
            logger.exception("注意力調度失敗: %s", e)
            raise
