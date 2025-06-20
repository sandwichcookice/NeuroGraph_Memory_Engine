import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class ActionDecoder(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return self.net(x)
        except Exception as e:
            logger.exception("動作解碼失敗: %s", e)
            raise
