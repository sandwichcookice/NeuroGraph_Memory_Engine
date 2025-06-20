import time
import math
from typing import Optional

import torch

from .memory_graph import MemoryGraph

class ShortTermMemory(MemoryGraph):
    """短期記憶模組，依據連結強度 L/R/C 建立可衰退的記憶圖。"""

    def __init__(
        self,
        embedding_dim: int,
        alpha_L: float = 0.05,
        alpha_R: float = 0.05,
        alpha_C: float = 0.01,
        lambda_c: float = 0.9,
        theta_L: float = 0.05,
        theta_R: float = 0.05,
        theta_C: float = 0.01,
    ):
        super().__init__(embedding_dim)
        self.w = torch.randn(embedding_dim * 4)
        self.alpha_L = alpha_L
        self.alpha_R = alpha_R
        self.alpha_C = alpha_C
        self.lambda_c = lambda_c
        self.theta_L = theta_L
        self.theta_R = theta_R
        self.theta_C = theta_C

    def add_state(self, embedding: torch.Tensor, context_id: int = 0, text: str = "") -> int:
        """新增節點並紀錄時間、上下文與原始文字。"""
        node_id = super().add_state(embedding)
        data = self.graph.nodes[node_id]
        data["ts"] = time.time()
        data["context"] = context_id
        data["text"] = text
        return node_id

    def _compute_L(self, emb_i: torch.Tensor, emb_j: torch.Tensor) -> float:
        feat = torch.cat([emb_i, emb_j, emb_i * emb_j, torch.abs(emb_i - emb_j)])
        return torch.sigmoid(torch.dot(self.w, feat)).item()

    def add_transition(self, src: int, dst: int, action: str, reward: float = 0.0):
        if not self.graph.has_node(src) or not self.graph.has_node(dst):
            raise ValueError("節點不存在")
        emb_i = self.graph.nodes[src]["emb"]
        emb_j = self.graph.nodes[dst]["emb"]
        L = self._compute_L(emb_i, emb_j)
        R = reward * L
        now = time.time()
        if self.graph.has_edge(src, dst):
            e = self.graph[src][dst]
            e["L"] = L
            e["R"] = R
            e["C"] = self.lambda_c * e.get("C", 1.0) + (1 - self.lambda_c)
            e["weight"] = e["L"] + e["R"] + e["C"]
            e["last_update"] = now
        else:
            self.graph.add_edge(
                src,
                dst,
                action=action,
                L=L,
                R=R,
                C=1.0,
                weight=L + R + 1.0,
                last_update=now,
            )

    def decay_and_prune(self, current_time: Optional[float] = None):
        """根據時間衰退邊權重並清除過低的連結與孤立節點。"""
        if current_time is None:
            current_time = time.time()
        for u, v, data in list(self.graph.edges(data=True)):
            dt = current_time - data.get("last_update", current_time)
            data["L"] *= math.exp(-self.alpha_L * dt)
            data["R"] *= math.exp(-self.alpha_R * dt)
            data["C"] *= math.exp(-self.alpha_C * dt)
            data["weight"] = data["L"] + data["R"] + data["C"]
            data["last_update"] = current_time
            if (
                data["L"] <= self.theta_L
                and data["R"] <= self.theta_R
                and data["C"] <= self.theta_C
            ):
                self.graph.remove_edge(u, v)
        for n, d in list(self.graph.nodes(data=True)):
            d["age"] += 1
            if self.graph.degree(n) == 0 and d["age"] > 0:
                self.graph.remove_node(n)

    def visualize(self, path: str):
        """輸出 STM 結構圖，失敗時丟出例外。"""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx

            plt.figure(figsize=(6, 4))
            pos = nx.spring_layout(self.graph)
            nx.draw(self.graph, pos, with_labels=True, node_color="skyblue")
            labels = {
                (u, v): f"{d.get('L',0):.2f}/{d.get('R',0):.2f}/{d.get('C',0):.2f}"
                for u, v, d in self.graph.edges(data=True)
            }
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=labels, font_size=6)
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
        except Exception as e:  # pragma: no cover - 測試時不易觸發
            raise RuntimeError(f"無法產生 STM 圖像: {e}")
