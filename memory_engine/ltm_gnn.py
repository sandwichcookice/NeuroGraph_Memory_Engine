import torch
import torch.nn as nn
import networkx as nx
from typing import Iterable, Tuple

class GNNLongTermMemory(nn.Module):
    """具備可微分結構的長期記憶模組。"""

    def __init__(self, embedding_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.graph = nx.DiGraph()
        # 動態節點與邊參數
        self.node_params = nn.ParameterDict()
        self.edge_params = nn.ParameterDict()
        self.self_proj = nn.Linear(embedding_dim, hidden_dim)
        self.neigh_proj = nn.Linear(embedding_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, 1)
        self.node_counter = 0

    # ------------------------------------------------------------------
    def add_state(
        self,
        emb: torch.Tensor | None = None,
        node_type: str = "state",
        node_id: int | None = None,
    ) -> int:
        """新增節點，若未提供嵌入則隨機初始化，並可指定節點編號。"""
        if emb is None:
            emb = torch.randn(self.embedding_dim)
        if node_id is None:
            idx = str(self.node_counter)
            self.node_counter += 1
        else:
            idx = str(node_id)
            self.node_counter = max(self.node_counter, node_id + 1)
        self.graph.add_node(idx, type=node_type)
        self.node_params[idx] = nn.Parameter(emb.detach().clone())
        return int(idx)

    def add_edge(self, src: int, dst: int, action: str = ""):
        """新增可微邊，並於圖中標記動作名稱。"""
        key = f"{src}->{dst}"
        if key not in self.edge_params:
            self.edge_params[key] = nn.Parameter(torch.randn(1))
            self.graph.add_edge(src, dst, action=action)
        self.graph[src][dst]["weight"] = float(self.edge_params[key].item())

    # ------------------------------------------------------------------
    def message_passing(self) -> dict[int, torch.Tensor]:
        """執行單層訊息傳遞並回傳新節點表徵。"""
        h = {int(k): p for k, p in self.node_params.items()}
        new_h = {}
        for n in self.graph.nodes:
            x_i = self.self_proj(h[int(n)])
            agg = torch.zeros_like(x_i)
            for j in self.graph.predecessors(n):
                w = self.edge_params.get(f"{j}->{n}")
                if w is None:
                    continue
                agg += w * self.neigh_proj(h[int(j)])
            new_h[int(n)] = torch.relu(x_i + agg)
        return new_h

    def forward(self, nodes: Iterable[int]) -> torch.Tensor:
        """計算指定節點的 readout 值。"""
        h = self.message_passing()
        outs = []
        for n in nodes:
            vec = h.get(n)
            if vec is None:
                raise ValueError(f"節點 {n} 不存在")
            outs.append(self.readout(vec))
        return torch.stack(outs).squeeze(-1)

    # ------------------------------------------------------------------
    def train_offline(
        self,
        samples: Iterable[Tuple[int, int, int, float]],
        epochs: int = 5,
        lr: float = 1e-3,
        lambda_sim: float = 0.05,
        lambda_reg: float = 1e-4,
    ) -> float:
        """完整的離線訓練流程。

        參數 ``samples`` 為 ``(state, action, next_state, reward)`` 四元組
        的可疊代序列。此函式會依序計算任務損失、語意相似度損失與正則化，
        並使用 AdamW 更新 ``Kᵢ`` 與 ``Wᵢⱼ``。
        """

        opt = torch.optim.AdamW(self.parameters(), lr=lr)
        last_loss = 0.0

        for _ in range(epochs):
            total = 0.0
            count = 0
            for s, _a, sp, r in samples:
                try:
                    q_vals = self.forward([s, sp])
                except Exception as exc:  # pragma: no cover - 無效樣本
                    print(f"train skip: {exc}", file=sys.stderr)
                    continue

                # 任務損失：回歸目標 Q 值
                target = torch.tensor([0.0, r], dtype=q_vals.dtype)
                loss_task = nn.functional.mse_loss(q_vals, target)

                # 語意相近損失：正獎勵時強化相似度
                loss_sim = torch.tensor(0.0)
                if r > 0:
                    emb_s = self.node_params.get(str(s))
                    emb_sp = self.node_params.get(str(sp))
                    if emb_s is not None and emb_sp is not None:
                        cos = nn.functional.cosine_similarity(emb_s, emb_sp, dim=0)
                        loss_sim = 1 - cos

                # 正則化項：參數 L2
                loss_reg = sum(p.pow(2).mean() for p in self.parameters())

                loss = loss_task + lambda_sim * loss_sim + lambda_reg * loss_reg

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                opt.step()

                total += float(loss.item())
                count += 1

            if count:
                last_loss = total / count

        return last_loss

    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        """回傳目前節點與邊參數，方便序列化。"""
        return {
            "nodes": {k: p.detach().cpu().tolist() for k, p in self.node_params.items()},
            "edges": {k: p.detach().cpu().item() for k, p in self.edge_params.items()},
        }

    # ------------------------------------------------------------------
    def decay_and_prune(self, gamma: float = 0.99, threshold: float = 0.01):
        """衰減邊權重並移除過低的連結。"""
        for key in list(self.edge_params.keys()):
            w = self.edge_params[key]
            w.data.mul_(gamma)
            if w.abs().item() < threshold:
                src, dst = map(int, key.split("->"))
                if self.graph.has_edge(src, dst):
                    self.graph.remove_edge(src, dst)
                del self.edge_params[key]
            else:
                src, dst = map(int, key.split("->"))
                if self.graph.has_edge(src, dst):
                    self.graph[src][dst]["weight"] = float(w.item())

    def find_goal(self, item: str) -> int | None:
        """尋找標記為特定目標的節點編號。"""
        for n, d in self.graph.nodes(data=True):
            if d.get("goal") == item:
                return int(n)
        return None
