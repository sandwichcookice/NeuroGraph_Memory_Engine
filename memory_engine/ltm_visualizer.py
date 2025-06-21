"""LTM 視覺化工具

提供 t-SNE 降維、邊權重熱圖與路徑梯度分析等功能，
協助檢查 GNNLongTermMemory 的訓練狀態。"""

import argparse
import logging
import os
import pickle
from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx
import torch
from sklearn.manifold import TSNE

from .ltm_gnn import GNNLongTermMemory

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------

def _load_snapshot(path: str) -> GNNLongTermMemory:
    """載入 GNNLongTermMemory 快照。"""
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, GNNLongTermMemory):
            raise TypeError("快照格式錯誤：非 GNNLongTermMemory")
        return obj
    except Exception as exc:  # pragma: no cover - 載入失敗處理
        logging.error("無法載入 LTM 快照: %s", exc)
        raise

# ---------------------------------------------------------------------------

def _visualize_nodes(ltm: GNNLongTermMemory, out_path: str) -> None:
    """以 t-SNE 繪製節點分佈圖。"""
    embeddings: list[list[float]] = []
    labels: list[str] = []
    for n, data in ltm.graph.nodes(data=True):
        param = ltm.node_params.get(str(n))
        if param is None:
            continue
        embeddings.append(param.detach().cpu().tolist())
        labels.append(data.get("type", "state"))
    if not embeddings:
        logging.warning("無節點可視化")
        return
    tsne = TSNE(n_components=2, init="pca", random_state=42)
    coords = tsne.fit_transform(embeddings)
    plt.figure(figsize=(6, 6))
    unique = sorted(set(labels))
    for t in unique:
        idx = [i for i, l in enumerate(labels) if l == t]
        plt.scatter(coords[idx, 0], coords[idx, 1], label=t)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logging.info("節點分佈圖已輸出至 %s", out_path)

# ---------------------------------------------------------------------------

def _visualize_edges(ltm: GNNLongTermMemory, out_path: str) -> None:
    """繪製邊權重熱圖。"""
    if ltm.graph.number_of_edges() == 0:
        logging.warning("無邊可視化")
        return
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(ltm.graph)
    weights = [abs(ltm.edge_params[f"{u}->{v}"].item()) for u, v in ltm.graph.edges]
    nx.draw_networkx(ltm.graph, pos, width=weights, node_size=300, with_labels=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logging.info("邊權重圖已輸出至 %s", out_path)

# ---------------------------------------------------------------------------

def _path_gradient(
    ltm: GNNLongTermMemory, nodes: Iterable[int], out_path: str
) -> None:
    """計算指定路徑對預測值的梯度並可視化。"""
    try:
        loss = ltm(torch.tensor(list(nodes)))
        grads = torch.autograd.grad(loss.sum(), list(ltm.edge_params.values()), retain_graph=True)
    except Exception as exc:  # pragma: no cover - 解析失敗
        logging.error("梯度計算失敗: %s", exc)
        return
    grad_map = {k: abs(g.item()) for k, g in zip(ltm.edge_params.keys(), grads)}
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(ltm.graph)
    edges, colors = [], []
    for u, v in ltm.graph.edges:
        key = f"{u}->{v}"
        edges.append((u, v))
        colors.append(grad_map.get(key, 0.0))
    nx.draw_networkx(
        ltm.graph, pos, edgelist=edges, edge_color=colors, edge_cmap=plt.cm.RdBu, node_size=300
    )
    plt.colorbar(label="|∂Q/∂W|")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logging.info("路徑梯度圖已輸出至 %s", out_path)

# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GNN LTM 視覺化工具")
    parser.add_argument("snapshot", help="快照檔案路徑")
    parser.add_argument("--outdir", default="viz_out", help="輸出資料夾")
    parser.add_argument("--path", nargs="*", type=int, help="指定路徑節點序列做梯度分析")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    ltm = _load_snapshot(args.snapshot)

    _visualize_nodes(ltm, os.path.join(args.outdir, "nodes.png"))
    _visualize_edges(ltm, os.path.join(args.outdir, "edges.png"))

    if args.path:
        _path_gradient(ltm, args.path, os.path.join(args.outdir, "grad.png"))

if __name__ == "__main__":  # pragma: no cover - 命令列入口
    main()

