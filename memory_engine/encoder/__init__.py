import logging
import torch
import torch.nn as nn
from collections import OrderedDict

logger = logging.getLogger(__name__)

class TextEncoder(nn.Module):
    """將文字編碼成向量並快取結果以加速重複查詢。"""

    def __init__(self, vocab_size: int = 1000, embed_dim: int = 64, hidden_dim: int = 128, cache_size: int = 100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.vocab = {"<pad>": 0}
        self.next_index = 1
        # 使用有序字典實作簡易LRU快取
        self.cache_size = cache_size
        self.cache: OrderedDict[str, torch.Tensor] = OrderedDict()

    def encode_tokens(self, text: str) -> torch.Tensor:
        tokens = text.lower().split()
        indices = []
        for tok in tokens:
            if tok not in self.vocab and self.next_index < self.embedding.num_embeddings:
                self.vocab[tok] = self.next_index
                self.next_index += 1
            indices.append(self.vocab.get(tok, 0))
        return torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    def forward(self, text: str) -> torch.Tensor:
        try:
            # 若快取已存在，直接回傳快取向量
            if text in self.cache:
                emb = self.cache.pop(text)
                # 重新放入確保最近使用
                self.cache[text] = emb
                return emb.clone()

            idx = self.encode_tokens(text)
            emb = self.embedding(idx)
            _, h = self.gru(emb)
            result = h.squeeze(0).squeeze(0)
            # 存入快取並維持大小限制
            self.cache[text] = result.detach().clone()
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
            return result
        except Exception as e:
            logger.exception("文字編碼失敗: %s", e)
            raise
