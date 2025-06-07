import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int = 1000, embed_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.vocab = {"<pad>": 0}
        self.next_index = 1

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
        idx = self.encode_tokens(text)
        emb = self.embedding(idx)
        _, h = self.gru(emb)
        return h.squeeze(0)
