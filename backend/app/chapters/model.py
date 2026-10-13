# Copied from experiments/m5-chapters/lib/model.py -- keep in sync (0009 §3).
"""Boundary tagger: frozen unit embeddings -> BiLSTM -> per-unit logit (0009 §3).

Besides the embedding, each unit gets two cheap side features:
- its relative position in the episode (t / duration), since intros
  and outros have typical chapter patterns;
- the cosine similarity to the previous unit, the raw signal the
  TextTiling baseline relies on, handed to the model directly.

Few hundred K parameters; trains on the 1060 in minutes per epoch.
This module is copied into the backend for inference.
"""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def side_features(emb: torch.Tensor, rel_pos: torch.Tensor) -> torch.Tensor:
    """[B, T, D] embeddings + [B, T] relative position -> [B, T, 2]."""
    prev = torch.roll(emb, shifts=1, dims=1)
    sim = nn.functional.cosine_similarity(emb, prev, dim=-1)
    sim[:, 0] = 1.0
    return torch.stack([rel_pos, sim], dim=-1)


class BoundaryTagger(nn.Module):
    def __init__(self, emb_dim: int = 384, proj: int = 256, hidden: int = 128, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.config = dict(emb_dim=emb_dim, proj=proj, hidden=hidden, layers=layers, dropout=dropout)
        self.inp = nn.Sequential(nn.Linear(emb_dim + 2, proj), nn.GELU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(proj, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, 1))

    def forward(self, emb: torch.Tensor, rel_pos: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """emb [B, T, D], rel_pos [B, T], lengths [B] -> logits [B, T] (padding positions undefined)."""
        x = self.inp(torch.cat([emb, side_features(emb, rel_pos)], dim=-1))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=emb.shape[1])
        return self.out(h).squeeze(-1)
