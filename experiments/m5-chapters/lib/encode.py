"""Frozen sentence encoders (docs/decisions/0009 §3).

Mean pooling over the last hidden state (masking padding), then L2
normalisation -- the documented pooling for both candidate models.
The e5 family expects a "passage: " prefix on every input.
"""

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ENCODERS = {
    "e5-small": ("intfloat/multilingual-e5-small", "passage: "),
    "minilm": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", ""),
    "e5-base": ("intfloat/multilingual-e5-base", "passage: "),
}


class Encoder:
    def __init__(self, key: str, device: str = "cpu", max_length: int = 128):
        name, self.prefix = ENCODERS[key]
        self.key, self.device, self.max_length = key, device, max_length
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).eval().to(device)
        self.dim = self.model.config.hidden_size

    @torch.inference_mode()
    def __call__(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        """Embeddings in input order. Texts are batched by length to cut padding."""
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        texts = [texts[i] for i in order]
        out = []
        for i in range(0, len(texts), batch_size):
            b = self.tok(
                [self.prefix + t for t in texts[i : i + batch_size]],
                padding=True, truncation=True, max_length=self.max_length, return_tensors="pt",
            ).to(self.device)
            h = self.model(**b).last_hidden_state
            mask = b["attention_mask"].unsqueeze(-1).to(h.dtype)
            v = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            out.append(torch.nn.functional.normalize(v, dim=-1).float().cpu().numpy())
        if not out:
            return np.zeros((0, self.dim), dtype=np.float32)
        sorted_emb = np.vstack(out)
        emb = np.empty_like(sorted_emb)
        emb[order] = sorted_emb
        return emb
