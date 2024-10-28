from __future__ import annotations

import torch.nn.functional as F
from torch import nn


class AttentionBlock(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        q_heads: int,
        kv_heads: int | None = None,
        attn_dropout: float = 0.1,
    ) -> None:
        assert emb_dim % q_heads == 0, "emb_dim must be divisible by q_heads"
        assert (
            kv_heads is None or emb_dim % kv_heads == 0
        ), "emb_dim must be divisible by kv_heads if defined"

        super().__init__()
        self.head_dim = emb_dim // q_heads
        self.q_heads = q_heads
        self.attn_dropout = attn_dropout
        self.kv_heads = kv_heads if kv_heads is not None else q_heads

        self.q = nn.Linear(emb_dim, self.head_dim * self.q_heads)
        self.kv = nn.Linear(emb_dim, self.head_dim * self.kv_heads * 2)
        self.o = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        b, n, _ = x.shape
        q = self.q(x)
        k, v = self.kv(x).chunk(2, dim=-1)

        q = q.view(b, n, self.q_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, n, self.kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.kv_heads, self.head_dim).transpose(1, 2)

        h_ = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=False,
            enable_gqa=True,
            dropout_p=self.attn_dropout if self.training else 0,
        )
        h_ = h_.transpose(1, 2).reshape(b, n, -1)
        return self.o(h_)
