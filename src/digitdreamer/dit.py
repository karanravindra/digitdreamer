from __future__ import annotations

import torch
from torch import nn

from digitdreamer.modules import MLP, AttentionBlock


class TransfomerBlock(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        q_heads: int,
        kv_heads: int | None = None,
        mlp_mult: int = 4,
        residual_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.attn = AttentionBlock(
            emb_dim,
            q_heads,
            kv_heads,
            attn_dropout=attn_dropout,
        )
        self.mlp = MLP(emb_dim, mlp_mult, mlp_dropout=mlp_dropout)
        self.drop = nn.Dropout(residual_dropout)

        self.c_proj = nn.Linear(emb_dim, emb_dim * 6)

    def forward(self, x, c):
        c = self.c_proj(c).chunk(6, dim=-1)

        x = self.drop(x) + c[0] * self.attn(c[1] * self.norm1(x) + c[2])
        return self.drop(x) + c[3] * self.mlp(c[4] * self.norm2(x) + c[5])


class DiT(nn.Module):
    def __init__(
        self,
        in_channels=8,
        patch_size=2,
        emb_dim=48,
        num_layers=8,
        q_heads=24,
        kv_heads=8,
        mlp_mult=2,
    ) -> None:
        super().__init__()
        self.num_patches = 2 // patch_size

        self.pos_emb = nn.Parameter(torch.randn(1, self.num_patches**2, emb_dim))
        self.in_proj = nn.Conv2d(
            in_channels,
            emb_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.blocks = nn.ModuleList(
            [
                TransfomerBlock(emb_dim, q_heads, kv_heads, mlp_mult)
                for _ in range(num_layers)
            ],
        )
        self.ln = nn.LayerNorm(emb_dim)
        self.out_proj = nn.ConvTranspose2d(
            emb_dim,
            in_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.t_emb = nn.Linear(1, emb_dim)
        self.c_emb = nn.Embedding(11, emb_dim)

    def forward(self, x, t=None, y=None):
        if t is None:
            t = torch.zeros((x.size(0), 1), device=x.device, dtype=torch.float32)
        if y is None:
            y = torch.zeros((x.size(0),), device=x.device, dtype=torch.long)

        c = self.t_emb(t.view(-1, 1)) + self.c_emb(y.view(-1))
        c = c.unsqueeze(1).expand(-1, self.num_patches**2, -1)

        x = self.in_proj(x).flatten(2).transpose(1, 2)
        x = x + self.pos_emb

        for block in self.blocks:
            x = block(x, c)

        x = self.ln(x)
        x = x.transpose(1, 2).reshape(x.size(0), -1, self.num_patches, self.num_patches)

        return self.out_proj(x)
