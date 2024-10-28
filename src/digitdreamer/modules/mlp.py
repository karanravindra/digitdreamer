import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    def __init__(self, emb_dim: int, mlp_mult: int, mlp_dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, emb_dim * mlp_mult)
        self.fc2 = nn.Linear(emb_dim * mlp_mult, emb_dim)
        self.drop = nn.Dropout(mlp_dropout)

    def forward(self, x):
        return self.fc2(F.silu(self.drop(self.fc1(x))))
