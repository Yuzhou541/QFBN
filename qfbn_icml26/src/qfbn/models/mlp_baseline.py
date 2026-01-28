# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn as nn


def _act(name: str) -> nn.Module:
    name = (name or "relu").lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    return nn.ReLU()


class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], out_dim: int, act: str = "relu"):
        super().__init__()
        layers = []
        d = int(in_dim)
        for h in list(hidden_dims):
            layers.append(nn.Linear(d, int(h)))
            layers.append(_act(act))
            d = int(h)
        layers.append(nn.Linear(d, int(out_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], n_classes: int, act: str = "relu"):
        super().__init__()
        layers = []
        d = int(in_dim)
        for h in list(hidden_dims):
            layers.append(nn.Linear(d, int(h)))
            layers.append(_act(act))
            d = int(h)
        layers.append(nn.Linear(d, int(n_classes)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # logits
