from __future__ import annotations
from typing import Callable, Dict, List
import torch
import torch.nn.functional as F


def _safe_exp(x: torch.Tensor) -> torch.Tensor:
    # prevent overflow
    return torch.exp(torch.clamp(x, -10.0, 10.0))


def _poly2(x: torch.Tensor) -> torch.Tensor:
    return x * x


def _poly3(x: torch.Tensor) -> torch.Tensor:
    return x * x * x


def build_f_atoms(names: List[str]) -> Dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    """
    Weight-space atoms f(w). Must include "zero" to represent deleted edges.
    """
    atoms: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {}
    for n in names:
        if n == "zero":
            atoms[n] = lambda w: torch.zeros_like(w)
        elif n == "id":
            atoms[n] = lambda w: w
        elif n == "sin":
            atoms[n] = torch.sin
        elif n == "tanh":
            atoms[n] = torch.tanh
        elif n == "exp":
            atoms[n] = _safe_exp
        elif n == "poly2":
            atoms[n] = _poly2
        elif n == "poly3":
            atoms[n] = _poly3
        else:
            raise ValueError(f"Unknown f atom: {n}")
    return atoms


def build_g_atoms(names: List[str]) -> Dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    """
    Data-space atoms g(z). "zero" corresponds to dropout-like deletion.
    """
    atoms: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {}
    for n in names:
        if n == "zero":
            atoms[n] = lambda z: torch.zeros_like(z)
        elif n == "id":
            atoms[n] = lambda z: z
        elif n == "relu":
            atoms[n] = F.relu
        elif n == "tanh":
            atoms[n] = torch.tanh
        elif n == "sigmoid":
            atoms[n] = torch.sigmoid
        else:
            raise ValueError(f"Unknown g atom: {n}")
    return atoms
