from __future__ import annotations
from typing import List, Optional

import torch
import torch.nn as nn

from .fbn_linear import FBNLinear, FBNRegularizers


class FBNMLP(nn.Module):
    """
    Multi-layer FBN MLP.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        f_atoms: List[str],
        g_mode: str = "fixed",
        g_fixed: str = "tanh",
        g_atoms: Optional[List[str]] = None,
        g_per_edge: bool = False,
    ):
        super().__init__()
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(FBNLinear(
                in_dim=dims[i],
                out_dim=dims[i+1],
                f_atom_names=f_atoms,
                g_mode=g_mode if i < len(dims) - 2 else "fixed",  # last layer fixed g=id by default
                g_fixed=g_fixed if i < len(dims) - 2 else "id",
                g_atom_names=g_atoms,
                g_per_edge=g_per_edge,
            ))
        self.layers = nn.ModuleList(layers)

    def regularization_loss(self, tau: float, regs: FBNRegularizers):
        total = torch.tensor(0.0, device=self.layers[0].W.device)
        logs = {}
        for li, layer in enumerate(self.layers):
            r, lg = layer.regularization_loss(tau, regs)
            total = total + r
            for k, v in lg.items():
                logs[f"layer{li}_{k}"] = v
        return total, logs

    def hard_f_indices(self):
        return [layer.hard_f_indices() for layer in self.layers]

    def forward(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, tau=tau)
        return h


class TeacherMLP(nn.Module):
    """
    Discrete teacher network (fixed f indices), used for synthetic recovery.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        f_atoms: List[str],
        g_fixed: str = "tanh",
    ):
        super().__init__()
        self.f_atoms = f_atoms
        self.g_fixed = g_fixed

        dims = [input_dim] + list(hidden_dims) + [output_dim]
        self.W = nn.ParameterList()
        self.b = nn.ParameterList()
        for i in range(len(dims) - 1):
            self.W.append(nn.Parameter(torch.randn(dims[i+1], dims[i]) * 0.1))
            self.b.append(nn.Parameter(torch.zeros(dims[i+1])))

        # store chosen atom indices per layer edge
        self.f_idx = []

    def set_params(self, W_list, b_list, f_idx_list):
        # W_list: list of tensors, b_list: list of tensors, f_idx_list: list of int tensors
        for i in range(len(self.W)):
            self.W[i].data.copy_(W_list[i])
            self.b[i].data.copy_(b_list[i])
        self.f_idx = f_idx_list

    def forward(self, x: torch.Tensor, atom_fns, g_fn) -> torch.Tensor:
        h = x
        for li in range(len(self.W)):
            W = self.W[li]
            b = self.b[li]
            idx = self.f_idx[li]  # (out,in)
            # apply f atom per edge
            # build W_eff by indexing per edge (loop is OK for synthetic sizes)
            W_eff = torch.zeros_like(W)
            for ai, name in enumerate(self.f_atoms):
                mask = (idx == ai)
                if mask.any():
                    W_eff[mask] = atom_fns[name](W[mask])

            z = h.unsqueeze(1) * W_eff.unsqueeze(0)  # (B,out,in)
            # g applied before sum
            if li < len(self.W) - 1:
                gz = g_fn(z)
            else:
                gz = z  # last layer identity
            h = gz.sum(dim=-1) + b.unsqueeze(0)
        return h
