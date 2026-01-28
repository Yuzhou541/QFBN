from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from qfbn.basis import build_f_atoms, build_g_atoms, ATOM_DEGREE
from qfbn.utils.metrics import entropy_from_probs


@dataclass
class FBNRegularizers:
    lambda_entropy_f: float = 0.0
    lambda_degree_f: float = 0.0
    lambda_nonzero_f: float = 0.0

    lambda_entropy_g: float = 0.0
    lambda_degree_g: float = 0.0
    lambda_nonzero_g: float = 0.0


class FBNLinear(nn.Module):
    """
    QFBN Linear layer with edge-wise f_ij on weights and (optional) g selection on data.

    Forward (strict mechanism):
      y_i = sum_j g_ij( f_ij(w_ij) * x_j ) + b_i

    Implementation choices:
    - f selection is ALWAYS edge-wise via softmax gates.
    - g can be:
        - fixed (fast)
        - selected per-node (g_i shared across j), which is consistent with your allowed simplification g_ij = g_i.
        - selected per-edge (heavier; optional)
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        f_atom_names: List[str],
        g_mode: str = "fixed",              # "fixed" or "select"
        g_fixed: str = "tanh",
        g_atom_names: Optional[List[str]] = None,
        g_per_edge: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # weight and bias
        self.W = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1)
        self.b = nn.Parameter(torch.zeros(out_dim))

        # f dictionary + gates (edge-wise)
        self.f_atom_names = list(f_atom_names)
        self.f_atoms = build_f_atoms(self.f_atom_names)
        self.Kf = len(self.f_atom_names)
        self.theta_f = nn.Parameter(torch.zeros(out_dim, in_dim, self.Kf))  # logits

        # g setup
        self.g_mode = g_mode
        self.g_per_edge = g_per_edge
        if self.g_mode not in ["fixed", "select"]:
            raise ValueError("g_mode must be 'fixed' or 'select'")

        if g_atom_names is None:
            g_atom_names = ["zero", "id", "relu", "tanh", "sigmoid"]
        self.g_atom_names = list(g_atom_names)
        self.g_atoms = build_g_atoms(self.g_atom_names)
        self.Kg = len(self.g_atom_names)

        if self.g_mode == "fixed":
            if g_fixed not in self.g_atoms:
                raise ValueError(f"Unknown g_fixed={g_fixed}")
            self.g_fixed = g_fixed
            self.theta_g = None
        else:
            self.g_fixed = None
            if self.g_per_edge:
                self.theta_g = nn.Parameter(torch.zeros(out_dim, in_dim, self.Kg))
            else:
                self.theta_g = nn.Parameter(torch.zeros(out_dim, self.Kg))

        # Precompute degrees
        self.f_degrees = torch.tensor([ATOM_DEGREE[n] for n in self.f_atom_names], dtype=torch.float32)
        self.g_degrees = torch.tensor([ATOM_DEGREE[n] for n in self.g_atom_names], dtype=torch.float32)

        # Identify "zero" index (must exist for your story)
        if "zero" not in self.f_atom_names:
            raise ValueError("f_atoms must include 'zero' to model deleted edges.")
        if "zero" not in self.g_atom_names:
            # g zero is optional conceptually, but keep it available
            pass
        self.f_zero_idx = self.f_atom_names.index("zero")
        self.g_zero_idx = self.g_atom_names.index("zero") if "zero" in self.g_atom_names else None
        # [ADD] optional oracle constraints (non-persistent buffers)
        self.register_buffer("_allowed_atoms_mask", None, persistent=False)  # (out,in,K) bool or None
        self.register_buffer("_fixed_f_idx", None, persistent=False)        # (out,in) long with -1 meaning free


    def f_probs(self, tau: float) -> torch.Tensor:
        return F.softmax(self.theta_f / max(tau, 1e-6), dim=-1)

    def g_probs(self, tau: float) -> Optional[torch.Tensor]:
        if self.g_mode == "fixed":
            return None
        return F.softmax(self.theta_g / max(tau, 1e-6), dim=-1)

    def expected_degree_f(self, pf: torch.Tensor) -> torch.Tensor:
        deg = self.f_degrees.to(pf.device)
        return (pf * deg).sum(dim=-1)  # (out,in)

    def expected_nonzero_f(self, pf: torch.Tensor) -> torch.Tensor:
        return 1.0 - pf[..., self.f_zero_idx]  # (out,in)

    def regularization_loss(self, tau: float, regs: FBNRegularizers) -> Tuple[torch.Tensor, Dict[str, float]]:
        pf = self.f_probs(tau)
        ent_f = entropy_from_probs(pf).mean()
        deg_f = self.expected_degree_f(pf).mean()
        nonzero_f = self.expected_nonzero_f(pf).mean()

        reg = regs.lambda_entropy_f * ent_f + regs.lambda_degree_f * deg_f + regs.lambda_nonzero_f * nonzero_f

        log = {
            "ent_f": float(ent_f.detach().cpu()),
            "deg_f": float(deg_f.detach().cpu()),
            "nonzero_f": float(nonzero_f.detach().cpu()),
        }

        if self.g_mode == "select":
            pg = self.g_probs(tau)
            ent_g = entropy_from_probs(pg).mean()
            deg = self.g_degrees.to(pg.device)
            deg_g = (pg * deg).sum(dim=-1).mean()
            if self.g_zero_idx is not None:
                nonzero_g = (1.0 - pg[..., self.g_zero_idx]).mean()
            else:
                nonzero_g = torch.tensor(0.0, device=pg.device)

            reg = reg + regs.lambda_entropy_g * ent_g + regs.lambda_degree_g * deg_g + regs.lambda_nonzero_g * nonzero_g
            log.update({
                "ent_g": float(ent_g.detach().cpu()),
                "deg_g": float(deg_g.detach().cpu()),
                "nonzero_g": float(nonzero_g.detach().cpu()),
            })

        return reg, log

    def hard_f_indices(self) -> torch.Tensor:
        # argmax over atoms
        return torch.argmax(self.theta_f, dim=-1)  # (out,in)

    def forward(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """
        x: (B, in)
        return: (B, out)
        """
        B = x.shape[0]

        # compute effective weights: w_eff = sum_r pi_r * f_r(W)
        pf = self.f_probs(tau)  # (out,in,Kf)
        # stack f_r(W)
        W_stack = []
        for name in self.f_atom_names:
            W_stack.append(self.f_atoms[name](self.W))
        Wf = torch.stack(W_stack, dim=-1)  # (out,in,Kf)
        W_eff = (pf * Wf).sum(dim=-1)      # (out,in)

        # contributions z_{b,i,j} = W_eff[i,j] * x[b,j]
        z = x.unsqueeze(1) * W_eff.unsqueeze(0)  # (B, out, in)

        # apply g before summation
        if self.g_mode == "fixed":
            g_fn = self.g_atoms[self.g_fixed]
            gz = g_fn(z)
        else:
            pg = self.g_probs(tau)
            if self.g_per_edge:
                # pg: (out,in,Kg)
                gz_stack = []
                for name in self.g_atom_names:
                    gz_stack.append(self.g_atoms[name](z))
                gz_all = torch.stack(gz_stack, dim=-1)  # (B,out,in,Kg)
                gz = (gz_all * pg.unsqueeze(0)).sum(dim=-1)  # (B,out,in)
            else:
                # pg: (out,Kg) shared across j
                gz_stack = []
                for name in self.g_atom_names:
                    gz_stack.append(self.g_atoms[name](z))
                gz_all = torch.stack(gz_stack, dim=-1)  # (B,out,in,Kg)
                gz = (gz_all * pg.unsqueeze(0).unsqueeze(2)).sum(dim=-1)

        y = gz.sum(dim=-1) + self.b.unsqueeze(0)  # (B,out)
        return y
    def set_f_constraints(self, allowed_atoms_mask=None, fixed_f_idx=None):
        """
        allowed_atoms_mask: bool tensor (out,in,K). True=allowed.
        fixed_f_idx: long tensor (out,in). -1 means free edge; >=0 means fixed atom index.
        """
        if allowed_atoms_mask is not None:
            self._allowed_atoms_mask = allowed_atoms_mask.bool()
        else:
            self._allowed_atoms_mask = None

        if fixed_f_idx is not None:
            self._fixed_f_idx = fixed_f_idx.long()
        else:
            self._fixed_f_idx = None

    # [ADD]
    def _apply_oracle_to_probs(self, probs: torch.Tensor) -> torch.Tensor:
        # probs: (out,in,K)
        if self._allowed_atoms_mask is not None:
            m = self._allowed_atoms_mask.to(dtype=probs.dtype)
            probs = probs * m
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        if self._fixed_f_idx is not None:
            fixed = self._fixed_f_idx  # (out,in)
            one_hot = F.one_hot(fixed.clamp_min(0), num_classes=probs.size(-1)).to(dtype=probs.dtype)
            if (fixed < 0).any():
                free = (fixed < 0).unsqueeze(-1).to(dtype=probs.dtype)
                fixm = (fixed >= 0).unsqueeze(-1).to(dtype=probs.dtype)
                probs = probs * free + one_hot * fixm
            else:
                probs = one_hot
        return probs


    def forward(self, *args, **kwargs):
        # ... compute logits ...
        # probs = F.softmax(logits / tau, dim=-1)
        # [ADD]
        probs = self._apply_oracle_to_probs(probs)
        # ... rest ...
        return out


    def hard_f_indices(self):
        # logits: (out,in,K)
        logits = self.f_logits  
        if self._allowed_atoms_mask is not None:
            logits = logits.masked_fill(~self._allowed_atoms_mask, -1e9)
        idx = torch.argmax(logits, dim=-1)
        if self._fixed_f_idx is not None:
            idx = torch.where(self._fixed_f_idx >= 0, self._fixed_f_idx, idx)
        return idx
