# -*- coding: utf-8 -*-
"""
qfbn.experiments.synth_recovery

synthetic recovery experiments with:
- mode=qfbn (default): learn atom types + recover mask
- mode=mlp: MLP baseline (no recovery metrics)
- mode=oracle_mask: oracle sparsity mask (mask F1=1 upper bound), forbid zero on non-zero edges
- mode=oracle_atom: oracle atom index (atom acc=1 upper bound), only learn continuous params
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# teacher/data generator (already in your repo)
from qfbn.data.synthetic import (
    make_teacher_student_regression,
    make_teacher_student_classification,
    build_f_atoms,
    build_g_atoms,
)

# baseline
from qfbn.models.mlp_baseline import MLPRegressor, MLPClassifier


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def short_uid(n: int = 8) -> str:
    alphabet = "0123456789abcdef"
    return "".join(random.choice(alphabet) for _ in range(n))


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_set(d: Dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def apply_cli_overrides(cfg: Dict[str, Any], sets: List[str]) -> Dict[str, Any]:
    for s in sets or []:
        if "=" not in s:
            raise ValueError(f"--set expects k=v, got: {s}")
        k, v = s.split("=", 1)
        deep_set(cfg, k.strip(), yaml.safe_load(v))
    return cfg


def get_device(cfg: Dict[str, Any]) -> torch.device:
    run = cfg.get("run", {})
    dev = str(run.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if dev.startswith("cuda") and not torch.cuda.is_available():
        dev = "cpu"
    return torch.device(dev)


def is_classification(cfg: Dict[str, Any]) -> bool:
    data = cfg.get("data", {})
    return "n_classes" in data


def tau_at(step: int, total_steps: int, sched: Dict[str, Any]) -> float:
    start = float(sched.get("start", 2.0))
    end = float(sched.get("end", 0.2))
    kind = str(sched.get("schedule", "exp")).lower()
    if total_steps <= 1:
        return end
    t = step / (total_steps - 1)
    if kind == "linear":
        return start + (end - start) * t
    if start <= 0 or end <= 0:
        return end
    return start * ((end / start) ** t)


@dataclass
class OracleSpec:
    mode: str  # "none" | "oracle_mask" | "oracle_atom"
    f_atoms: List[str]
    teacher_f_idx_list: List[torch.Tensor]  # list[(out,in)]
    zero_id: int


class QFBNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, f_atoms: List[str], w_scale: float, b_scale: float, device: torch.device):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.f_atoms = list(f_atoms)
        self.K = len(self.f_atoms)

        self.W = nn.Parameter(torch.randn(self.out_dim, self.in_dim, device=device) * float(w_scale))
        self.b = nn.Parameter(torch.randn(self.out_dim, device=device) * float(b_scale))

        self.f_logits = nn.Parameter(torch.zeros(self.out_dim, self.in_dim, self.K, device=device))

        self.register_buffer("_allowed_mask", None, persistent=False)  # (out,in,K) bool
        self.register_buffer("_fixed_idx", None, persistent=False)    # (out,in) long, -1 free

    def set_oracle(self, allowed_mask: Optional[torch.Tensor], fixed_idx: Optional[torch.Tensor]) -> None:
        self._allowed_mask = allowed_mask.bool() if allowed_mask is not None else None
        self._fixed_idx = fixed_idx.long() if fixed_idx is not None else None

    def probs(self, tau: float) -> torch.Tensor:
        logits = self.f_logits / float(tau)
        if self._allowed_mask is not None:
            logits = logits.masked_fill(~self._allowed_mask, -1e9)
        p = F.softmax(logits, dim=-1)

        if self._fixed_idx is not None:
            fixed = self._fixed_idx
            one_hot = F.one_hot(fixed.clamp_min(0), num_classes=self.K).to(dtype=p.dtype)
            if (fixed < 0).any():
                free = (fixed < 0).unsqueeze(-1).to(dtype=p.dtype)
                fixm = (fixed >= 0).unsqueeze(-1).to(dtype=p.dtype)
                p = p * free + one_hot * fixm
            else:
                p = one_hot
        return p

    def effective_W(self, atom_fns: Dict[str, Any], tau: float) -> torch.Tensor:
        p = self.probs(tau)  # (out,in,K)
        Wk = []
        for name in self.f_atoms:
            Wk.append(atom_fns[name](self.W))  # (out,in)
        Wk = torch.stack(Wk, dim=-1)  # (out,in,K)
        return (p * Wk).sum(dim=-1)  # (out,in)

    def forward(self, h: torch.Tensor, atom_fns: Dict[str, Any], g_fn: Any, tau: float, is_last: bool) -> torch.Tensor:
        W_eff = self.effective_W(atom_fns, tau)  # (out,in)
        z = h.unsqueeze(1) * W_eff.unsqueeze(0)  # (B,out,in)
        gz = z if is_last else g_fn(z)
        out = gz.sum(dim=-1) + self.b.unsqueeze(0)  # (B,out)
        return out

    def hard_f_idx(self) -> torch.Tensor:
        logits = self.f_logits
        if self._allowed_mask is not None:
            logits = logits.masked_fill(~self._allowed_mask, -1e9)
        idx = logits.argmax(dim=-1)  # (out,in)
        if self._fixed_idx is not None:
            idx = torch.where(self._fixed_idx >= 0, self._fixed_idx, idx)
        return idx


class QFBNNet(nn.Module):
    def __init__(
        self,
        dims: List[int],
        f_atoms: List[str],
        g_name: str,
        w_scale: float,
        b_scale: float,
        device: torch.device,
    ):
        super().__init__()
        self.dims = list(map(int, dims))
        self.f_atoms = list(f_atoms)
        self.g_name = str(g_name)

        self.atom_fns = build_f_atoms(self.f_atoms)
        g_fns = build_g_atoms(["zero", "id", "relu", "tanh", "sigmoid"])
        if self.g_name not in g_fns:
            raise ValueError(f"Student g={self.g_name} not in supported g atoms.")
        self.g_fn = g_fns[self.g_name]

        layers = []
        for li in range(len(self.dims) - 1):
            layers.append(QFBNLayer(self.dims[li], self.dims[li + 1], self.f_atoms, w_scale, b_scale, device))
        self.layers = nn.ModuleList(layers)

    def set_oracle(self, spec: Optional[OracleSpec], device: torch.device) -> None:
        for layer in self.layers:
            layer.set_oracle(None, None)
        if spec is None or spec.mode == "none":
            return

        teacher_list = spec.teacher_f_idx_list
        if len(teacher_list) != len(self.layers):
            raise RuntimeError(f"Oracle alignment failed: teacher has {len(teacher_list)} layers, student has {len(self.layers)}")

        # IMPORTANT: require identical atom ordering
        if list(spec.f_atoms) != list(self.f_atoms):
            raise RuntimeError(
                "Oracle requires identical f_atoms ordering between teacher(meta) and student. "
                f"teacher(meta)={spec.f_atoms}, student={self.f_atoms}"
            )

        K = len(self.f_atoms)
        zero_id = int(spec.zero_id)

        for li, (layer, tidx_cpu) in enumerate(zip(self.layers, teacher_list)):
            tidx = tidx_cpu.to(device)  # (out,in)
            outd, ind = tidx.shape

            allowed = torch.ones((outd, ind, K), dtype=torch.bool, device=device)
            fixed = torch.full((outd, ind), -1, dtype=torch.long, device=device)

            if spec.mode == "oracle_mask":
                is_zero = (tidx == zero_id)
                fixed[is_zero] = zero_id
                # forbid zero on non-zero edges (clear zero channel at those positions)
                allowed[..., zero_id][~is_zero] = False
            elif spec.mode == "oracle_atom":
                fixed = tidx
                allowed = F.one_hot(tidx.clamp_min(0), num_classes=K).bool()
            else:
                raise ValueError(f"Unknown oracle mode: {spec.mode}")

            layer.set_oracle(allowed, fixed)

    def forward(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        h = x
        for li, layer in enumerate(self.layers):
            h = layer(h, self.atom_fns, self.g_fn, tau, is_last=(li == len(self.layers) - 1))
        return h

    def hard_f_idx_list(self) -> List[torch.Tensor]:
        return [layer.hard_f_idx().detach().clone() for layer in self.layers]


def compute_f1(pred_pos: torch.Tensor, true_pos: torch.Tensor) -> float:
    # pred_pos/true_pos: bool
    tp = (pred_pos & true_pos).sum().item()
    fp = (pred_pos & ~true_pos).sum().item()
    fn = (~pred_pos & true_pos).sum().item()
    denom = (2 * tp + fp + fn)
    if denom <= 0:
        return 0.0
    return float(2 * tp / denom)


def recovery_metrics(pred_f_idx_list: List[torch.Tensor], teacher_meta: Dict[str, Any], device: torch.device) -> Tuple[float, float]:
    """
    atom_acc: exact atom-index match over all edges
    mask_f1: F1 on NON-ZERO support (standard sparsity mask recovery)
    """
    f_atoms = list(teacher_meta["f_atoms"])
    if "zero" not in f_atoms:
        raise ValueError("Recovery metrics require atom list to include 'zero'.")
    zero_id = f_atoms.index("zero")

    teacher_list = [t.to(device) for t in teacher_meta["f_idx_list"]]
    assert len(pred_f_idx_list) == len(teacher_list)

    correct = 0
    total = 0
    f1s = []

    for pidx, tidx in zip(pred_f_idx_list, teacher_list):
        pidx = pidx.to(device)
        tidx = tidx.to(device)

        total += tidx.numel()
        correct += (pidx == tidx).sum().item()

        pred_nonzero = (pidx != zero_id)
        true_nonzero = (tidx != zero_id)
        f1s.append(compute_f1(pred_nonzero, true_nonzero))

    atom_acc = float(correct / max(1, total))
    mask_f1 = float(sum(f1s) / max(1, len(f1s)))
    return atom_acc, mask_f1


def train_qfbn_regression(model: QFBNNet, x_tr: torch.Tensor, y_tr: torch.Tensor, train_cfg: Dict[str, Any], device: torch.device) -> None:
    epochs = int(train_cfg.get("epochs", 400))
    bs = int(train_cfg.get("batch_size", 256))
    lr_w = float(train_cfg.get("lr_w", train_cfg.get("lr", 1e-3)))
    lr_logits = float(train_cfg.get("lr_logits", train_cfg.get("lr", 1e-3)))
    wd = float(train_cfg.get("weight_decay", 0.0))
    tau_sched = dict(train_cfg.get("tau_schedule", {"start": 2.0, "end": 0.2, "schedule": "exp"}))

    lambda_sparsity = float(train_cfg.get("lambda_sparsity", 0.0))
    zero_id = model.f_atoms.index("zero") if (lambda_sparsity > 0 and "zero" in model.f_atoms) else None

    logits_params, weight_params = [], []
    for m in model.modules():
        if isinstance(m, QFBNLayer):
            logits_params.append(m.f_logits)
            weight_params.append(m.W)
            weight_params.append(m.b)

    opt = torch.optim.Adam(
        [
            {"params": weight_params, "lr": lr_w, "weight_decay": wd},
            {"params": logits_params, "lr": lr_logits, "weight_decay": 0.0},
        ]
    )

    n = x_tr.size(0)
    steps_per_epoch = max(1, math.ceil(n / bs))
    total_steps = epochs * steps_per_epoch
    step = 0

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for si in range(0, n, bs):
            idx = perm[si: si + bs]
            xb = x_tr[idx]
            yb = y_tr[idx]

            tau = tau_at(step, total_steps, tau_sched)
            pred = model(xb, tau=tau)
            loss = F.mse_loss(pred, yb)

            if lambda_sparsity > 0 and zero_id is not None:
                p_nonzero_all = []
                for layer in model.layers:
                    p = layer.probs(tau)
                    p_nonzero_all.append(1.0 - p[..., zero_id])
                p_nonzero = torch.cat([t.reshape(-1) for t in p_nonzero_all], dim=0).mean()
                loss = loss + lambda_sparsity * p_nonzero

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1


def train_qfbn_classification(model: QFBNNet, x_tr: torch.Tensor, y_tr: torch.Tensor, train_cfg: Dict[str, Any], device: torch.device) -> None:
    epochs = int(train_cfg.get("epochs", 400))
    bs = int(train_cfg.get("batch_size", 256))
    lr_w = float(train_cfg.get("lr_w", train_cfg.get("lr", 1e-3)))
    lr_logits = float(train_cfg.get("lr_logits", train_cfg.get("lr", 1e-3)))
    wd = float(train_cfg.get("weight_decay", 0.0))
    tau_sched = dict(train_cfg.get("tau_schedule", {"start": 2.0, "end": 0.2, "schedule": "exp"}))

    lambda_sparsity = float(train_cfg.get("lambda_sparsity", 0.0))
    zero_id = model.f_atoms.index("zero") if (lambda_sparsity > 0 and "zero" in model.f_atoms) else None

    logits_params, weight_params = [], []
    for m in model.modules():
        if isinstance(m, QFBNLayer):
            logits_params.append(m.f_logits)
            weight_params.append(m.W)
            weight_params.append(m.b)

    opt = torch.optim.Adam(
        [
            {"params": weight_params, "lr": lr_w, "weight_decay": wd},
            {"params": logits_params, "lr": lr_logits, "weight_decay": 0.0},
        ]
    )

    n = x_tr.size(0)
    steps_per_epoch = max(1, math.ceil(n / bs))
    total_steps = epochs * steps_per_epoch
    step = 0

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for si in range(0, n, bs):
            idx = perm[si: si + bs]
            xb = x_tr[idx]
            yb = y_tr[idx]

            tau = tau_at(step, total_steps, tau_sched)
            logits = model(xb, tau=tau)
            loss = F.cross_entropy(logits, yb)

            if lambda_sparsity > 0 and zero_id is not None:
                p_nonzero_all = []
                for layer in model.layers:
                    p = layer.probs(tau)
                    p_nonzero_all.append(1.0 - p[..., zero_id])
                p_nonzero = torch.cat([t.reshape(-1) for t in p_nonzero_all], dim=0).mean()
                loss = loss + lambda_sparsity * p_nonzero

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1


@torch.no_grad()
def eval_regression(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    pred = model(x, tau=0.2) if isinstance(model, QFBNNet) else model(x)
    return float(F.mse_loss(pred, y).item())


@torch.no_grad()
def eval_classification(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    model.eval()
    logits = model(x, tau=0.2) if isinstance(model, QFBNNet) else model(x)
    ce = float(F.cross_entropy(logits, y).item())
    acc = float((logits.argmax(dim=-1) == y).float().mean().item())
    return ce, acc


def train_mlp_regression(model: MLPRegressor, x_tr: torch.Tensor, y_tr: torch.Tensor, train_cfg: Dict[str, Any], device: torch.device) -> None:
    epochs = int(train_cfg.get("epochs", 400))
    bs = int(train_cfg.get("batch_size", 256))
    lr = float(train_cfg.get("lr", 1e-3))
    wd = float(train_cfg.get("weight_decay", 0.0))

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    n = x_tr.size(0)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for si in range(0, n, bs):
            idx = perm[si: si + bs]
            xb = x_tr[idx]
            yb = y_tr[idx]
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()


def train_mlp_classification(model: MLPClassifier, x_tr: torch.Tensor, y_tr: torch.Tensor, train_cfg: Dict[str, Any], device: torch.device) -> None:
    epochs = int(train_cfg.get("epochs", 400))
    bs = int(train_cfg.get("batch_size", 256))
    lr = float(train_cfg.get("lr", 1e-3))
    wd = float(train_cfg.get("weight_decay", 0.0))

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    n = x_tr.size(0)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for si in range(0, n, bs):
            idx = perm[si: si + bs]
            xb = x_tr[idx]
            yb = y_tr[idx]
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()


def aggregate_mean_std(vals: List[float]) -> Tuple[float, float]:
    if len(vals) == 0:
        return float("nan"), float("nan")
    m = sum(vals) / len(vals)
    v = sum((x - m) ** 2 for x in vals) / max(1, (len(vals) - 1))
    return float(m), float(math.sqrt(v))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--set", action="append", default=[], help="Override config keys, e.g. --set teacher.p_zero=0.4")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg = apply_cli_overrides(cfg, args.set)

    run_cfg = cfg.get("run", {})
    out_dir = str(run_cfg.get("out_dir", "runs"))
    exp_name = str(run_cfg.get("exp_name", Path(args.config).stem))
    n_seeds = int(run_cfg.get("n_seeds", 5))
    base_seed = int(run_cfg.get("base_seed", 0))

    mode = str(cfg.get("mode", "qfbn")).lower()
    if mode not in ["qfbn", "mlp", "oracle_mask", "oracle_atom"]:
        raise ValueError(f"Unsupported mode={mode}")

    device = get_device(cfg)

    run_dir_name = f"{now_tag()}_{short_uid()}_{exp_name}"
    run_dir = Path(out_dir) / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config_resolved.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    per_seed_rows: List[Dict[str, Any]] = []

    teacher_cfg = cfg.get("teacher", {})
    student_cfg = cfg.get("student", {})
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})

    for si in range(n_seeds):
        seed = base_seed + si
        set_global_seed(seed)

        if is_classification(cfg):
            (x_tr, y_tr, _), (_, _, _), (x_te, y_te, _), meta = make_teacher_student_classification(cfg, seed, device)

            in_dim = int(data_cfg["input_dim"])
            n_classes = int(data_cfg["n_classes"])
            student_hidden = list(student_cfg.get("hidden_dims", teacher_cfg.get("hidden_dims", [])))

            # IMPORTANT: force student atoms to equal teacher(meta) atoms (index alignment)
            f_atoms = list(meta["f_atoms"])
            g_name = str(teacher_cfg.get("g", "tanh"))

            if mode == "mlp":
                model = MLPClassifier(in_dim, student_hidden, n_classes, act="relu").to(device)
                train_mlp_classification(model, x_tr, y_tr, train_cfg, device)
                test_ce, test_acc = eval_classification(model, x_te, y_te)
                atom_acc, mask_f1 = float("nan"), float("nan")
            else:
                dims = [in_dim] + list(student_hidden) + [n_classes]
                model = QFBNNet(
                    dims=dims,
                    f_atoms=f_atoms,
                    g_name=g_name,
                    w_scale=float(student_cfg.get("w_scale", teacher_cfg.get("w_scale", 1.0))),
                    b_scale=float(student_cfg.get("b_scale", teacher_cfg.get("b_scale", 1.0))),
                    device=device,
                ).to(device)

                if mode in ["oracle_mask", "oracle_atom"]:
                    if "zero" not in f_atoms:
                        raise ValueError("Oracle modes require 'zero' in f_atoms.")
                    oracle = OracleSpec(
                        mode=mode,
                        f_atoms=f_atoms,
                        teacher_f_idx_list=meta["f_idx_list"],
                        zero_id=f_atoms.index("zero"),
                    )
                    model.set_oracle(oracle, device)

                train_qfbn_classification(model, x_tr, y_tr, train_cfg, device)
                test_ce, test_acc = eval_classification(model, x_te, y_te)
                pred_idx_list = model.hard_f_idx_list()
                atom_acc, mask_f1 = recovery_metrics(pred_idx_list, meta, device)

            per_seed_rows.append(
                {
                    "seed": seed,
                    "task": "recovery_ce",
                    "mode": mode,
                    "test_loss": float(test_ce),
                    "test_acc": float(test_acc),
                    "recovery_atom_acc": float(atom_acc),
                    "recovery_mask_f1": float(mask_f1),
                }
            )

        else:
            (x_tr, y_tr), (_, _), (x_te, y_te), meta = make_teacher_student_regression(cfg, seed, device)

            in_dim = int(data_cfg["input_dim"])
            out_dim = int(data_cfg["output_dim"])
            student_hidden = list(student_cfg.get("hidden_dims", teacher_cfg.get("hidden_dims", [])))

            f_atoms = list(meta["f_atoms"])
            g_name = str(teacher_cfg.get("g", "tanh"))

            if mode == "mlp":
                model = MLPRegressor(in_dim, student_hidden, out_dim, act="relu").to(device)
                train_mlp_regression(model, x_tr, y_tr, train_cfg, device)
                test_mse = eval_regression(model, x_te, y_te)
                atom_acc, mask_f1 = float("nan"), float("nan")
            else:
                dims = [in_dim] + list(student_hidden) + [out_dim]
                model = QFBNNet(
                    dims=dims,
                    f_atoms=f_atoms,
                    g_name=g_name,
                    w_scale=float(student_cfg.get("w_scale", teacher_cfg.get("w_scale", 1.0))),
                    b_scale=float(student_cfg.get("b_scale", teacher_cfg.get("b_scale", 1.0))),
                    device=device,
                ).to(device)

                if mode in ["oracle_mask", "oracle_atom"]:
                    if "zero" not in f_atoms:
                        raise ValueError("Oracle modes require 'zero' in f_atoms.")
                    oracle = OracleSpec(
                        mode=mode,
                        f_atoms=f_atoms,
                        teacher_f_idx_list=meta["f_idx_list"],
                        zero_id=f_atoms.index("zero"),
                    )
                    model.set_oracle(oracle, device)

                train_qfbn_regression(model, x_tr, y_tr, train_cfg, device)
                test_mse = eval_regression(model, x_te, y_te)
                pred_idx_list = model.hard_f_idx_list()
                atom_acc, mask_f1 = recovery_metrics(pred_idx_list, meta, device)

            per_seed_rows.append(
                {
                    "seed": seed,
                    "task": "recovery_mse",
                    "mode": mode,
                    "test_loss": float(test_mse),
                    "test_acc": float("nan"),
                    "recovery_atom_acc": float(atom_acc),
                    "recovery_mask_f1": float(mask_f1),
                }
            )

    task = per_seed_rows[0]["task"] if per_seed_rows else "unknown"
    test_losses = [r["test_loss"] for r in per_seed_rows]
    atom_accs = [r["recovery_atom_acc"] for r in per_seed_rows if not math.isnan(float(r["recovery_atom_acc"]))]
    mask_f1s = [r["recovery_mask_f1"] for r in per_seed_rows if not math.isnan(float(r["recovery_mask_f1"]))]
    test_accs = [r["test_acc"] for r in per_seed_rows if not math.isnan(float(r["test_acc"]))]

    test_loss_mean, test_loss_std = aggregate_mean_std(test_losses)
    atom_acc_mean, atom_acc_std = aggregate_mean_std(atom_accs)
    mask_f1_mean, mask_f1_std = aggregate_mean_std(mask_f1s)
    test_acc_mean, test_acc_std = aggregate_mean_std(test_accs)

    summary = {
        "run_dir_name": run_dir_name,
        "exp_name": exp_name,
        "task": task,
        "mode": mode,
        "n_seeds": n_seeds,
        "p_zero": cfg.get("teacher", {}).get("p_zero", cfg.get("data", {}).get("p_zero", None)),
        "noise_std": cfg.get("data", {}).get("noise_std", None),
        "logit_noise": cfg.get("data", {}).get("logit_noise", None),
        "test_loss_mean": test_loss_mean,
        "test_loss_std": test_loss_std,
        "recovery_atom_acc_mean": atom_acc_mean,
        "recovery_atom_acc_std": atom_acc_std,
        "recovery_mask_f1_mean": mask_f1_mean,
        "recovery_mask_f1_std": mask_f1_std,
        "test_acc_mean": test_acc_mean,
        "test_acc_std": test_acc_std,
        "run_dir": str(run_dir.resolve()),
    }

    with open(run_dir / "per_seed_metrics.json", "w", encoding="utf-8") as f:
        json.dump(per_seed_rows, f, indent=2, ensure_ascii=False)

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[OK] done. run_dir={run_dir}")
    out_print = {
        "task": summary["task"],
        "mode": summary["mode"],
        "n_seeds": summary["n_seeds"],
        "test_loss_mean": summary["test_loss_mean"],
        "test_loss_std": summary["test_loss_std"],
        "recovery_atom_acc_mean": summary["recovery_atom_acc_mean"],
        "recovery_atom_acc_std": summary["recovery_atom_acc_std"],
        "recovery_mask_f1_mean": summary["recovery_mask_f1_mean"],
        "recovery_mask_f1_std": summary["recovery_mask_f1_std"],
    }
    if not math.isnan(float(summary["test_acc_mean"])):
        out_print["test_acc_mean"] = summary["test_acc_mean"]
        out_print["test_acc_std"] = summary["test_acc_std"]
    print(out_print)


if __name__ == "__main__":
    main()
