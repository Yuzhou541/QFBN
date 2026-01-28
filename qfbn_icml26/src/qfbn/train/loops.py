from __future__ import annotations
from typing import Dict, Any, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from qfbn.models.fbn_linear import FBNRegularizers
from qfbn.utils.metrics import entropy_from_probs


def _make_loader(x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(x, y)
    # Windows-safe: num_workers=0
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False)


@torch.no_grad()
def _eval_mse(model, loader, tau: float, device: torch.device) -> float:
    model.eval()
    total = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb, tau=tau)
        loss = F.mse_loss(pred, yb, reduction="sum").item()
        total += loss
        n += yb.numel()
    return total / max(n, 1)


@torch.no_grad()
def _eval_ce(model, loader, tau: float, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb, tau=tau)
        loss = F.cross_entropy(logits, yb, reduction="sum").item()
        total_loss += loss
        pred = torch.argmax(logits, dim=-1)
        correct += int((pred == yb).sum().item())
        total += int(yb.numel())
    return total_loss / max(total, 1), correct / max(total, 1)


def train_recovery_mse(
    model,
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_va: torch.Tensor,
    y_va: torch.Tensor,
    cfg_train: Dict[str, Any],
    tau_fn,
    regs: FBNRegularizers,
    device: torch.device,
    eval_every: int = 10,
):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg_train["lr"]), weight_decay=float(cfg_train.get("weight_decay", 0.0)))
    bs = int(cfg_train["batch_size"])
    epochs = int(cfg_train["epochs"])
    grad_clip = float(cfg_train.get("grad_clip", 0.0))

    tr_loader = _make_loader(x_tr, y_tr, bs, True)
    va_loader = _make_loader(x_va, y_va, bs, False)

    best_val = float("inf")
    best_state = None
    curves = []

    for ep in range(epochs):
        model.train()
        tau = float(tau_fn(ep, epochs))
        running = 0.0
        nsum = 0

        for xb, yb in tr_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb, tau=tau)
            loss = F.mse_loss(pred, yb)

            reg_loss, reg_log = model.regularization_loss(tau=tau, regs=regs)
            total_loss = loss + reg_loss
            total_loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            opt.step()

            running += float(loss.detach().cpu()) * xb.shape[0]
            nsum += xb.shape[0]

        if (ep % eval_every) == 0 or ep == epochs - 1:
            val = _eval_mse(model, va_loader, tau=tau, device=device)
            if val < best_val:
                best_val = val
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            # track mean entropy and expected degree on first layer for diagnostics
            layer0 = model.layers[0]
            pf = layer0.f_probs(tau)
            ent = entropy_from_probs(pf).mean().item()
            deg = layer0.expected_degree_f(pf).mean().item()
            nonzero = layer0.expected_nonzero_f(pf).mean().item()

            curves.append({
                "epoch": ep,
                "tau": tau,
                "train_mse": running / max(nsum, 1),
                "val_mse": val,
                "ent_f_layer0": ent,
                "deg_f_layer0": deg,
                "nonzero_f_layer0": nonzero,
            })

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return {"best_val": best_val, "curves": curves}


def train_recovery_ce(
    model,
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_va: torch.Tensor,
    y_va: torch.Tensor,
    cfg_train: Dict[str, Any],
    tau_fn,
    regs: FBNRegularizers,
    device: torch.device,
    eval_every: int = 10,
):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg_train["lr"]), weight_decay=float(cfg_train.get("weight_decay", 0.0)))
    bs = int(cfg_train["batch_size"])
    epochs = int(cfg_train["epochs"])
    grad_clip = float(cfg_train.get("grad_clip", 0.0))

    tr_loader = _make_loader(x_tr, y_tr, bs, True)
    va_loader = _make_loader(x_va, y_va, bs, False)

    best_val = float("inf")
    best_state = None
    curves = []

    for ep in range(epochs):
        model.train()
        tau = float(tau_fn(ep, epochs))
        running = 0.0
        nsum = 0

        for xb, yb in tr_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb, tau=tau)
            loss = F.cross_entropy(logits, yb)

            reg_loss, reg_log = model.regularization_loss(tau=tau, regs=regs)
            total_loss = loss + reg_loss
            total_loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            opt.step()

            running += float(loss.detach().cpu()) * xb.shape[0]
            nsum += xb.shape[0]

        if (ep % eval_every) == 0 or ep == epochs - 1:
            val_loss, val_acc = _eval_ce(model, va_loader, tau=tau, device=device)
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            layer0 = model.layers[0]
            pf = layer0.f_probs(tau)
            ent = entropy_from_probs(pf).mean().item()
            deg = layer0.expected_degree_f(pf).mean().item()
            nonzero = layer0.expected_nonzero_f(pf).mean().item()

            curves.append({
                "epoch": ep,
                "tau": tau,
                "train_ce": running / max(nsum, 1),
                "val_ce": val_loss,
                "val_acc": val_acc,
                "ent_f_layer0": ent,
                "deg_f_layer0": deg,
                "nonzero_f_layer0": nonzero,
            })

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return {"best_val": best_val, "curves": curves}
