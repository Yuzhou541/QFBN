from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Tuple, List

import torch
import torch.nn.functional as F

from qfbn.basis.atoms import build_f_atoms, build_g_atoms


def _sample_x(n: int, d: int, x_dist: str, scale: float, device: torch.device) -> torch.Tensor:
    x_dist = x_dist.lower()
    if x_dist in ("normal", "gauss", "gaussian"):
        return torch.randn(n, d, device=device) * scale
    if x_dist in ("uniform", "uni"):
        return (torch.rand(n, d, device=device) * 2.0 - 1.0) * scale
    raise ValueError(f"Unknown x_dist={x_dist}")


def _teacher_cache_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    tcfg = dict(cfg.get("teacher", {}))
    return dict(tcfg.get("cache", {}))


def _canonicalize(obj: Any) -> Any:
    """Make obj JSON-stable."""
    if isinstance(obj, dict):
        return {str(k): _canonicalize(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(x) for x in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


def _teacher_cache_key(cfg: Dict[str, Any], task_tag: str, seed: int, dims: List[int]) -> str:
    tcfg = dict(cfg["teacher"])
    cache = _teacher_cache_cfg(cfg)

    scope = str(cache.get("scope", "per_seed")).lower()
    shared_id = str(cache.get("shared_id", "teacher_v1"))

    payload = {
        "task": task_tag,
        "dims": dims,
        "f_atoms": list(tcfg["f_atoms"]),
        "g": str(tcfg["g"]),
        "w_scale": float(tcfg.get("w_scale", 1.0)),
        "b_scale": float(tcfg.get("b_scale", 1.0)),
        "p_zero": float(tcfg.get("p_zero", 0.0)),
        "atom_probs": dict(tcfg.get("atom_probs", {})),
        "scope": scope,
        "shared_id": shared_id if scope == "shared" else None,
        "seed": int(seed) if scope == "per_seed" else None,
    }
    s = json.dumps(_canonicalize(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _teacher_cache_path(cfg: Dict[str, Any], key: str) -> str:
    cache = _teacher_cache_cfg(cfg)
    cache_dir = str(cache.get("dir", "teacher_cache"))
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"teacher_{key}.pt")


def _maybe_load_teacher(cfg: Dict[str, Any], key: str, device: torch.device) -> Dict[str, Any] | None:
    cache = _teacher_cache_cfg(cfg)
    if not bool(cache.get("enabled", False)):
        return None
    path = _teacher_cache_path(cfg, key)
    if not os.path.exists(path):
        return None
    pack = torch.load(path, map_location="cpu")
    # move tensors to device
    pack["W_list"] = [w.to(device) for w in pack["W_list"]]
    pack["b_list"] = [b.to(device) for b in pack["b_list"]]
    pack["f_idx_list"] = [idx.to(device) for idx in pack["f_idx_list"]]
    return pack


def _maybe_save_teacher(cfg: Dict[str, Any], key: str, pack: Dict[str, Any]) -> str | None:
    cache = _teacher_cache_cfg(cfg)
    if not bool(cache.get("enabled", False)):
        return None
    path = _teacher_cache_path(cfg, key)

    # always save CPU tensors for portability
    cpu_pack = {
        "dims": list(pack["dims"]),
        "f_atoms": list(pack["f_atoms"]),
        "g": str(pack["g"]),
        "W_list": [w.detach().cpu() for w in pack["W_list"]],
        "b_list": [b.detach().cpu() for b in pack["b_list"]],
        "f_idx_list": [idx.detach().cpu() for idx in pack["f_idx_list"]],
    }
    torch.save(cpu_pack, path)
    return path


def _sample_teacher_f_indices(
    outd: int,
    ind: int,
    f_atoms: List[str],
    p_zero: float,
    atom_probs: Dict[str, float],
    device: torch.device,
) -> torch.Tensor:
    """
    Sample per-edge atom index for teacher f.
    - 'zero' atom handled by p_zero
    - non-zero atoms sampled by atom_probs (over remaining atoms)
    """
    A = len(f_atoms)
    zero_idx = f_atoms.index("zero") if "zero" in f_atoms else None
    f_idx = torch.empty(outd, ind, dtype=torch.long, device=device)

    if zero_idx is None:
        # no zero atom; sample all from atom_probs over f_atoms
        probs = torch.tensor([float(atom_probs.get(a, 0.0)) for a in f_atoms], device=device)
        probs = probs / probs.sum().clamp_min(1e-12)
        sampled = torch.multinomial(probs, num_samples=outd * ind, replacement=True)
        return sampled.view(outd, ind)

    # decide zero mask
    m_zero = (torch.rand(outd, ind, device=device) < float(p_zero))
    f_idx[m_zero] = int(zero_idx)

    # sample remaining from non-zero atoms
    nonzero_atoms = [a for a in f_atoms if a != "zero"]
    nonzero_indices = [f_atoms.index(a) for a in nonzero_atoms]
    probs = torch.tensor([float(atom_probs.get(a, 0.0)) for a in nonzero_atoms], device=device)
    probs = probs / probs.sum().clamp_min(1e-12)

    m_nz = ~m_zero
    if m_nz.any():
        sampled = torch.multinomial(probs, num_samples=int(m_nz.sum().item()), replacement=True)
        sampled_global = torch.tensor([nonzero_indices[i] for i in sampled.tolist()], device=device, dtype=torch.long)
        f_idx[m_nz] = sampled_global
    return f_idx


def make_teacher_student_regression(cfg: Dict[str, Any], seed: int, device: torch.device):
    """
    Returns:
      (x_tr,y_tr),(x_va,y_va),(x_te,y_te),
      teacher_meta: includes f_idx per layer and optional cache_path
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    data = cfg["data"]
    tcfg = cfg["teacher"]

    input_dim = int(data["input_dim"])
    output_dim = int(data["output_dim"])
    hidden_dims = list(tcfg["hidden_dims"])

    f_atoms = list(tcfg["f_atoms"])
    g_name = str(tcfg["g"])

    atom_fns = build_f_atoms(f_atoms)
    g_fns = build_g_atoms(["zero", "id", "relu", "tanh", "sigmoid"])
    if g_name not in g_fns:
        raise ValueError(f"Teacher g={g_name} not in supported g atoms.")
    g_fn = g_fns[g_name]

    dims = [input_dim] + hidden_dims + [output_dim]
    key = _teacher_cache_key(cfg, task_tag="regression", seed=seed, dims=dims)
    loaded = _maybe_load_teacher(cfg, key, device)

    if loaded is None:
        W_list, b_list, f_idx_list = [], [], []
        for li in range(len(dims) - 1):
            outd, ind = dims[li + 1], dims[li]
            W = torch.randn(outd, ind, device=device) * float(tcfg["w_scale"])
            b = torch.randn(outd, device=device) * float(tcfg["b_scale"])
            f_idx = _sample_teacher_f_indices(
                outd, ind, f_atoms, float(tcfg["p_zero"]), dict(tcfg["atom_probs"]), device
            )
            W_list.append(W)
            b_list.append(b)
            f_idx_list.append(f_idx)

        cache_path = _maybe_save_teacher(
            cfg,
            key,
            {"dims": dims, "f_atoms": f_atoms, "g": g_name, "W_list": W_list, "b_list": b_list, "f_idx_list": f_idx_list},
        )
    else:
        W_list = loaded["W_list"]
        b_list = loaded["b_list"]
        f_idx_list = loaded["f_idx_list"]
        cache_path = _teacher_cache_path(cfg, key)

    def teacher_forward(x: torch.Tensor) -> torch.Tensor:
        h = x
        for li in range(len(W_list)):
            W = W_list[li]
            b = b_list[li]
            idx = f_idx_list[li]
            W_eff = torch.zeros_like(W)
            for ai, name in enumerate(f_atoms):
                mask = (idx == ai)
                if mask.any():
                    W_eff[mask] = atom_fns[name](W[mask])
            z = h.unsqueeze(1) * W_eff.unsqueeze(0)  # (B,out,in)
            gz = g_fn(z) if li < len(W_list) - 1 else z
            h = gz.sum(dim=-1) + b.unsqueeze(0)
        return h

    x_tr = _sample_x(int(data["n_train"]), input_dim, str(data["x_dist"]), float(data["x_scale"]), device)
    x_va = _sample_x(int(data["n_val"]), input_dim, str(data["x_dist"]), float(data["x_scale"]), device)
    x_te = _sample_x(int(data["n_test"]), input_dim, str(data["x_dist"]), float(data["x_scale"]), device)

    y_tr = teacher_forward(x_tr)
    y_va = teacher_forward(x_va)
    y_te = teacher_forward(x_te)

    noise_std = float(data.get("noise_std", 0.0))
    if noise_std > 0:
        # no generator passed => Windows CUDA safe
        y_tr = y_tr + noise_std * torch.randn_like(y_tr)
        y_va = y_va + noise_std * torch.randn_like(y_va)
        y_te = y_te + noise_std * torch.randn_like(y_te)

    meta = {
        "task": "regression",
        "f_atoms": f_atoms,
        "g": g_name,
        "dims": dims,
        "f_idx_list": [t.detach().clone() for t in f_idx_list],
        "teacher_cache_key": key,
        "teacher_cache_path": cache_path,
    }
    return (x_tr, y_tr), (x_va, y_va), (x_te, y_te), meta


def make_teacher_student_classification(cfg: Dict[str, Any], seed: int, device: torch.device):
    """
    Teacher outputs logits; labels are argmax(softmax(logits)).
    Return train/val/test splits with labels and teacher f_idx.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    data = cfg["data"]
    tcfg = cfg["teacher"]

    input_dim = int(data["input_dim"])
    n_classes = int(data["n_classes"])
    hidden_dims = list(tcfg["hidden_dims"])

    f_atoms = list(tcfg["f_atoms"])
    g_name = str(tcfg["g"])

    atom_fns = build_f_atoms(f_atoms)
    g_fns = build_g_atoms(["zero", "id", "relu", "tanh", "sigmoid"])
    if g_name not in g_fns:
        raise ValueError(f"Teacher g={g_name} not in supported g atoms.")
    g_fn = g_fns[g_name]

    dims = [input_dim] + hidden_dims + [n_classes]
    key = _teacher_cache_key(cfg, task_tag="classification", seed=seed, dims=dims)
    loaded = _maybe_load_teacher(cfg, key, device)

    if loaded is None:
        W_list, b_list, f_idx_list = [], [], []
        for li in range(len(dims) - 1):
            outd, ind = dims[li + 1], dims[li]
            W = torch.randn(outd, ind, device=device) * float(tcfg["w_scale"])
            b = torch.randn(outd, device=device) * float(tcfg["b_scale"])
            f_idx = _sample_teacher_f_indices(
                outd, ind, f_atoms, float(tcfg["p_zero"]), dict(tcfg["atom_probs"]), device
            )
            W_list.append(W)
            b_list.append(b)
            f_idx_list.append(f_idx)

        cache_path = _maybe_save_teacher(
            cfg,
            key,
            {"dims": dims, "f_atoms": f_atoms, "g": g_name, "W_list": W_list, "b_list": b_list, "f_idx_list": f_idx_list},
        )
    else:
        W_list = loaded["W_list"]
        b_list = loaded["b_list"]
        f_idx_list = loaded["f_idx_list"]
        cache_path = _teacher_cache_path(cfg, key)

    def teacher_logits(x: torch.Tensor) -> torch.Tensor:
        h = x
        for li in range(len(W_list)):
            W = W_list[li]
            b = b_list[li]
            idx = f_idx_list[li]
            W_eff = torch.zeros_like(W)
            for ai, name in enumerate(f_atoms):
                mask = (idx == ai)
                if mask.any():
                    W_eff[mask] = atom_fns[name](W[mask])
            z = h.unsqueeze(1) * W_eff.unsqueeze(0)  # (B,out,in)
            gz = g_fn(z) if li < len(W_list) - 1 else z
            h = gz.sum(dim=-1) + b.unsqueeze(0)
        return h  # logits

    x_tr = _sample_x(int(data["n_train"]), input_dim, str(data["x_dist"]), float(data["x_scale"]), device)
    x_va = _sample_x(int(data["n_val"]), input_dim, str(data["x_dist"]), float(data["x_scale"]), device)
    x_te = _sample_x(int(data["n_test"]), input_dim, str(data["x_dist"]), float(data["x_scale"]), device)

    l_tr = teacher_logits(x_tr)
    l_va = teacher_logits(x_va)
    l_te = teacher_logits(x_te)

    logit_noise = float(data.get("logit_noise", 0.0))
    if logit_noise > 0:
        l_tr = l_tr + logit_noise * torch.randn_like(l_tr)
        l_va = l_va + logit_noise * torch.randn_like(l_va)
        l_te = l_te + logit_noise * torch.randn_like(l_te)

    y_tr = torch.argmax(F.softmax(l_tr, dim=-1), dim=-1)
    y_va = torch.argmax(F.softmax(l_va, dim=-1), dim=-1)
    y_te = torch.argmax(F.softmax(l_te, dim=-1), dim=-1)

    meta = {
        "task": "classification",
        "f_atoms": f_atoms,
        "g": g_name,
        "dims": dims,
        "f_idx_list": [t.detach().clone() for t in f_idx_list],
        "teacher_cache_key": key,
        "teacher_cache_path": cache_path,
    }
    return (x_tr, y_tr, l_tr), (x_va, y_va, l_va), (x_te, y_te, l_te), meta
