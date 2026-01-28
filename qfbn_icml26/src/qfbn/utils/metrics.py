from __future__ import annotations
from typing import Dict, List, Tuple, Any
import numpy as np
import torch


def entropy_from_probs(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    p: (..., K)
    returns: (...) entropy
    """
    p = torch.clamp(p, eps, 1.0)
    return -(p * torch.log(p)).sum(dim=-1)


def mask_f1(pred_nonzero: torch.Tensor, true_nonzero: torch.Tensor, eps: float = 1e-12) -> float:
    """
    pred_nonzero, true_nonzero: boolean tensors with same shape.
    """
    pred = pred_nonzero.flatten().float()
    true = true_nonzero.flatten().float()
    tp = (pred * true).sum().item()
    fp = (pred * (1 - true)).sum().item()
    fn = ((1 - pred) * true).sum().item()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return float(f1)


def atom_confusion_and_scores(
    pred_idx: torch.Tensor,
    true_idx: torch.Tensor,
    atom_names: List[str],
) -> Dict[str, Any]:
    """
    pred_idx,true_idx: int tensors same shape
    """
    pred = pred_idx.flatten().cpu().numpy().astype(int)
    true = true_idx.flatten().cpu().numpy().astype(int)
    k = len(atom_names)
    conf = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(true, pred):
        conf[t, p] += 1

    # per-class precision/recall/f1
    scores = {}
    for i, name in enumerate(atom_names):
        tp = conf[i, i]
        fp = conf[:, i].sum() - tp
        fn = conf[i, :].sum() - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (2 * precision * recall / max(precision + recall, 1e-12)) if (precision + recall) > 0 else 0.0
        scores[name] = {"precision": float(precision), "recall": float(recall), "f1": float(f1)}

    acc = float((pred == true).mean())
    return {
        "acc": acc,
        "confusion": conf.tolist(),
        "per_atom": scores,
    }


def summarize_atom_counts(pred_idx: torch.Tensor, atom_names: List[str]) -> Dict[str, int]:
    pred = pred_idx.flatten().cpu().numpy().astype(int)
    counts = {n: 0 for n in atom_names}
    for i in pred:
        counts[atom_names[i]] += 1
    return counts
