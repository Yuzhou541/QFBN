from __future__ import annotations
import math
from typing import Callable, Dict, Any


def make_tau_schedule(cfg_tau: Dict[str, Any]) -> Callable[[int, int], float]:
    """
    returns tau(epoch, epochs)->float
    schedule:
      - linear: start + (end-start)*t
      - exp: start*(end/start)^t
    """
    start = float(cfg_tau["start"])
    end = float(cfg_tau["end"])
    mode = str(cfg_tau.get("schedule", "exp"))

    if mode == "linear":
        def tau_fn(ep: int, E: int) -> float:
            t = min(max(ep / max(E - 1, 1), 0.0), 1.0)
            return start + (end - start) * t
        return tau_fn

    if mode == "exp":
        def tau_fn(ep: int, E: int) -> float:
            t = min(max(ep / max(E - 1, 1), 0.0), 1.0)
            if start <= 0 or end <= 0:
                return max(end, 1e-6)
            return start * ((end / start) ** t)
        return tau_fn

    raise ValueError(f"Unknown tau schedule: {mode}")
