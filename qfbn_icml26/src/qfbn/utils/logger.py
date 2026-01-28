from __future__ import annotations
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict

from .io import ensure_dir, dump_yaml


@dataclass
class RunLogger:
    run_dir: str

    def path(self, rel: str) -> str:
        return os.path.join(self.run_dir, rel)


def make_run_logger(out_dir: str, exp_name: str, config: Dict[str, Any]) -> RunLogger:
    """
    Creates: runs/YYYYMMDD_HHMMSS_<8hex>/
    Saves: config.yaml
    """
    ensure_dir(out_dir)
    ts = time.strftime("%Y%m%d_%H%M%S")
    rid = uuid.uuid4().hex[:8]
    run_dir = os.path.join(out_dir, f"{ts}_{rid}_{exp_name}")
    ensure_dir(run_dir)

    dump_yaml(config, os.path.join(run_dir, "config.yaml"))
    ensure_dir(os.path.join(run_dir, "plots"))
    return RunLogger(run_dir=run_dir)
