# scripts/mkcfg.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import yaml


def _parse_scalar(v: str) -> Any:
    s = v.strip()
    if s.lower() in {"true", "false"}:
        return s.lower() == "true"
    # int
    try:
        if s.startswith("0") and s != "0" and not s.startswith("0."):
            # keep as string for things like IDs
            raise ValueError
        return int(s)
    except Exception:
        pass
    # float
    try:
        return float(s)
    except Exception:
        pass
    # string
    return s


def _set_by_path(d: Dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    cur: Dict[str, Any] = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, required=True, help="base yaml")
    ap.add_argument("--out", type=str, required=True, help="output yaml")
    ap.add_argument("--set", action="append", default=[], help="k=v, dot-path supported")
    args = ap.parse_args()

    base_p = Path(args.base)
    out_p = Path(args.out)

    cfg = yaml.safe_load(base_p.read_text(encoding="utf-8"))
    if cfg is None:
        cfg = {}

    # apply overrides
    for kv in args.set:
        if "=" not in kv:
            raise ValueError(f"--set expects k=v, got: {kv}")
        k, v = kv.split("=", 1)
        _set_by_path(cfg, k.strip(), _parse_scalar(v))

    out_p.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


if __name__ == "__main__":
    main()
