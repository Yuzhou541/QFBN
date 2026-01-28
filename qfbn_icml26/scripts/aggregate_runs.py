# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


FIELDS = [
    "run_dir_name",
    "exp_name",
    "task",
    "mode",
    "n_seeds",
    "p_zero",
    "noise_std",
    "logit_noise",
    "test_loss_mean",
    "test_loss_std",
    "recovery_atom_acc_mean",
    "recovery_atom_acc_std",
    "recovery_mask_f1_mean",
    "recovery_mask_f1_std",
    "test_acc_mean",
    "test_acc_std",
    "teacher_g",
    "teacher_hidden_dims",
    "student_hidden_dims",
    "tau_schedule",
    "run_dir",
]


def _find_summary_json(run_dir: Path) -> Optional[Path]:
    for name in ["summary.json", "metrics.json", "result.json"]:
        p = run_dir / name
        if p.exists():
            return p
    return None


def _read_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=str, default="runs", help="Runs root directory")
    ap.add_argument("--out", type=str, default="runs/_aggregate/summary_all_runs.csv")
    ap.add_argument("--dedup", type=str, default="latest", choices=["none", "latest"])
    args = ap.parse_args()

    runs_root = Path(args.runs)
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for d in runs_root.iterdir():
        if not d.is_dir():
            continue
        sj = _find_summary_json(d)
        if sj is None:
            continue
        try:
            row = _read_json(sj)
            # normalize
            if "run_dir_name" not in row:
                row["run_dir_name"] = d.name
            if "run_dir" not in row:
                row["run_dir"] = str(d.resolve())
            if "mode" not in row:
                row["mode"] = "qfbn"
            rows.append(row)
        except Exception:
            continue

    # dedup by exp_name
    if args.dedup == "latest":
        best = {}
        for r in rows:
            key = str(r.get("exp_name", ""))
            # use run_dir_name lexicographic timestamp prefix
            rid = str(r.get("run_dir_name", ""))
            if key not in best or rid > str(best[key].get("run_dir_name", "")):
                best[key] = r
        rows = list(best.values())

    rows.sort(key=lambda r: str(r.get("run_dir_name", "")))

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            out = {}
            for k in FIELDS:
                v = r.get(k, "")
                # keep dict/list as JSON string
                if isinstance(v, (dict, list)):
                    v = json.dumps(v, ensure_ascii=False)
                out[k] = v
            w.writerow(out)

    print(f"[OK] wrote {len(rows)} rows to: {out_csv.as_posix()}")


if __name__ == "__main__":
    main()
