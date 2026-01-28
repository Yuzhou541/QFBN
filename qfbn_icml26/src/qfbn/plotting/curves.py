from __future__ import annotations

from typing import List, Dict, Any
import os
import csv
import json
from datetime import datetime


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _save_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_curves(curves: List[Dict[str, Any]], out_dir: str, tag: str) -> None:
    """
    IMPORTANT:
    This project must run even when matplotlib is broken (NumPy2 vs mpl built with NumPy1).
    Therefore, we DO NOT import matplotlib at all.

    Output:
      - curves_{tag}.csv : always
      - curves_{tag}.json : always (for robust downstream plotting)
      - plot_disabled_{tag}.txt : always (documents why PNG is not generated)
    """
    _ensure_dir(out_dir)

    # Always write CSV
    csv_path = os.path.join(out_dir, f"curves_{tag}.csv")
    _save_csv(curves, csv_path)

    # Also write JSON (safer for types / missing fields)
    json_path = os.path.join(out_dir, f"curves_{tag}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(curves, f, indent=2)

    # Write a note explaining why we didn't plot
    note_path = os.path.join(out_dir, f"plot_disabled_{tag}.txt")
    with open(note_path, "w", encoding="utf-8") as f:
        f.write("PNG plotting is disabled by design to keep experiments runnable.\n")
        f.write("Reason: current environment has NumPy 2.x while matplotlib binary appears built for NumPy 1.x.\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"CSV: {csv_path}\n")
        f.write(f"JSON: {json_path}\n")
