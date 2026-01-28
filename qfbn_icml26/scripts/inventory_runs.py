import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUTC = ROOT / "artifacts" / "run_inventory.csv"
OUTM = ROOT / "artifacts" / "run_inventory.md"

def count_files(d: Path, pattern: str) -> int:
    return sum(1 for _ in d.rglob(pattern))

def exists(d: Path, rel: str) -> int:
    return 1 if (d / rel).exists() else 0

def main():
    rows = []
    for d in RUNS.iterdir():
        if not d.is_dir():
            continue
        if d.name.startswith("_"):
            continue

        rows.append({
            "exp_name": d.name,
            "run_dir": str(d).replace("\\", "/"),
            "has_config_yaml": exists(d, "config.yaml"),
            "has_summary_json": exists(d, "summary.json"),
            "has_metrics_json": exists(d, "metrics.json"),
            "has_per_seed_jsonl": exists(d, "per_seed.jsonl"),
            "n_curves_csv": count_files(d, "curves_*.csv"),
            "n_plots_files": count_files(d / "plots", "*") if (d / "plots").exists() else 0,
        })

    OUTC.parent.mkdir(parents=True, exist_ok=True)
    keys = ["exp_name","run_dir","has_config_yaml","has_summary_json","has_metrics_json","has_per_seed_jsonl","n_curves_csv","n_plots_files"]

    with OUTC.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # markdown (first 80 lines only to keep it readable)
    lines = []
    lines.append("| " + " | ".join(keys) + " |")
    lines.append("| " + " | ".join(["---"]*len(keys)) + " |")
    for r in rows[:200]:  # you can increase if you want
        lines.append("| " + " | ".join(str(r.get(k,"")) for k in keys) + " |")
    OUTM.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] wrote {OUTC} with {len(rows)} rows")
    print(f"[OK] wrote {OUTM} (preview table)")

if __name__ == "__main__":
    main()
