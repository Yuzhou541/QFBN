import csv, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INP  = ROOT / "artifacts" / "results.csv"
OUT1 = ROOT / "artifacts" / "main_table.csv"
OUT2 = ROOT / "artifacts" / "main_table.md"

KEEP_MODES = ["qfbn", "mlp", "oracle_mask", "oracle_atom"]
METRIC = "recovery_mask_f1_mean"

def to_float(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "none", "null", "inf", "-inf"):
        return None
    try:
        v = float(s)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None

with INP.open("r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

# normalize mode
for r in rows:
    m = (r.get("mode") or "").strip()
    r["mode"] = m if m else "(empty)"

# filter
use = [r for r in rows if r["mode"] in KEEP_MODES]

need_cols = ["task", "mode", "exp_name", "test_loss_mean", "recovery_atom_acc_mean", METRIC]
for c in need_cols:
    if not any(c in r for r in use):
        raise SystemExit(f"[ERROR] column missing from results.csv: {c}")

# pick best by METRIC per (task, mode)
best = {}
nan_count = 0
for r in use:
    key = (r.get("task","").strip(), r["mode"])
    f1 = to_float(r.get(METRIC))
    if f1 is None:
        nan_count += 1
        continue
    prev = best.get(key)
    if prev is None or f1 > prev["_f1"]:
        rr = dict(r)
        rr["_f1"] = f1
        best[key] = rr

print(f"[INFO] skipped rows with invalid {METRIC}: {nan_count}")

# write csv
out_rows = []
for (task, mode), r in sorted(best.items(), key=lambda x: (x[0][0], x[0][1])):
    out_rows.append({
        "task": task,
        "mode": mode,
        "exp_name": r.get("exp_name",""),
        "test_loss_mean": r.get("test_loss_mean",""),
        "recovery_atom_acc_mean": r.get("recovery_atom_acc_mean",""),
        METRIC: r.get(METRIC,""),
        "run_dir": r.get("run_dir",""),
        "__result_file": r.get("__result_file",""),
        "__backfilled_from_per_seed": r.get("__backfilled_from_per_seed",""),
    })

OUT1.parent.mkdir(parents=True, exist_ok=True)
with OUT1.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else [])
    if out_rows:
        w.writeheader()
        w.writerows(out_rows)

# write markdown
def md_table(rows, headers):
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(h,"")) for h in headers) + " |")
    return "\n".join(lines)

headers = ["task","mode","exp_name","test_loss_mean","recovery_atom_acc_mean",METRIC]
OUT2.write_text(md_table(out_rows, headers), encoding="utf-8")

print(f"[OK] wrote {OUT1}")
print(f"[OK] wrote {OUT2}")
