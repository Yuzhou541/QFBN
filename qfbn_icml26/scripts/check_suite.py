import csv
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
RES  = ROOT / "artifacts" / "results.csv"

def parse_float(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s)
    except Exception:
        return None

def parse_int(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return int(float(s))
    except Exception:
        return None

if not RES.exists():
    raise SystemExit(f"[ERR] missing file: {RES}")

with RES.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames or []
    rows = list(reader)

print(f"[OK] loaded {len(rows)} rows from {RES}")


need_cols = ["exp_name", "run_dir"]
missing_cols = [c for c in need_cols if c not in fieldnames]
if missing_cols:
    print(f"[ERR] results.csv missing columns: {missing_cols}")
else:
    print(f"[OK] required columns present: {need_cols}")


for c in ["task", "mode", "n_seeds"]:
    if c not in fieldnames:
        print(f"[WARN] column not found: {c}")


if "task" in fieldnames and "mode" in fieldnames:
    cnt = Counter((r.get("task", ""), r.get("mode", "")) for r in rows)
    print("\n[COUNT] runs by (task, mode):")
    for (t, m), k in sorted(cnt.items(), key=lambda x: x[1], reverse=True):
        print(f"  {k:4d}  task={t or '(empty)'}  mode={m or '(empty)'}")
else:
    print("\n[SKIP] no task/mode columns, cannot group by (task, mode).")


if "n_seeds" in fieldnames:
    bad = []
    for r in rows:
        ns = parse_int(r.get("n_seeds"))
        if ns is None or ns != 5:
            bad.append((r.get("exp_name",""), r.get("task",""), r.get("mode",""), r.get("n_seeds",""), r.get("run_dir","")))
    print(f"\n[CHECK] n_seeds != 5 (or missing): {len(bad)}")
    if bad:
        print("  showing first 20:")
        for x in bad[:20]:
            print(f"   - exp={x[0]} task={x[1]} mode={x[2]} n_seeds={x[3]} run={x[4]}")
else:
    print("\n[SKIP] no n_seeds column.")


metrics = ["test_loss_mean","test_loss_std","recovery_atom_acc_mean","recovery_atom_acc_std",
           "recovery_mask_f1_mean","recovery_mask_f1_std","test_acc_mean","test_acc_std"]
present = [m for m in metrics if m in fieldnames]
missing = [m for m in metrics if m not in fieldnames]
print("\n[METRICS] present:", present)
if missing:
    print("[METRICS] missing:", missing)


if "task" in fieldnames and "recovery_mask_f1_mean" in fieldnames:
    buckets = defaultdict(list)
    for r in rows:
        t = r.get("task", "")
        f1 = parse_float(r.get("recovery_mask_f1_mean"))
        if t != "" and f1 is not None:
            buckets[t].append((f1, r))

    print("\n[TOP] by recovery_mask_f1_mean (per task):")
    for t in sorted(buckets.keys()):
        lst = sorted(buckets[t], key=lambda x: x[0], reverse=True)[:10]
        print(f"\nTask={t}  (show top {len(lst)})")
        for rank, (f1, r) in enumerate(lst, 1):
            mode = r.get("mode", "")
            atom = r.get("recovery_atom_acc_mean", "")
            loss = r.get("test_loss_mean", "")
            exp  = r.get("exp_name", "")
            run  = r.get("run_dir", "")
            print(f"  #{rank:02d} f1={f1:.6f}  atom={atom}  loss={loss}  mode={mode}  exp={exp}")

else:
    print("\n[SKIP] no task or recovery_mask_f1_mean column.")
# ---- EXTRA: Top per (task, mode), excluding oracle if desired ----
if "task" in fieldnames and "mode" in fieldnames and "recovery_mask_f1_mean" in fieldnames:
    keep_modes = {"qfbn", "mlp", "oracle_mask", "oracle_atom"}  
    best = defaultdict(list)
    for r in rows:
        t = r.get("task","")
        m = r.get("mode","")
        if t == "" or m == "" or m not in keep_modes:
            continue
        f1 = parse_float(r.get("recovery_mask_f1_mean"))
        if f1 is None:
            continue
        best[(t,m)].append((f1, r))

    print("\n[TOP PER MODE] by recovery_mask_f1_mean:")
    for (t,m) in sorted(best.keys()):
        lst = sorted(best[(t,m)], key=lambda x: x[0], reverse=True)[:3]
        print(f"\nTask={t}  Mode={m}  (top {len(lst)})")
        for rank,(f1,r) in enumerate(lst,1):
            print(f"  #{rank:02d} f1={f1:.6f} atom={r.get('recovery_atom_acc_mean','')} "
                  f"loss={r.get('test_loss_mean','')} exp={r.get('exp_name','')}")

print("\n[DONE]")
