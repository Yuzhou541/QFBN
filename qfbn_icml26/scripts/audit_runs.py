import json, csv, math, re
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT  = ROOT / "artifacts" / "audit_runs.csv"

REQUIRED_METRICS_ANY = [
    "test_loss_mean",
    "recovery_atom_acc_mean",
    "recovery_mask_f1_mean",
]

SKIP_PREFIX = ("_",)  # skip _aggregate, _failed_*, etc.

def safe_load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def is_finite_number(x):
    return isinstance(x, (int, float)) and math.isfinite(x)

def is_missing_value(x):
    if x is None:
        return True
    if isinstance(x, str):
        s = x.strip().lower()
        return (s == "" or s == "nan" or s == "none" or s == "null" or s == "inf" or s == "-inf")
    return False

def promote_common_nested(d: dict):
    """
    Some runs store metrics under keys like "metrics" or "result".
    Promote them to top-level to stabilize downstream scripts.
    """
    if not isinstance(d, dict):
        return d
    out = dict(d)
    for k in ["metrics", "result", "summary"]:
        v = out.get(k, None)
        if isinstance(v, dict):
            out.pop(k, None)
            # do not overwrite existing top-level keys
            for kk, vv in v.items():
                if kk not in out:
                    out[kk] = vv
    return out

def pick_result_file(run_dir: Path):
    # prefer summary.json if both exist
    for name in ["summary.json", "metrics.json"]:
        p = run_dir / name
        if p.exists():
            return p
    return None

rows = []
for d in RUNS.iterdir():
    if d.name.startswith(SKIP_PREFIX):
        continue
    if not d.is_dir():
        continue

    p = pick_result_file(d)
    if p is None:
        rows.append({
            "exp_name": d.name,
            "run_dir": str(d).replace("\\", "/"),
            "status": "missing_results",
            "result_file": "",
            "parse_ok": 0,
        })
        continue

    data = safe_load_json(p)
    if data is None or not isinstance(data, dict):
        rows.append({
            "exp_name": d.name,
            "run_dir": str(d).replace("\\", "/"),
            "status": "json_parse_failed",
            "result_file": p.name,
            "parse_ok": 0,
        })
        continue

    data = promote_common_nested(data)

    # minimal sanity checks: treat missing OR empty as missing
    status = "ok"
    missing_keys = [k for k in REQUIRED_METRICS_ANY if (k not in data) or is_missing_value(data.get(k))]
    if missing_keys:
        status = "missing_metrics:" + ",".join(missing_keys)

    # numeric sanity (only check keys that exist and are not "missing_value")
    bad_nums = []
    for k in REQUIRED_METRICS_ANY:
        if k in data and (not is_missing_value(data.get(k))) and (not is_finite_number(data.get(k))):
            bad_nums.append(k)
    if bad_nums:
        status = "non_finite:" + ",".join(bad_nums)

    row = {
        "exp_name": d.name,
        "run_dir": str(d).replace("\\", "/"),
        "status": status,
        "result_file": p.name,
        "parse_ok": 1,
    }

    # include a few commonly used fields if present
    for k in ["task", "mode", "n_seeds"] + REQUIRED_METRICS_ANY:
        if k in data:
            row[k] = data[k]
    rows.append(row)

OUT.parent.mkdir(parents=True, exist_ok=True)
keys = sorted({k for r in rows for k in r.keys()})
with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)

c = Counter(r["status"] for r in rows)
print("[SUMMARY]", dict(c))
print(f"[OK] wrote {OUT} with {len(rows)} rows")
