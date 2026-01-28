import csv, json, math, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT  = ROOT / "artifacts" / "results.csv"

SKIP_PREFIX = ("_",)

NEED_METRICS = [
    "test_loss_mean",
    "recovery_atom_acc_mean",
    "recovery_mask_f1_mean",
]

# per-seed key candidates (we'll accept either "..._mean" style or per-seed raw names)
PER_SEED_CANDIDATES = {
    "test_loss": ["test_loss", "loss", "test_loss_mean"],
    "recovery_atom_acc": ["recovery_atom_acc", "atom_acc", "recovery_atom_acc_mean"],
    "recovery_mask_f1": ["recovery_mask_f1", "mask_f1", "recovery_mask_f1_mean"],
    "test_acc": ["test_acc", "acc", "test_acc_mean"],
}

def safe_load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def is_finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)

def is_missing_value(x):
    if x is None:
        return True
    if isinstance(x, str):
        s = x.strip().lower()
        return (s == "" or s == "nan" or s == "none" or s == "null" or s == "inf" or s == "-inf")
    return False

def promote_common_nested(d: dict):
    if not isinstance(d, dict):
        return d
    out = dict(d)
    for k in ["metrics", "result", "summary"]:
        v = out.get(k, None)
        if isinstance(v, dict):
            out.pop(k, None)
            for kk, vv in v.items():
                if kk not in out:
                    out[kk] = vv
    return out

def pick_result_file(run_dir: Path):
    for name in ["summary.json", "metrics.json"]:
        p = run_dir / name
        if p.exists():
            return p
    return None

def infer_from_config_yaml(run_dir: Path, key: str):
    # naive parse: look for a line like "mode: qfbn" at any indentation
    p = run_dir / "config.yaml"
    if not p.exists():
        return None
    try:
        txt = p.read_text(encoding="utf-8")
    except Exception:
        return None
    m = re.search(rf"(?m)^\s*{re.escape(key)}\s*:\s*([^\s#]+)", txt)
    if not m:
        return None
    return m.group(1).strip()

def aggregate(vals):
    finite = [v for v in vals if is_finite(v)]
    if not finite:
        return None, None
    mean = sum(finite) / len(finite)
    var = sum((x - mean) ** 2 for x in finite) / len(finite)
    return mean, math.sqrt(var)

def read_per_seed(run_dir: Path):
    p = run_dir / "per_seed.jsonl"
    if not p.exists():
        return []
    out = []
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if isinstance(d, dict):
                    out.append(d)
            except Exception:
                continue
    except Exception:
        return []
    return out

def backfill_from_per_seed(data: dict, run_dir: Path):
    # if any NEED_METRICS missing/empty/non-finite -> try per_seed
    need_backfill = False
    for k in NEED_METRICS:
        if (k not in data) or is_missing_value(data.get(k)) or (isinstance(data.get(k), float) and (not math.isfinite(data.get(k)))):
            need_backfill = True
            break
    if not need_backfill:
        return data

    seeds = read_per_seed(run_dir)
    if not seeds:
        return data

    # collect candidates
    collected = {}
    for base, cands in PER_SEED_CANDIDATES.items():
        vals = []
        for s in seeds:
            v = None
            for kk in cands:
                if kk in s and not is_missing_value(s.get(kk)):
                    v = s.get(kk)
                    break
            if is_finite(v):
                vals.append(float(v))
        if vals:
            mu, sd = aggregate(vals)
            collected[f"{base}_mean"] = mu
            collected[f"{base}_std"]  = sd

    # map to expected names if we got them
    for k in ["test_loss", "recovery_atom_acc", "recovery_mask_f1", "test_acc"]:
        if f"{k}_mean" in collected and (f"{k}_mean" not in data or is_missing_value(data.get(f"{k}_mean"))):
            data[f"{k}_mean"] = collected[f"{k}_mean"]
        if f"{k}_std" in collected and (f"{k}_std" not in data or is_missing_value(data.get(f"{k}_std"))):
            data[f"{k}_std"] = collected[f"{k}_std"]

    data["__backfilled_from_per_seed"] = 1
    return data

def load_one(run_dir: Path):
    p = pick_result_file(run_dir)
    if p is None:
        return None

    d = safe_load_json(p)
    if d is None or not isinstance(d, dict):
        return None

    d = promote_common_nested(d)

    # fill exp_name/run_dir
    d["run_dir"] = str(run_dir).replace("\\", "/")
    d["exp_name"] = run_dir.name
    d["__result_file"] = p.name

    # infer mode/task if missing
    if ("mode" not in d) or is_missing_value(d.get("mode")):
        m = infer_from_config_yaml(run_dir, "mode")
        if m:
            d["mode"] = m
    if ("task" not in d) or is_missing_value(d.get("task")):
        t = infer_from_config_yaml(run_dir, "task")
        if t:
            d["task"] = t

    # backfill key metrics from per_seed if needed
    d = backfill_from_per_seed(d, run_dir)
    return d

rows = []
for d in RUNS.iterdir():
    if d.name.startswith(SKIP_PREFIX):
        continue
    if not d.is_dir():
        continue
    x = load_one(d)
    if x is not None:
        rows.append(x)

# union columns
keys = sorted({k for r in rows for k in r.keys()})

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    for r in rows:
        w.writerow(r)

print(f"[OK] wrote {OUT} with {len(rows)} rows")
