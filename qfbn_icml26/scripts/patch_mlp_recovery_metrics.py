import json, math, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"

SKIP_PREFIX = ("_",)

REQ_MEAN = ["recovery_atom_acc_mean", "recovery_mask_f1_mean"]
REQ_STD  = ["recovery_atom_acc_std",  "recovery_mask_f1_std"]

def safe_load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def safe_write_json(p: Path, d: dict):
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

def is_missing_value(x):
    if x is None:
        return True
    if isinstance(x, str):
        s = x.strip().lower()
        return s in ("", "nan", "none", "null", "inf", "-inf")
    return False

def is_finite_number(x):
    return isinstance(x, (int, float)) and math.isfinite(x)

def pick_result_file(run_dir: Path):
    for name in ["summary.json", "metrics.json"]:
        p = run_dir / name
        if p.exists():
            return p
    return None

def infer_from_config_yaml(run_dir: Path, key: str):
    p = run_dir / "config.yaml"
    if not p.exists():
        return None
    try:
        txt = p.read_text(encoding="utf-8")
    except Exception:
        return None
    m = re.search(rf"(?m)^\s*{re.escape(key)}\s*:\s*([^\s#]+)", txt)
    return m.group(1).strip() if m else None

patched = 0
seen_mlp = 0

for d in RUNS.iterdir():
    if not d.is_dir():
        continue
    if d.name.startswith(SKIP_PREFIX):
        continue

    p = pick_result_file(d)
    if p is None:
        continue

    data = safe_load_json(p)
    if not isinstance(data, dict):
        continue

    mode = data.get("mode", None)
    if is_missing_value(mode):
        mode = infer_from_config_yaml(d, "mode")

    if mode != "mlp":
        continue

    seen_mlp += 1
    changed = False


    for k in REQ_MEAN:
        v = data.get(k, None)
        if (k not in data) or is_missing_value(v) or (not is_finite_number(v)):
            data[k] = 0.0
            changed = True


    for k in REQ_STD:
        v = data.get(k, None)
        if (k not in data) or is_missing_value(v) or (not is_finite_number(v)):
            data[k] = 0.0
            changed = True

    if changed:
        data["__patched_mlp_recovery_metrics"] = 1
        safe_write_json(p, data)
        patched += 1

print(f"[OK] mlp runs seen: {seen_mlp}")
print(f"[OK] result files patched: {patched}")
