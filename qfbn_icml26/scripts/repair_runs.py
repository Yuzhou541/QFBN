# scripts/repair_runs.py
import json, math, statistics, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"

REC_KEYS = [
    "recovery_atom_acc_mean", "recovery_atom_acc_std",
    "recovery_mask_f1_mean", "recovery_mask_f1_std",
]

def _is_bad_float(x):
    return isinstance(x, float) and (math.isnan(x) or math.isinf(x))

def sanitize(obj):
    """Recursively replace NaN/Inf with None."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if _is_bad_float(obj):
        return None
    return obj

def read_json_allow_nan(p: Path):
    # Python's json can parse NaN/Infinity by default (non-standard JSON).
    return json.loads(p.read_text(encoding="utf-8"))

def write_json_strict(p: Path, data):
    # ensure no NaN/Inf written back
    txt = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)
    p.write_text(txt + "\n", encoding="utf-8")

def parse_mode_from_config(config_path: Path):
    # best-effort: find top-level "mode: xxx"
    if not config_path.exists():
        return None
    for line in config_path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*mode\s*:\s*([A-Za-z0-9_]+)\s*$", line)
        if m:
            return m.group(1)
    return None

def aggregate_from_per_seed(per_seed_path: Path):
    """Aggregate numeric fields from per_seed.jsonl into a summary dict."""
    lines = per_seed_path.read_text(encoding="utf-8").splitlines()
    per_seed = {}
    for line in lines:
        try:
            d = json.loads(line)
        except Exception:
            continue
        if "seed" not in d:
            continue
        s = int(d["seed"])
        # keep the last record for each seed (common if it logs multiple times)
        per_seed[s] = d

    seeds = sorted(per_seed.keys())
    if not seeds:
        return None, []

    # collect numeric keys
    keys = set()
    for d in per_seed.values():
        keys |= set(d.keys())

    out = {}
    out["n_seeds"] = len(seeds)

    # propagate some common identifiers if present
    for k in ["task", "mode"]:
        for s in seeds:
            if k in per_seed[s]:
                out[k] = per_seed[s][k]
                break

    # aggregate numeric scalars
    for k in sorted(keys):
        if k in ["seed"]:
            continue
        vals = []
        for s in seeds:
            v = per_seed[s].get(k, None)
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                vals.append(float(v))
        if len(vals) == len(seeds) and len(vals) > 0:
            # store per-seed mean/std as *_mean/*_std if scalar, else keep original name if already mean
            # If key already endswith _mean/_std, just keep it as-is by averaging across seeds.
            out[f"{k}_mean"] = statistics.mean(vals)
            out[f"{k}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0

    out["seeds"] = seeds
    return out, seeds

def main():
    repaired = 0
    sanitized = 0
    missing = 0

    for d in RUNS.iterdir():
        if not d.is_dir():
            continue
        if d.name.startswith("_"):
            continue  # exclude _aggregate etc.

        summary_p = d / "summary.json"
        metrics_p = d / "metrics.json"
        config_p = d / "config.yaml"
        per_seed_p = d / "per_seed.jsonl"

        has_result = summary_p.exists() or metrics_p.exists()

        # 1) If missing results but has per_seed, backfill summary.json
        if not has_result:
            if per_seed_p.exists():
                agg, seeds = aggregate_from_per_seed(per_seed_p)
                if agg is not None:
                    # best-effort mode from config if not present
                    agg.setdefault("mode", parse_mode_from_config(config_p))
                    agg["run_dir"] = str(d).replace("\\", "/")
                    agg["exp_name"] = d.name
                    write_json_strict(summary_p, sanitize(agg))
                    repaired += 1
                else:
                    missing += 1
            else:
                missing += 1
            continue

        # 2) Sanitize existing json(s)
        for p in [summary_p, metrics_p]:
            if not p.exists():
                continue
            data = read_json_allow_nan(p)
            data = sanitize(data)

            mode = data.get("mode") or parse_mode_from_config(config_p)
            if mode == "mlp":
                # set recovery metrics to N/A (null) rather than NaN
                for k in REC_KEYS:
                    if k in data:
                        data[k] = None

            # keep helpful identifiers
            data.setdefault("run_dir", str(d).replace("\\", "/"))
            data.setdefault("exp_name", d.name)

            write_json_strict(p, data)
            sanitized += 1

    print("[OK] repair_runs done.")
    print({"repaired_from_per_seed": repaired, "sanitized_files": sanitized, "still_missing_runs": missing})

if __name__ == "__main__":
    main()
