import json
from pathlib import Path

p = Path(r"runs/20260121_222541_c4e4b033_synth_recovery_mse/per_seed.jsonl")
seeds = set()
n_lines = 0

for line in p.read_text(encoding="utf-8").splitlines():
    n_lines += 1
    try:
        d = json.loads(line)
    except Exception:
        continue
    if "seed" in d:
        seeds.add(int(d["seed"]))

print("seeds:", sorted(seeds))
print("n_lines:", n_lines)
