# -*- coding: utf-8 -*-
"""
scripts/run_sweeps.py

One-click runner for ICML-style sweeps:
- p_zero sweep (main): modes = qfbn/mlp/oracle_mask/oracle_atom, both CE & MSE
- logit-noise sweep (appendix): CE
- output-noise sweep (appendix): MSE
- scaling sweeps (width/depth): both CE & MSE

Usage (recommended):
  conda run -n qfbn python .\\scripts\\run_sweeps.py --run_all

Or run subsets:
  conda run -n qfbn python .\\scripts\\run_sweeps.py --pzero
  conda run -n qfbn python .\\scripts\\run_sweeps.py --logit_noise
  conda run -n qfbn python .\\scripts\\run_sweeps.py --noise_std
  conda run -n qfbn python .\\scripts\\run_sweeps.py --scale

Notes:
- We set logit_noise/noise_std to 0 for p_zero and scaling sweeps to keep "clean teacher".
- We keep exp_name schema stable so your artifact script can auto-detect groups.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CFG_CE = PROJECT_ROOT / "configs" / "synth_recovery_ce.yaml"
CFG_MSE = PROJECT_ROOT / "configs" / "synth_recovery_mse.yaml"

MODES_DEFAULT = ["qfbn", "mlp", "oracle_mask", "oracle_atom"]

PZERO_VALUES = [0.2, 0.4, 0.6, 0.8]
LOGIT_NOISE_VALUES = [0.0, 0.1, 0.2, 0.4]
NOISE_STD_VALUES = [0.0, 0.05, 0.1, 0.2]

SCALE_WIDTHS = [16, 32, 64, 128]
SCALE_DEPTHS = [1, 2, 3]
BASE_WIDTH_FOR_DEPTH = 64

# For depth>1, we use safer optimization overrides (optional but helpful).
DEPTH_SAFE_OVERRIDES = [
    ("train.lr_w", "5e-4"),
    ("train.lr_logits", "1e-3"),
    ("train.epochs", "600"),
]

# Default anchor p_zero for robustness sweeps
ROBUST_PZERO = 0.6


def _run(cmd: List[str]) -> None:
    print("\n[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _python_module_cmd(module: str, config_path: Path, sets: List[Tuple[str, str]]) -> List[str]:
    cmd = [sys.executable, "-m", module, "--config", str(config_path)]
    for k, v in sets:
        cmd += ["--set", f"{k}={v}"]
    return cmd


def _ensure_files() -> None:
    if not CFG_CE.exists():
        raise FileNotFoundError(f"Missing config: {CFG_CE}")
    if not CFG_MSE.exists():
        raise FileNotFoundError(f"Missing config: {CFG_MSE}")


def run_pzero(modes: List[str]) -> None:
    print("\n=== [SWEEP] p_zero (MAIN) ===")
    for pz in PZERO_VALUES:
        for mode in modes:
            # CE (clean teacher: logit_noise=0)
            exp = f"ce_pzero_{pz}_{mode}"
            sets = [
                ("mode", mode),
                ("teacher.p_zero", str(pz)),
                ("data.logit_noise", "0.0"),
                ("run.exp_name", exp),
                ("run.device", "cuda"),
            ]
            _run(_python_module_cmd("qfbn.experiments.synth_recovery", CFG_CE, sets))

            # MSE (clean teacher: noise_std=0)
            exp = f"mse_pzero_{pz}_{mode}"
            sets = [
                ("mode", mode),
                ("teacher.p_zero", str(pz)),
                ("data.noise_std", "0.0"),
                ("run.exp_name", exp),
                ("run.device", "cuda"),
            ]
            _run(_python_module_cmd("qfbn.experiments.synth_recovery", CFG_MSE, sets))


def run_logit_noise(modes: List[str]) -> None:
    print("\n=== [SWEEP] logit_noise (APPENDIX, CE) ===")
    for ln in LOGIT_NOISE_VALUES:
        for mode in modes:
            exp = f"ce_logitnoise_{ln}_{mode}"
            sets = [
                ("mode", mode),
                ("teacher.p_zero", str(ROBUST_PZERO)),
                ("data.logit_noise", str(ln)),
                ("run.exp_name", exp),
                ("run.device", "cuda"),
            ]
            _run(_python_module_cmd("qfbn.experiments.synth_recovery", CFG_CE, sets))


def run_noise_std(modes: List[str]) -> None:
    print("\n=== [SWEEP] noise_std (APPENDIX, MSE) ===")
    for ns in NOISE_STD_VALUES:
        for mode in modes:
            exp = f"mse_noise_{ns}_{mode}"
            sets = [
                ("mode", mode),
                ("teacher.p_zero", str(ROBUST_PZERO)),
                ("data.noise_std", str(ns)),
                ("run.exp_name", exp),
                ("run.device", "cuda"),
            ]
            _run(_python_module_cmd("qfbn.experiments.synth_recovery", CFG_MSE, sets))


def run_scaling(modes: List[str]) -> None:
    print("\n=== [SWEEP] scaling (WIDTH/DEPTH) ===")

    # WIDTH: teacher & student must match (recovery metrics defined only when shapes match)
    for w in SCALE_WIDTHS:
        for mode in modes:
            # CE
            exp = f"ce_scale_width_{w}_{mode}"
            sets = [
                ("mode", mode),
                ("teacher.hidden_dims", f"[{w}]"),
                ("student.hidden_dims", f"[{w}]"),
                ("teacher.p_zero", str(ROBUST_PZERO)),
                ("data.logit_noise", "0.0"),
                ("run.exp_name", exp),
                ("run.device", "cuda"),
            ]
            _run(_python_module_cmd("qfbn.experiments.synth_recovery", CFG_CE, sets))

            # MSE
            exp = f"mse_scale_width_{w}_{mode}"
            sets = [
                ("mode", mode),
                ("teacher.hidden_dims", f"[{w}]"),
                ("student.hidden_dims", f"[{w}]"),
                ("teacher.p_zero", str(ROBUST_PZERO)),
                ("data.noise_std", "0.0"),
                ("run.exp_name", exp),
                ("run.device", "cuda"),
            ]
            _run(_python_module_cmd("qfbn.experiments.synth_recovery", CFG_MSE, sets))

    # DEPTH: keep width fixed, vary number of hidden layers
    for d in SCALE_DEPTHS:
        hlist = "[" + ",".join([str(BASE_WIDTH_FOR_DEPTH)] * d) + "]"
        for mode in modes:
            extra = []
            if d >= 2:
                extra = DEPTH_SAFE_OVERRIDES[:]  # safer opts for deeper nets

            # CE
            exp = f"ce_scale_depth_{d}_{mode}"
            sets = [
                ("mode", mode),
                ("teacher.hidden_dims", hlist),
                ("student.hidden_dims", hlist),
                ("teacher.p_zero", str(ROBUST_PZERO)),
                ("data.logit_noise", "0.0"),
                ("run.exp_name", exp),
                ("run.device", "cuda"),
            ] + extra
            _run(_python_module_cmd("qfbn.experiments.synth_recovery", CFG_CE, sets))

            # MSE
            exp = f"mse_scale_depth_{d}_{mode}"
            sets = [
                ("mode", mode),
                ("teacher.hidden_dims", hlist),
                ("student.hidden_dims", hlist),
                ("teacher.p_zero", str(ROBUST_PZERO)),
                ("data.noise_std", "0.0"),
                ("run.exp_name", exp),
                ("run.device", "cuda"),
            ] + extra
            _run(_python_module_cmd("qfbn.experiments.synth_recovery", CFG_MSE, sets))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", type=str, nargs="*", default=MODES_DEFAULT, help="Modes to run.")
    parser.add_argument("--pzero", action="store_true", help="Run p_zero sweep (main).")
    parser.add_argument("--logit_noise", action="store_true", help="Run logit-noise sweep (CE appendix).")
    parser.add_argument("--noise_std", action="store_true", help="Run output-noise sweep (MSE appendix).")
    parser.add_argument("--scale", action="store_true", help="Run scaling sweeps (width/depth).")
    parser.add_argument("--run_all", action="store_true", help="Run all sweeps.")
    args = parser.parse_args()

    _ensure_files()

    modes = [m.lower() for m in args.modes]
    for m in modes:
        if m not in ["qfbn", "mlp", "oracle_mask", "oracle_atom"]:
            raise ValueError(f"Unsupported mode: {m}")

    if args.run_all:
        run_pzero(modes)
        run_logit_noise(modes)
        run_noise_std(modes)
        run_scaling(modes)
        return

    any_flag = args.pzero or args.logit_noise or args.noise_std or args.scale
    if not any_flag:
        print("No sweep selected. Use --run_all or one of: --pzero --logit_noise --noise_std --scale")
        return

    if args.pzero:
        run_pzero(modes)
    if args.logit_noise:
        run_logit_noise(modes)
    if args.noise_std:
        run_noise_std(modes)
    if args.scale:
        run_scaling(modes)


if __name__ == "__main__":
    main()
