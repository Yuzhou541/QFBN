# -*- coding: utf-8 -*-
"""
scripts/make_icml_artifacts.py

Generate ICML-ready tables/plots from runs/_aggregate/summary_all_runs.csv.

Key upgrades:
- Multi-method comparison (qfbn/mlp/oracle_mask/oracle_atom) in the SAME table/plot
- Saves BOTH .png and .pdf plots (LaTeX-friendly)
- Robust grouping by exp_name patterns:
    ce_pzero_{x}_{mode}
    mse_pzero_{x}_{mode}
    ce_logitnoise_{x}_{mode}
    mse_noise_{x}_{mode}
    ce_scale_width_{x}_{mode}
    mse_scale_width_{x}_{mode}
    ce_scale_depth_{x}_{mode}
    mse_scale_depth_{x}_{mode}
- Optional dedup=latest: keep newest run if same exp_name appears multiple times

Usage:
  conda run -n qfbn python .\\scripts\\make_icml_artifacts.py --csv runs\\_aggregate\\summary_all_runs.csv --out runs\\_aggregate\\icml --dedup latest
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# matplotlib is optional; tables should still generate if plotting fails
_PLOT_OK = True
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception as e:  # pragma: no cover
    _PLOT_OK = False
    plt = None  # type: ignore
    _PLOT_IMPORT_ERR = repr(e)


MODE_ORDER = ["qfbn", "mlp", "oracle_mask", "oracle_atom"]
MODE_PRETTY = {
    "qfbn": "QFBN",
    "mlp": "MLP",
    "oracle_mask": "OracleMask",
    "oracle_atom": "OracleAtom",
}

GROUP_SPECS = {
    # group_id: (x_label, x_key_for_caption)
    "ce_pzero": ("$p_{zero}$", "teacher sparsity"),
    "mse_pzero": ("$p_{zero}$", "teacher sparsity"),
    "ce_logitnoise": ("logit noise", "teacher logit noise"),
    "mse_noise": ("noise std", "additive output noise"),
    "ce_scale_width": ("width", "hidden width"),
    "mse_scale_width": ("width", "hidden width"),
    "ce_scale_depth": ("depth", "network depth"),
    "mse_scale_depth": ("depth", "network depth"),
}


@dataclass
class Row:
    exp_name: str
    task: str
    mode: str
    group_id: str
    x: float
    run_dir_name: str
    # metrics
    test_loss_mean: float
    test_loss_std: float
    test_acc_mean: float
    test_acc_std: float
    atom_acc_mean: float
    atom_acc_std: float
    mask_f1_mean: float
    mask_f1_std: float


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def _extract_dt(run_dir_name_or_path: str) -> datetime:
    """
    Expect prefix like 20260122_234651_....
    If missing, return epoch.
    """
    m = re.search(r"(\d{8}_\d{6})", run_dir_name_or_path)
    if not m:
        return datetime(1970, 1, 1)
    try:
        return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
    except Exception:
        return datetime(1970, 1, 1)


def _infer_mode(exp_name: str, mode_field: str) -> str:
    if mode_field:
        m = mode_field.strip().lower()
        if m in MODE_ORDER:
            return m
    # fallback: parse suffix
    for m in MODE_ORDER:
        if exp_name.lower().endswith("_" + m):
            return m
    return "qfbn"


def _infer_group_and_x(exp_name: str) -> Optional[Tuple[str, float]]:
    """
    Parse exp_name patterns, return (group_id, x_value).
    """
    # canonical patterns: <prefix>_<group>_<x>_<mode>
    # Examples:
    # ce_pzero_0.2_qfbn
    # mse_noise_0.1_oracle_mask
    toks = exp_name.split("_")
    if len(toks) < 4:
        return None
    prefix = toks[0].lower()
    group = toks[1].lower()

    # scale_* has 5 tokens: ce_scale_width_16_qfbn
    if group == "scale" and len(toks) >= 5:
        sub = toks[2].lower()  # width/depth
        x_str = toks[3]
        group_id = f"{prefix}_scale_{sub}"
        return (group_id, _safe_float(x_str))

    # normal: ce_pzero_0.2_qfbn ; ce_logitnoise_0.2_qfbn ; mse_noise_0.1_qfbn
    x_str = toks[2]
    group_id = f"{prefix}_{group}"
    if group_id not in GROUP_SPECS:
        return None
    return (group_id, _safe_float(x_str))


def _fmt_pm(mean: float, std: float, sig: int = 4) -> str:
    if math.isnan(mean):
        return "-"
    if math.isnan(std):
        return f"{mean:.{sig}g}"
    return f"{mean:.{sig}g} $\\pm$ {std:.{sig}g}"


def _write_table(
    out_path: Path,
    caption: str,
    label: str,
    x_name: str,
    x_vals: List[float],
    by_mode: Dict[str, Dict[float, Tuple[float, float]]],
    sig: int = 4,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    modes = [m for m in MODE_ORDER if m in by_mode]
    col_spec = "l" + ("c" * len(modes))

    lines: List[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{6pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.05}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\hline")
    header = [x_name] + [MODE_PRETTY[m] for m in modes]
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\hline")

    for xv in x_vals:
        row = [f"{xv:g}"]
        for m in modes:
            mean, std = by_mode[m].get(xv, (float("nan"), float("nan")))
            row.append(_fmt_pm(mean, std, sig=sig))
        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_metric(
    out_dir: Path,
    group_id: str,
    metric_name: str,
    x_vals: List[float],
    by_mode: Dict[str, Dict[float, Tuple[float, float]]],
    x_label: str,
    y_label: str,
    filename_stem: str,
) -> None:
    if not _PLOT_OK:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure()  # type: ignore
    ax = fig.add_subplot(111)  # type: ignore

    for m in MODE_ORDER:
        if m not in by_mode:
            continue
        ys = []
        es = []
        xs = []
        for xv in x_vals:
            mean, std = by_mode[m].get(xv, (float("nan"), float("nan")))
            if math.isnan(mean):
                continue
            xs.append(xv)
            ys.append(mean)
            es.append(0.0 if math.isnan(std) else std)
        if len(xs) == 0:
            continue
        ax.errorbar(xs, ys, yerr=es, marker="o", linewidth=1.5, capsize=2, label=MODE_PRETTY[m])  # type: ignore

    ax.set_xlabel(x_label)  # type: ignore
    ax.set_ylabel(y_label)  # type: ignore
    ax.set_title(f"{group_id}: {metric_name}")  # type: ignore
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)  # type: ignore
    ax.legend()  # type: ignore
    fig.tight_layout()  # type: ignore

    png_path = out_dir / f"{filename_stem}.png"
    pdf_path = out_dir / f"{filename_stem}.pdf"
    fig.savefig(png_path, dpi=200)  # type: ignore
    fig.savefig(pdf_path)  # type: ignore
    plt.close(fig)  # type: ignore


def _read_rows(csv_path: Path) -> List[Row]:
    rows: List[Row] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            exp_name = str(r.get("exp_name", "")).strip()
            if not exp_name:
                continue
            task = str(r.get("task", "")).strip()
            mode = _infer_mode(exp_name, str(r.get("mode", "")).strip())
            run_dir_name = str(r.get("run_dir_name", "")).strip()
            if not run_dir_name:
                run_dir_name = str(r.get("run_dir", "")).strip()

            gi = _infer_group_and_x(exp_name)
            if gi is None:
                continue
            group_id, x = gi

            rows.append(
                Row(
                    exp_name=exp_name,
                    task=task,
                    mode=mode,
                    group_id=group_id,
                    x=x,
                    run_dir_name=run_dir_name,
                    test_loss_mean=_safe_float(r.get("test_loss_mean")),
                    test_loss_std=_safe_float(r.get("test_loss_std")),
                    test_acc_mean=_safe_float(r.get("test_acc_mean")),
                    test_acc_std=_safe_float(r.get("test_acc_std")),
                    atom_acc_mean=_safe_float(r.get("recovery_atom_acc_mean")),
                    atom_acc_std=_safe_float(r.get("recovery_atom_acc_std")),
                    mask_f1_mean=_safe_float(r.get("recovery_mask_f1_mean")),
                    mask_f1_std=_safe_float(r.get("recovery_mask_f1_std")),
                )
            )
    return rows


def _dedup_latest(rows: List[Row]) -> List[Row]:
    best: Dict[str, Row] = {}
    for r in rows:
        key = r.exp_name
        if key not in best:
            best[key] = r
            continue
        if _extract_dt(r.run_dir_name) >= _extract_dt(best[key].run_dir_name):
            best[key] = r
    return list(best.values())


def _collect_metric(rows: List[Row], group_id: str, metric: str) -> Tuple[List[float], Dict[str, Dict[float, Tuple[float, float]]]]:
    """
    Return sorted x_vals and by_mode[x] -> (mean,std)
    metric in: test_loss, test_acc, atom_acc, mask_f1
    """
    subset = [r for r in rows if r.group_id == group_id]
    x_vals = sorted({r.x for r in subset if not math.isnan(r.x)})

    by_mode: Dict[str, Dict[float, Tuple[float, float]]] = {}
    for r in subset:
        by_mode.setdefault(r.mode, {})
        if metric == "test_loss":
            by_mode[r.mode][r.x] = (r.test_loss_mean, r.test_loss_std)
        elif metric == "test_acc":
            by_mode[r.mode][r.x] = (r.test_acc_mean, r.test_acc_std)
        elif metric == "atom_acc":
            by_mode[r.mode][r.x] = (r.atom_acc_mean, r.atom_acc_std)
        elif metric == "mask_f1":
            by_mode[r.mode][r.x] = (r.mask_f1_mean, r.mask_f1_std)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return x_vals, by_mode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--dedup", type=str, default="latest", choices=["none", "latest"])
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_root = Path(args.out)
    plots_dir = out_root / "plots"
    tables_dir = out_root / "tables"
    out_root.mkdir(parents=True, exist_ok=True)

    # helpful sanity print (prevents the numpy/matplotlib ABI confusion)
    print(f"[PY] {os.path.abspath(os.sys.executable)}")
    if not _PLOT_OK:
        print(f"[WARN] matplotlib import failed -> plots disabled. err={_PLOT_IMPORT_ERR}")

    rows = _read_rows(csv_path)
    if args.dedup == "latest":
        rows = _dedup_latest(rows)

    print(f"[IN] rows={len(rows)}  csv={csv_path}")
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # baseline text files (take p_zero=0.6, qfbn) if exist
    def _write_baseline(task_prefix: str, path: Path) -> None:
        cand = [r for r in rows if r.exp_name == f"{task_prefix}_pzero_0.6_qfbn"]
        if not cand:
            path.write_text("baseline not found (expected exp_name: "
                            f"{task_prefix}_pzero_0.6_qfbn)\n", encoding="utf-8")
            return
        r = cand[0]
        if task_prefix == "ce":
            txt = (
                f"CE baseline @ p_zero=0.6 (QFBN):\n"
                f"test CE = {r.test_loss_mean:.6g} ± {r.test_loss_std:.6g}\n"
                f"test acc = {r.test_acc_mean:.6g} ± {r.test_acc_std:.6g}\n"
                f"atom acc = {r.atom_acc_mean:.6g} ± {r.atom_acc_std:.6g}\n"
                f"mask F1  = {r.mask_f1_mean:.6g} ± {r.mask_f1_std:.6g}\n"
            )
        else:
            txt = (
                f"MSE baseline @ p_zero=0.6 (QFBN):\n"
                f"test MSE = {r.test_loss_mean:.6g} ± {r.test_loss_std:.6g}\n"
                f"atom acc = {r.atom_acc_mean:.6g} ± {r.atom_acc_std:.6g}\n"
                f"mask F1  = {r.mask_f1_mean:.6g} ± {r.mask_f1_std:.6g}\n"
            )
        path.write_text(txt, encoding="utf-8")

    _write_baseline("ce", tables_dir / "baseline_ce.txt")
    _write_baseline("mse", tables_dir / "baseline_mse.txt")

    # Generate tables + plots for each known group
    for group_id, (x_label, _) in GROUP_SPECS.items():
        # tables
        if group_id.startswith("ce_"):
            # test acc table
            x_vals, by_mode = _collect_metric(rows, group_id, "test_acc")
            _write_table(
                tables_dir / f"tab_{group_id}_testacc.tex",
                caption=f"Test accuracy vs {x_label} (mean$\\pm$std over seeds).",
                label=f"tab:{group_id}_testacc",
                x_name=x_label,
                x_vals=x_vals,
                by_mode=by_mode,
                sig=4,
            )
            # test loss table
            x_vals, by_mode = _collect_metric(rows, group_id, "test_loss")
            _write_table(
                tables_dir / f"tab_{group_id}_testloss.tex",
                caption=f"Test CE vs {x_label} (mean$\\pm$std over seeds).",
                label=f"tab:{group_id}_testloss",
                x_name=x_label,
                x_vals=x_vals,
                by_mode=by_mode,
                sig=4,
            )
            # recovery atom acc
            x_vals, by_mode = _collect_metric(rows, group_id, "atom_acc")
            _write_table(
                tables_dir / f"tab_{group_id}_atomacc.tex",
                caption=f"Atom recovery accuracy vs {x_label} (mean$\\pm$std).",
                label=f"tab:{group_id}_atomacc",
                x_name=x_label,
                x_vals=x_vals,
                by_mode=by_mode,
                sig=4,
            )
            # recovery mask f1
            x_vals, by_mode = _collect_metric(rows, group_id, "mask_f1")
            _write_table(
                tables_dir / f"tab_{group_id}_maskf1.tex",
                caption=f"Support-mask F1 vs {x_label} (mean$\\pm$std).",
                label=f"tab:{group_id}_maskf1",
                x_name=x_label,
                x_vals=x_vals,
                by_mode=by_mode,
                sig=4,
            )
        else:
            # mse groups
            x_vals, by_mode = _collect_metric(rows, group_id, "test_loss")
            _write_table(
                tables_dir / f"tab_{group_id}_testloss.tex",
                caption=f"Test MSE vs {x_label} (mean$\\pm$std over seeds).",
                label=f"tab:{group_id}_testloss",
                x_name=x_label,
                x_vals=x_vals,
                by_mode=by_mode,
                sig=4,
            )
            x_vals, by_mode = _collect_metric(rows, group_id, "atom_acc")
            _write_table(
                tables_dir / f"tab_{group_id}_atomacc.tex",
                caption=f"Atom recovery accuracy vs {x_label} (mean$\\pm$std).",
                label=f"tab:{group_id}_atomacc",
                x_name=x_label,
                x_vals=x_vals,
                by_mode=by_mode,
                sig=4,
            )
            x_vals, by_mode = _collect_metric(rows, group_id, "mask_f1")
            _write_table(
                tables_dir / f"tab_{group_id}_maskf1.tex",
                caption=f"Support-mask F1 vs {x_label} (mean$\\pm$std).",
                label=f"tab:{group_id}_maskf1",
                x_name=x_label,
                x_vals=x_vals,
                by_mode=by_mode,
                sig=4,
            )

        # plots (skip if matplotlib broken)
        if _PLOT_OK:
            if group_id.startswith("ce_"):
                x_vals, by_mode = _collect_metric(rows, group_id, "test_acc")
                _plot_metric(plots_dir, group_id, "test acc", x_vals, by_mode, x_label, "test acc", f"{group_id}_testacc")
                x_vals, by_mode = _collect_metric(rows, group_id, "test_loss")
                _plot_metric(plots_dir, group_id, "test CE", x_vals, by_mode, x_label, "test CE", f"{group_id}_testloss")
                x_vals, by_mode = _collect_metric(rows, group_id, "atom_acc")
                _plot_metric(plots_dir, group_id, "atom acc", x_vals, by_mode, x_label, "atom acc", f"{group_id}_atomacc")
                x_vals, by_mode = _collect_metric(rows, group_id, "mask_f1")
                _plot_metric(plots_dir, group_id, "mask F1", x_vals, by_mode, x_label, "mask F1", f"{group_id}_maskf1")
            else:
                x_vals, by_mode = _collect_metric(rows, group_id, "test_loss")
                _plot_metric(plots_dir, group_id, "test MSE", x_vals, by_mode, x_label, "test MSE", f"{group_id}_testloss")
                x_vals, by_mode = _collect_metric(rows, group_id, "atom_acc")
                _plot_metric(plots_dir, group_id, "atom acc", x_vals, by_mode, x_label, "atom acc", f"{group_id}_atomacc")
                x_vals, by_mode = _collect_metric(rows, group_id, "mask_f1")
                _plot_metric(plots_dir, group_id, "mask F1", x_vals, by_mode, x_label, "mask F1", f"{group_id}_maskf1")

    # A compact LaTeX snippet referencing the MAIN tables/figs (pzero + scaling), noise in appendix.
    snippet = []
    snippet.append("% Auto-generated snippet (ICML-style).")
    snippet.append("% MAIN: p_zero sweeps")
    snippet.append("\\input{runs/_aggregate/icml/tables/tab_ce_pzero_testacc.tex}")
    snippet.append("\\input{runs/_aggregate/icml/tables/tab_ce_pzero_testloss.tex}")
    snippet.append("\\input{runs/_aggregate/icml/tables/tab_ce_pzero_atomacc.tex}")
    snippet.append("\\input{runs/_aggregate/icml/tables/tab_ce_pzero_maskf1.tex}")
    snippet.append("\\input{runs/_aggregate/icml/tables/tab_mse_pzero_testloss.tex}")
    snippet.append("\\input{runs/_aggregate/icml/tables/tab_mse_pzero_atomacc.tex}")
    snippet.append("\\input{runs/_aggregate/icml/tables/tab_mse_pzero_maskf1.tex}")
    snippet.append("")
    snippet.append("% FIGURES (use pdf for LaTeX)")
    snippet.append("\\begin{figure}[t]\\centering")
    snippet.append("\\includegraphics[width=\\linewidth]{runs/_aggregate/icml/plots/ce_pzero_maskf1.pdf}")
    snippet.append("\\caption{Support-mask F1 vs teacher sparsity $p_{zero}$ (multi-method).}")
    snippet.append("\\label{fig:ce_pzero_maskf1}")
    snippet.append("\\end{figure}")
    snippet.append("")
    snippet.append("\\begin{figure}[t]\\centering")
    snippet.append("\\includegraphics[width=\\linewidth]{runs/_aggregate/icml/plots/mse_pzero_testloss.pdf}")
    snippet.append("\\caption{Test MSE vs teacher sparsity $p_{zero}$ (multi-method).}")
    snippet.append("\\label{fig:mse_pzero_mse}")
    snippet.append("\\end{figure}")
    snippet.append("")
    snippet.append("% APPENDIX: noise robustness")
    snippet.append("\\input{runs/_aggregate/icml/tables/tab_ce_logitnoise_testacc.tex}")
    snippet.append("\\input{runs/_aggregate/icml/tables/tab_mse_noise_testloss.tex}")
    (out_root / "icml_snippet.tex").write_text("\n".join(snippet) + "\n", encoding="utf-8")

    print("[OK] ICML artifacts generated.")
    print("[OUT] plots: ", plots_dir)
    print("[OUT] tables:", tables_dir)
    print("[OUT] snippet:", out_root / "icml_snippet.tex")


if __name__ == "__main__":
    main()
