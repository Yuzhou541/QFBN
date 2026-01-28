# QFBN (Quantized Functional-Basis Networks) — Refactored

This project implements the Functional-Basis Networks idea:

- Edge-wise weight function: f_ij(w_ij), selected from a dictionary F using softmax gates with temperature annealing.
- Edge-wise (or per-node) activation: g_ij(z), optional selection from dictionary G.
- Forward (strictly follows your mechanism):
  y_i = sum_j g_ij( f_ij(w_ij) * x_j ) + b_i

synthetic recovery experiments:
- Regression recovery (MSE)
- Classification recovery (Cross-Entropy)

Metrics:
- Edge atom accuracy
- Nonzero mask F1
- Per-atom precision/recall/F1
- Curves: entropy (wave -> particle), expected degree (degree curriculum), expected nonzero

## Install (PowerShell)
```powershell
conda activate qfbn
pip install -r requirements.txt
pip install -e .
