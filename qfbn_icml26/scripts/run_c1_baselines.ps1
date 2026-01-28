$ErrorActionPreference = "Stop"

# Resolve project root robustly (scripts/..)
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $ProjectRoot

# Force Python to see repo root and src/ (works even if editable install is misconfigured)
$env:PYTHONPATH = "$ProjectRoot;$ProjectRoot\src"

Write-Host "[INFO] ProjectRoot = $ProjectRoot"
Write-Host "[INFO] PYTHONPATH  = $env:PYTHONPATH"

# Sanity check: can python find qfbn?
conda run -n qfbn python -c "import importlib.util; s=importlib.util.find_spec('qfbn'); print('qfbn spec:', s); import sys; print('sys.path[0:5]=', sys.path[0:5])"

$baseMSE = "configs/synth_recovery_mse.yaml"
$baseCE  = "configs/synth_recovery_ce.yaml"

$pzeros = @(0.2, 0.4, 0.6, 0.8)
$modes  = @("qfbn", "mlp", "oracle_mask", "oracle_atom")

foreach ($pz in $pzeros) {
  foreach ($mode in $modes) {
    $exp = "mse_pzero_${pz}_${mode}"
    Write-Host "`n[RUN] $exp  using $baseMSE"
    conda run -n qfbn python -m qfbn.experiments.synth_recovery `
      --config $baseMSE `
      --set "mode=$mode" `
      --set "teacher.p_zero=$pz" `
      --set "run.exp_name=$exp"
  }

  foreach ($mode in $modes) {
    $exp = "ce_pzero_${pz}_${mode}"
    Write-Host "`n[RUN] $exp  using $baseCE"
    conda run -n qfbn python -m qfbn.experiments.synth_recovery `
      --config $baseCE `
      --set "mode=$mode" `
      --set "teacher.p_zero=$pz" `
      --set "run.exp_name=$exp"
  }
}

Pop-Location
Write-Host "`n[ALL DONE] C1 baselines finished."
