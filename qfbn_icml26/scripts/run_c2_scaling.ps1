$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $ProjectRoot

$env:PYTHONPATH = "$ProjectRoot;$ProjectRoot\src"

Write-Host "[INFO] ProjectRoot = $ProjectRoot"
Write-Host "[INFO] PYTHONPATH  = $env:PYTHONPATH"

conda run -n qfbn python -c "import importlib.util; print('qfbn spec:', importlib.util.find_spec('qfbn'))"

$baseMSE = "configs/synth_recovery_mse.yaml"
$baseCE  = "configs/synth_recovery_ce.yaml"

$widths = @(16, 32, 64, 128)
$depths = @(1, 2, 3)

$pz_mse = 0.6
$pz_ce  = 0.6

foreach ($w in $widths) {
  $hd = "[$w]"

  $exp = "mse_scale_width_${w}_qfbn"
  Write-Host "`n[RUN] $exp"
  conda run -n qfbn python -m qfbn.experiments.synth_recovery `
    --config $baseMSE `
    --set "mode=qfbn" `
    --set "teacher.p_zero=$pz_mse" `
    --set "teacher.hidden_dims=$hd" `
    --set "student.hidden_dims=$hd" `
    --set "run.exp_name=$exp"

  $exp = "ce_scale_width_${w}_qfbn"
  Write-Host "`n[RUN] $exp"
  conda run -n qfbn python -m qfbn.experiments.synth_recovery `
    --config $baseCE `
    --set "mode=qfbn" `
    --set "teacher.p_zero=$pz_ce" `
    --set "teacher.hidden_dims=$hd" `
    --set "student.hidden_dims=$hd" `
    --set "run.exp_name=$exp"
}

foreach ($d in $depths) {
  if ($d -eq 1) { $hd = "[32]" }
  if ($d -eq 2) { $hd = "[32,32]" }
  if ($d -eq 3) { $hd = "[32,32,32]" }

  $exp = "mse_scale_depth_${d}_qfbn"
  Write-Host "`n[RUN] $exp"
  conda run -n qfbn python -m qfbn.experiments.synth_recovery `
    --config $baseMSE `
    --set "mode=qfbn" `
    --set "teacher.p_zero=$pz_mse" `
    --set "teacher.hidden_dims=$hd" `
    --set "student.hidden_dims=$hd" `
    --set "run.exp_name=$exp"

  $exp = "ce_scale_depth_${d}_qfbn"
  Write-Host "`n[RUN] $exp"
  conda run -n qfbn python -m qfbn.experiments.synth_recovery `
    --config $baseCE `
    --set "mode=qfbn" `
    --set "teacher.p_zero=$pz_ce" `
    --set "teacher.hidden_dims=$hd" `
    --set "student.hidden_dims=$hd" `
    --set "run.exp_name=$exp"
}

Pop-Location
Write-Host "`n[ALL DONE] C2 scaling finished."
