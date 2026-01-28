# scripts/run_sweeps.ps1
$ErrorActionPreference = "Stop"

# Ensure we are at project root
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "[INFO] cwd = $(Get-Location)"

# Quick sanity: can we import qfbn?
python -c "import qfbn; print('[OK] import qfbn')"

# -----------------------------
# Common helper
# -----------------------------
function Run-Exp {
    param(
        [string]$config,
        [string]$exp_name,
        [string[]]$sets
    )
    Write-Host ""
    Write-Host "[RUN] $exp_name using $config"
    $args = @("-m","qfbn.experiments.synth_recovery","--config",$config,"--set","run.exp_name=$exp_name")
    foreach ($s in $sets) { $args += @("--set",$s) }
    python @args
}

# -----------------------------
# C1: Baselines / structural oracles on p_zero sweep
# -----------------------------
$pzeros = @("0.2","0.4","0.6","0.8")
$methods = @("qfbn","mlp","oracle_mask","oracle_atom")

foreach ($pz in $pzeros) {
    foreach ($m in $methods) {
        Run-Exp "configs/synth_recovery_mse.yaml" "mse_pzero_${pz}_${m}" @(
            "mode=$m",
            "teacher.p_zero=$pz"
        )
        Run-Exp "configs/synth_recovery_ce.yaml" "ce_pzero_${pz}_${m}" @(
            "mode=$m",
            "teacher.p_zero=$pz"
        )
    }
}

# -----------------------------
# Robustness sweeps (ONLY qfbn; add more methods if you want)
# -----------------------------

# Classification: true logit_noise sweep (note: this is separate from default 0.05)
$logit_noises = @("0.0","0.1","0.2","0.4")
foreach ($ln in $logit_noises) {
    Run-Exp "configs/synth_recovery_ce.yaml" "ce_logitnoise_${ln}_qfbn" @(
        "mode=qfbn",
        "data.logit_noise=$ln"
    )
}

# Regression: true noise_std sweep
$noises = @("0.0","0.05","0.1","0.2")
foreach ($ns in $noises) {
    Run-Exp "configs/synth_recovery_mse.yaml" "mse_noise_${ns}_qfbn" @(
        "mode=qfbn",
        "data.noise_std=$ns"
    )
}

Write-Host ""
Write-Host "[ALL DONE] sweeps finished."
