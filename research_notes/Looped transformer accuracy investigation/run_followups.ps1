# Sequential GPU pipeline after the ext50k extension: waits for it, then
# 1) dyn50k: retrain three seed-42 trajectories with per-epoch weight snapshots,
# 2) fixed-checkpoint follow-up evaluation (harder sets, inference-depth sweep),
# 3) residual-stream / PCA / probe analysis of every checkpoint and snapshot.
$py = 'C:\Users\omnap\miniconda3\envs\llmdyn\python.exe'
$root = 'D:\Projects\Loopedtrans\research_notes\Looped transformer accuracy investigation'
Set-Location $root
while (-not (Test-Path "$root\extension_complete.json")) { Start-Sleep -Seconds 30 }
function Run($name, [string[]]$argv) {
    "=== $name $(Get-Date -Format o)"
    & $py -u @argv 2>&1 | Where-Object { "$_" -notmatch 'UserWarning|warnings.warn|self.encoder = TransformerEncoder' }
    "=== $name exit $LASTEXITCODE $(Get-Date -Format o)"
}
Run 'dyn50k 30-epoch trajectories' @("$root\extension_experiments.py", '--prefix', 'dyn50k', '--snapshots', '--epochs', '30', '--runs', 'original_baseline@42,original_looped@42')
Run 'dyn50k 20-epoch Pre-LN trajectory' @("$root\extension_experiments.py", '--prefix', 'dyn50k_preln', '--snapshots', '--epochs', '20', '--runs', 'original_baseline_preln@42')
Run 'follow-up evaluation' @("$root\fullscale_followup_eval.py")
Run 'residual analysis' @("$root\residual_analysis.py")
'PIPELINE COMPLETE'
