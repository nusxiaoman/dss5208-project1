Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

Write-Host "=== Starting sweep at $(Get-Date) ===" -ForegroundColor Yellow

New-Item -ItemType Directory -Force -Path histories,plots,results | Out-Null

$DATA = ".\nytaxi2022_cleaned.npz"   # <-- change if your NPZ name differs
if (-not (Test-Path $DATA)) { throw "Data file not found: $DATA" }

Write-Host "mpiexec:" (Get-Command mpiexec).Source -ForegroundColor DarkCyan
Write-Host "Python:"  (py -V) -ForegroundColor DarkCyan

$acts = @("relu","tanh","sigmoid")
$Mlist = @(32,64,128,256,512)
$seed  = 123

foreach ($act in $acts) {
  foreach ($M in $Mlist) {
    $n = 128
    if ($act -eq "relu")    { if ($M -ge 256) { $n = 256 } else { $n = 128 } }
    if ($act -eq "tanh")    { if ($M -ge 128) { $n = 128 } else { $n = 64  } }
    if ($act -eq "sigmoid") { if ($M -ge 256) { $n = 128 } else { $n = 64  } }

    $tag = "${act}_bs${M}_n${n}_P4"
    Write-Host ">>> RUN $tag" -ForegroundColor Cyan

    mpiexec -n 4 py .\train_mpi.py --data $DATA `
      --activation $act --batch $M --hidden $n --lr 1e-3 `
      --epochs 1 --eval_every 1000 --eval_sample 2000000 --seed $seed

    if (-not (Test-Path .\history.csv)) { throw "history.csv not found for $tag" }

    Copy-Item .\history.csv ".\histories\history_$tag.csv" -Force

    py .\plot_history.py `
      --in_csv ".\histories\history_$tag.csv" `
      --out_png ".\plots\trainhist_$tag.png" `
      --title "R(theta) vs iter — $tag"

    if (-not (Test-Path ".\plots\trainhist_$tag.png")) { throw "plot not created for $tag" }
  }
}

Write-Host "=== Sweep complete at $(Get-Date) ===" -ForegroundColor Green
