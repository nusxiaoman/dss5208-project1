# scaling.ps1 — training time vs processes for one chosen config
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path histories,plots | Out-Null

$DATA = ".\nytaxi2022_cleaned.npz"

# Pick ONE triple (change these if you want):
#   relu, M=64,  n=128   -> fastest, almost same RMSE
#   relu, M=256, n=256   -> balanced  (default)
#   relu, M=512, n=256   -> best RMSE, slowest
$act = "relu"; $M = 256; $n = 256

$seed = 123
$plist = @(1,2,4,8)

foreach ($p in $plist) {
  $tag = "${act}_bs${M}_n${n}_P${p}"
  Write-Host ">>> SCALE $tag" -ForegroundColor Yellow

  & mpiexec -n $p py .\train_mpi.py --data $DATA --activation $act --batch $M --hidden $n --lr 1e-3 --epochs 1 --eval_every 1000 --eval_sample 2000000 --eval_block 100000 --seed $seed

  if (Test-Path ".\history.csv") {
    Copy-Item ".\history.csv" ".\histories\history_$tag.csv" -Force
    py .\plot_history.py --in_csv ".\histories\history_$tag.csv" --out_png ".\plots\trainhist_$tag.png" --title "R(theta) vs iter - $tag"
  }
  else {
    Write-Warning "history.csv not found for $tag"
  }
}
