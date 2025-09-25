# scaling.ps1 — training time vs processes for one chosen config (Windows-lock safe)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path histories,plots,results,tmp | Out-Null

# --- Auto-select data mode (memmap preferred) ---
$useNpy = Test-Path .\memmap_data\X_train.npy
$useNpz = Test-Path .\nytaxi2022_cleaned.npz
if (-not $useNpy -and -not $useNpz) {
  throw "No data found. Prepare memmap_data (prep_memmap_from_npz.py) or nytaxi2022_cleaned.npz (data_prep.py)."
}
if ($useNpy) {
  Write-Host "Data mode: memmap (--npy_root .\memmap_data)" -ForegroundColor Yellow
  $dataArg = @("--npy_root", ".\memmap_data")
} else {
  Write-Host "Data mode: npz (--data .\nytaxi2022_cleaned.npz)" -ForegroundColor Yellow
  $dataArg = @("--data", ".\nytaxi2022_cleaned.npz")
}

# --- (Optional but recommended) Avoid BLAS oversubscription ---
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'

# Pick ONE triple (change these if you want):
#   relu, M=64,  n=128   -> fastest, almost same RMSE
#   relu, M=256, n=256   -> balanced  (default)
#   relu, M=512, n=256   -> best RMSE, slowest
$act = "relu"; $M = 256; $n = 256

$seed  = 123
$plist = @(1,2,4,8)

# Prepare scaling table with header
$out = "results\scaling_table.csv"
"activation,batch,hidden,lr,procs,train_time,rmse_train,rmse_test,speedup,efficiency" | Set-Content -Encoding UTF8 $out
$t1 = $null  # baseline P=1 time for speedup

foreach ($p in $plist) {
  $tag = "${act}_bs${M}_n${n}_P${p}"
  Write-Host ">>> SCALE $tag" -ForegroundColor Yellow

  # Clean stale artifacts
  Remove-Item ".\history.csv" -ErrorAction SilentlyContinue
  Remove-Item ".\model_final.npz" -ErrorAction SilentlyContinue

  # Train (use the venv's Python)
  & mpiexec -n $p python .\train_mpi.py @dataArg `
    --activation $act --batch $M --hidden $n --lr 1e-3 `
    --epochs 1 --eval_every 1000 --eval_sample 2000000 --eval_block 100000 --seed $seed

  if (Test-Path ".\history.csv") {
    Copy-Item ".\history.csv" ".\histories\history_$tag.csv" -Force
    if (Test-Path .\plot_history.py) {
      python .\plot_history.py --in_csv ".\histories\history_$tag.csv" `
        --out_png ".\plots\trainhist_$tag.png" `
        --title "R(theta) vs iter - $tag"
    }
  } else {
    Write-Warning "history.csv not found for $tag"
  }

  if (-not (Test-Path ".\model_final.npz")) {
    Write-Warning "model_final.npz not found for $tag; skipping summary row."
    continue
  }

  # ---- Safe read of model_final.npz (copy → retry) ----
  $src = ".\model_final.npz"
  $dst = ".\tmp\model_$tag.npz"

  $copied = $false
  for ($i=1; $i -le 10; $i++) {
    try { Copy-Item $src $dst -Force; $copied = $true; break } catch { Start-Sleep -Milliseconds 500 }
  }
  if (-not $copied) {
    Write-Warning "Could not copy $src for $tag; skipping row."
    continue
  }

  $py = @'
import numpy as np, sys
m = np.load(sys.argv[1], allow_pickle=True)
meta = dict(m["meta"].item())
act = meta.get("act"); M = meta.get("batch"); n = meta.get("hidden"); lr = meta.get("lr")
P = int(meta.get("P")); t = float(meta.get("time_sec"))
tr = float(meta.get("train_rmse")); te = float(meta.get("test_rmse"))
print(f"{act},{M},{n},{lr},{P},{t},{tr},{te}")
'@
  $tf = New-TemporaryFile
  Set-Content -LiteralPath $tf -Value $py -Encoding UTF8

  $row = $null
  for ($i=1; $i -le 10; $i++) {
    try { $row = & python $tf $dst; break } catch { Start-Sleep -Milliseconds 500 }
  }
  Remove-Item $tf -Force
  Remove-Item $dst -Force -ErrorAction SilentlyContinue

  if (-not $row) {
    Write-Warning "Could not read model meta for $tag; row skipped."
    continue
  }

  $parts = $row -split ','
  $procs = [int]$parts[4]
  $time  = [double]$parts[5]

  if ($procs -eq 1 -and -not $t1) { $t1 = $time }

  $speed = if ($t1 -and $time -gt 0) { [double]($t1 / $time) } else { "" }
  $eff   = if ($speed -ne "" -and $procs -gt 0) { [double]($speed / $procs) } else { "" }

  # Append row with speedup/efficiency
  Add-Content -Encoding UTF8 -Path $out -Value ($row + "," + $speed + "," + $eff)
}

Write-Host "Scaling complete. Table: $out; plots in .\plots" -ForegroundColor Green
