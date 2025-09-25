# sweep.ps1 — σ×M sweep at P=4, auto-select data mode, write run_summary.csv (Windows-lock safe)

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

# --- Avoid BLAS oversubscription (one thread per rank) ---
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'

# --- Sweep config ---
$acts = @("relu","tanh","sigmoid")
$Mlist = @(32,64,128,256,512)
$seed  = 123
$P     = 4

# --- Ensure summary file has a header ---
$summary = "results\run_summary.csv"
if (-not (Test-Path $summary)) {
  "activation,batch,hidden,lr,procs,train_time,rmse_train,rmse_test" | Set-Content -Encoding UTF8 $summary
}

foreach ($act in $acts) {
  foreach ($M in $Mlist) {
    # choose hidden units n by simple rule-of-thumb
    $n = 128
    if ($act -eq "relu")    { if ($M -ge 256) { $n = 256 } else { $n = 128 } }
    if ($act -eq "tanh")    { if ($M -ge 128) { $n = 128 } else { $n = 64  } }
    if ($act -eq "sigmoid") { if ($M -ge 256) { $n = 128 } else { $n = 64  } }

    $tag = "{0}_bs{1}_n{2}_P{3}" -f $act,$M,$n,$P
    Write-Host ">>> RUN $tag" -ForegroundColor Cyan

    # clean stale artifacts (ignore errors if locked)
    Remove-Item .\history.csv -ErrorAction SilentlyContinue
    Remove-Item .\model_final.npz -ErrorAction SilentlyContinue

    # --- Train ---
    & mpiexec -n $P python .\train_mpi.py @dataArg `
      --activation $act --batch $M --hidden $n --lr 1e-3 `
      --epochs 1 --eval_every 1000 --eval_sample 2000000 --eval_block 100000 --seed $seed

    if (-not (Test-Path .\history.csv))     { throw "history.csv not found for $tag" }
    if (-not (Test-Path .\model_final.npz)) { throw "model_final.npz not found for $tag" }

    # copy history
    Copy-Item .\history.csv ".\histories\history_$tag.csv" -Force

    # plot
    & python .\plot_history.py `
      --in_csv ".\histories\history_$tag.csv" `
      --out_png ".\plots\trainhist_$tag.png" `
      --title "R(theta) vs iter — $tag"

    # --- Append summary row from model meta (Windows-lock safe: copy + retry) ---
    $src = ".\model_final.npz"
    $dst = ".\tmp\model_$tag.npz"

    # 1) Copy with retries (handle OneDrive/AV locks)
    $copied = $false
    for ($i=1; $i -le 10; $i++) {
      try {
        Copy-Item $src $dst -Force
        $copied = $true
        break
      } catch {
        Start-Sleep -Milliseconds 500
      }
    }
    if (-not $copied) {
      Write-Warning "Could not copy $src for $tag; skipping summary append."
      continue
    }

    # 2) Python snippet that reads a given file path
    $py = @'
import numpy as np, sys
m = np.load(sys.argv[1], allow_pickle=True)
meta = dict(m["meta"].item())
act = meta.get("act"); M = meta.get("batch"); n = meta.get("hidden"); lr = meta.get("lr")
P = meta.get("P"); t = meta.get("time_sec"); tr = meta.get("train_rmse"); te = meta.get("test_rmse")
print(f"{act},{M},{n},{lr},{P},{t},{tr},{te}")
'@

    $tf = New-TemporaryFile
    Set-Content -LiteralPath $tf -Value $py -Encoding UTF8

    # 3) Read with retries
    $row = $null
    for ($i=1; $i -le 10; $i++) {
      try {
        $row = & python $tf $dst
        break
      } catch {
        Start-Sleep -Milliseconds 500
      }
    }
    Remove-Item $tf -Force
    Remove-Item $dst -Force -ErrorAction SilentlyContinue

    if ($row) {
      Add-Content -Encoding UTF8 -Path $summary -Value $row
    } else {
      Write-Warning "Could not read model meta for $tag; skipping summary append."
    }
  }
}

# Optional: build aggregate plots/tables if the helper exists
if (Test-Path .\summarize_results.py) {
  & python .\summarize_results.py
}

Write-Host "Sweep complete. Plots in .\plots, histories in .\histories, summary in $summary" -ForegroundColor Green
