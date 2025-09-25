# sweep_debug.ps1 — sanity-check one training run before the big sweep
# - Detects memmap vs npz
# - Prints env (mpiexec, python path)
# - Limits BLAS threads per MPI rank
# - Runs one quick job
# - Verifies history.csv/model_final.npz
# - Plots training history
# - Extracts time/RMSE from model meta
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "=== Starting sweep_debug at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

# 0) Where are mpiexec & python?
$mpiexec = (Get-Command mpiexec -ErrorAction Stop).Source
$pyexe   = & python -c "import sys; print(sys.executable)"
Write-Host "mpiexec: $mpiexec"
Write-Host "python : $pyexe"

# 1) Ensure output folders exist
New-Item -ItemType Directory -Force -Path histories, plots, results,tmp | Out-Null

# 2) Choose data mode automatically
$useNpy = Test-Path .\memmap_data\X_train.npy
$useNpz = Test-Path .\nytaxi2022_cleaned.npz
if (-not $useNpy -and -not $useNpz) {
  throw "No data found. Create memmap_data\X_train.npy... (prep_memmap_from_npz.py) OR nytaxi2022_cleaned.npz (data_prep.py)."
}
if ($useNpy) { $mode = "npy"; $dataArg = @("--npy_root", ".\memmap_data") }
else         { $mode = "npz"; $dataArg = @("--data", ".\nytaxi2022_cleaned.npz") }
Write-Host "Data mode: $mode"

# 3) Avoid thread oversubscription on one machine
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'

# 4) Pick a small-but-real config
$act = "relu"; $M = 256; $n = 256; $lr = "1e-3"; $epochs = 1; $P = 4
$tag = "{0}_bs{1}_n{2}_P{3}" -f $act,$M,$n,$P
Write-Host ">>> RUN $tag"

# 5) Clean previous artifacts
Remove-Item .\history.csv -ErrorAction SilentlyContinue
Remove-Item .\model_final.npz -ErrorAction SilentlyContinue

# 6) Train (one run)
& mpiexec -n $P python .\train_mpi.py @dataArg `
  --activation $act --batch $M --hidden $n --lr $lr `
  --epochs $epochs --eval_every 1000 --eval_sample 2000000 --eval_block 100000 --seed 123

# 7) Verify outputs
if (-not (Test-Path .\history.csv)) { throw "history.csv not found for $tag" }
if (-not (Test-Path .\model_final.npz)) { throw "model_final.npz not found for $tag" }

# 8) Copy history & plot
Copy-Item .\history.csv ".\histories\history_$tag.csv" -Force
& python .\plot_history.py --in_csv .\history.csv `
  --out_png ".\plots\trainhist_$tag.png" `
  --title "Training history — $tag"


# 9) Extract time/RMSE from model meta -> results\debug_summary.csv
#    (Windows-lock safe: copy model_final.npz to a temp file and retry reads)
$summary = "results\debug_summary.csv"
if (-not (Test-Path $summary)) {
  "activation,batch,hidden,lr,procs,train_time,rmse_train,rmse_test" | Set-Content -Encoding UTF8 $summary
}

$src = ".\model_final.npz"
$dst = ".\tmp\model_$tag.npz"

# Copy with retries (handles short-lived OneDrive/AV locks)
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
  goto AfterAppend
}

# Python snippet reads the passed file path
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

# Read with retries
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
  Write-Host "Wrote: $summary" -ForegroundColor Green
} else {
  Write-Warning "Could not read model meta for $tag; skipping summary append."
}

:AfterAppend

Write-Host "OK — Debug run completed."
Write-Host "  history  : histories\history_$tag.csv"
Write-Host "  plot     : plots\trainhist_$tag.png"
Write-Host "  summary  : results\debug_summary.csv"
