# sweep.ps1 — 3 activations × 5 batch sizes at P=4 (chunked eval + subsample)
New-Item -ItemType Directory -Force -Path histories,plots,results | Out-Null

$DATA = ".\nytaxi2022_cleaned.npz"
$acts = @("relu","tanh","sigmoid")
$Mlist = @(32,64,128,256,512)
$seed  = 123

foreach ($act in $acts) {
  foreach ($M in $Mlist) {
    # choose hidden units n by simple rule-of-thumb
    $n = 128
    if ($act -eq "relu")    { if ($M -ge 256) { $n = 256 } else { $n = 128 } }
    if ($act -eq "tanh")    { if ($M -ge 128) { $n = 128 } else { $n = 64  } }
    if ($act -eq "sigmoid") { if ($M -ge 256) { $n = 128 } else { $n = 64  } }

    $tag = "${act}_bs${M}_n${n}_P4"
    Write-Host ">>> RUN $tag" -ForegroundColor Cyan

    mpiexec -n 4 py .\train_mpi.py --data $DATA `
      --activation $act --batch $M --hidden $n --lr 1e-3 `
      --epochs 1 --eval_every 1000 --eval_sample 2000000 --eval_block 100000 --seed $seed

    if (-not (Test-Path .\history.csv)) {
      throw "history.csv not found for $tag"
    }
    Copy-Item .\history.csv ".\histories\history_$tag.csv" -Force

    py .\plot_history.py `
      --in_csv ".\histories\history_$tag.csv" `
      --out_png ".\plots\trainhist_$tag.png" `
      --title "R(theta) vs iter — $tag"
  }
}

Write-Host "Sweep complete. Plots in .\plots, histories in .\histories, summary in run_summary.csv" -ForegroundColor Green
