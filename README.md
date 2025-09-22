# DSS5208 Project 1 — Distributed Training on NYC Taxi (MPI)

Scalable training of a 1-hidden-layer neural network on NYC taxi data using **MPI (mpi4py)**.  
Includes: σ×M sweep (3 activations × 5 batch sizes), training histories, RMSE (train/test), training time, and strong-scaling (P = 1,2,4,8).

> **Data note:** raw/large data (e.g., `nytaxi2022.csv`, `*.npz`) are **not** tracked in Git. See `.gitignore`.

---

## Repo layout

```text
├─ train_mpi.py              # MPI training (eval subsample + chunked RMSE + no-copy loads)
├─ data_prep.py              # One-time data prep → nytaxi2022_cleaned.npz (not committed)
├─ plot_history.py           # Plot R(theta) vs iteration from history.csv
├─ sweep.ps1                 # σ×M sweep at P=4
├─ scaling.ps1               # Training time vs processes (P = 1,2,4,8)
├─ summarize_results.py      # Tables + plots for sweep (RMSE/time vs batch)
├─ scaling_summary.py        # Scaling table + time/speedup/efficiency plots
├─ results/                  # CSV summaries (kept in repo)
├─ plots/                    # Figures (kept in repo)
├─ REPORT.md                 # Project write-up
└─ .gitignore                # Excludes raw/large data + caches
```

Requirements

Windows 10/11

Python 3.11+ (tested on 3.13)

MS-MPI (mpiexec on PATH)

Python packages: numpy, pandas, matplotlib, mpi4py

```powershell
py -m pip install --upgrade pip
py -m pip install numpy pandas matplotlib mpi4py
# verify
Get-Command mpiexec
mpiexec -n 2 py -c "from mpi4py import MPI; print('rank', MPI.COMM_WORLD.Get_rank(), 'of', MPI.COMM_WORLD.Get_size())"
```


Data

Prepare once (produces nytaxi2022_cleaned.npz, not committed):

```powershell
py .\data_prep.py --input_path .\nytaxi2022.csv --output_path .\nytaxi2022_cleaned.npz
```

Quick start
1) Smoke test one run
```powershell
mpiexec -n 4 py .\train_mpi.py --data .\nytaxi2022_cleaned.npz `
  --activation relu --batch 256 --hidden 256 --lr 1e-3 `
  --epochs 1 --eval_every 1000 --eval_sample 2000000 --eval_block 100000 --seed 123

py .\plot_history.py --in_csv .\history.csv `
  --out_png .\plots\training_history_smoke.png `
  --title "Smoke test"
```
2) σ×M sweep (P=4)
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\sweep.ps1
py .\summarize_results.py
```

Outputs:

results/best_per_activation_batch.csv, results/top5_overall.csv

plots/rmse_vs_batch.png, plots/time_vs_batch.png

histories/history_<tag>.csv, plots/trainhist_<tag>.png

3) Strong-scaling runs (P = 1,2,4,8)

Edit scaling.ps1 to pick your (activation, batch, hidden), then:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scaling.ps1
py .\scaling_summary.py
```

Outputs:

results/scaling_table.csv

plots/scaling_time.png, plots/scaling_speedup.png, plots/scaling_efficiency.png

Tip (optional, helps P≥8 on one node):
```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
```
Results (example)

Insert your own figures, e.g.:

![Test RMSE vs batch](plots/rmse_vs_batch.png)
![Training time vs batch](plots/time_vs_batch.png)
![Scaling: time](plots/scaling_time.png)
![Scaling: speedup](plots/scaling_speedup.png)
![Scaling: efficiency](plots/scaling_efficiency.png)

Implementation details / improvements

Broadcast only minibatch indices per step (not full epoch permutations).

Chunked RMSE (--eval_block) to bound memory during eval.

Subsampled periodic eval (--eval_sample) for fast R(θ) tracking.

No-copy loads for float32 arrays to avoid multi-GB duplicates.

Explicit biases (no giant temporary concatenations).

Early-stop patience on R(θ) trend.
