# DSS5208 Project 1 — Distributed Training on NYC Taxi (MPI)

Scalable training of a 1-hidden-layer neural network on NYC taxi data using **MPI (mpi4py)**.  
Includes: σ×M sweep (3 activations × 5 batch sizes), training histories, RMSE (train/test), training time, and strong-scaling (P = 1,2,4,8).

> **Data note:** raw/large data (e.g., `nytaxi2022.csv`, `*.npz`) are **not** tracked in Git. See `.gitignore`.

---

## Repo layout

.
├─ train_mpi.py # MPI training (eval subsample + chunked RMSE + no-copy loads)
├─ data_prep.py # One-time data prep → nytaxi2022_cleaned.npz (not committed)
├─ plot_history.py # Plot R(theta) vs iteration from history.csv
├─ sweep.ps1 # σ×M sweep at P=4
├─ scaling.ps1 # Training time vs processes (P = 1,2,4,8)
├─ summarize_results.py # Tables + plots for sweep (RMSE/time vs batch)
├─ scaling_summary.py # Scaling table + time/speedup/efficiency plots
├─ results/ # CSV summaries (kept in repo)
├─ plots/ # Figures (kept in repo)
├─ REPORT.md # Project write-up
└─ .gitignore # Excludes raw/large data + caches

markdown
Copy code

---

## Requirements

- **Windows 10/11**
- **Python 3.11+** (tested with 3.13)
- **MS-MPI** (mpiexec on PATH: `C:\Program Files\Microsoft MPI\Bin`)
- Python packages: `numpy`, `pandas`, `matplotlib`, `mpi4py`

Install (PowerShell):

```powershell
py -m pip install --upgrade pip
py -m pip install numpy pandas matplotlib mpi4py
Verify MPI:

powershell
Copy code
Get-Command mpiexec
mpiexec -n 2 py -c "from mpi4py import MPI; print('rank', MPI.COMM_WORLD.Get_rank(), 'of', MPI.COMM_WORLD.Get_size())"
Data
Prepare once (produces nytaxi2022_cleaned.npz, not committed):

powershell
Copy code
py .\data_prep.py --input_path .\nytaxi2022.csv --output_path .\nytaxi2022_cleaned.npz
Keep the .npz locally; do not push to GitHub.

Quick start
1) Smoke test one training run
powershell
Copy code
mpiexec -n 4 py .\train_mpi.py --data .\nytaxi2022_cleaned.npz `
  --activation relu --batch 256 --hidden 256 --lr 1e-3 `
  --epochs 1 --eval_every 1000 --eval_sample 2000000 --eval_block 100000 --seed 123

py .\plot_history.py --in_csv .\history.csv `
  --out_png .\plots\training_history_smoke.png `
  --title "Smoke test"
Artifacts: history.csv, model_final.npz, plots/training_history_smoke.png.

2) Run the σ×M sweep (P=4)
powershell
Copy code
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\sweep.ps1
Outputs:

run_summary.csv (rows: activation,batch,hidden,lr,procs,train_time,rmse_train,rmse_test)

histories/history_<tag>.csv

plots/trainhist_<tag>.png

3) Summarize sweep (tables + plots)
powershell
Copy code
py .\summarize_results.py
Outputs:

results/best_per_activation_batch.csv

results/top5_overall.csv

plots/rmse_vs_batch.png

plots/time_vs_batch.png

4) Strong scaling (P = 1,2,4,8)
Edit scaling.ps1 to set the chosen triple (e.g., relu, M=256, n=256), then:

powershell
Copy code
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scaling.ps1
py .\scaling_summary.py
Outputs:

results/scaling_table.csv

plots/scaling_time.png, plots/scaling_speedup.png, plots/scaling_efficiency.png

Tip (single-node scaling): set BLAS threads to 1 per MPI rank before running:

powershell
Copy code
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
Parameters (defaults used)
Activations 
𝜎
σ: relu, tanh, sigmoid

Batch sizes 
𝑀
M: 32, 64, 128, 256, 512

Hidden units 
𝑛
n: rule-of-thumb by (σ, M) in sweep.ps1

Learning rate: 1e-3

Epochs: 1

Eval cadence: --eval_every 1000

Periodic eval subsample: --eval_sample 2_000_000

Chunked eval block: --eval_block 100_000

Processes: P=4 for sweep; P ∈ {1,2,4,8} for scaling

Example results (from this run)
Best test RMSE (sweep): relu, M=512, n=256 → ~119.6571 (train time ~177.9 s)

Fast alt (~same RMSE): relu, M=64, n=128 → ~119.6631 (train time ~49.7 s)

Scaling (relu, M=256, n=256):

P	time (s)	speedup	efficiency
1	157.908	1.00	100%
2	130.635	1.21	60%
4	65.836	2.40	60%
8	104.511	1.51	19%

Observation: good scaling to P=4; diminishing returns at P=8 on one node due to memory bandwidth, comms overhead, and BLAS thread oversubscription.

Implementation highlights
Broadcast only minibatch indices per step (not full epoch permutations).

Chunked RMSE (--eval_block) to bound eval memory on huge N.

Subsampled periodic eval (--eval_sample) for faster R(θ) tracking.

No-copy float32 loads to avoid multi-GB duplicates.

Explicit biases (no large hstack temps).

Early-stop patience on R(θ) trend.

(Optional) BLAS single-threading per rank for better strong scaling.

Troubleshooting
mpiexec not found → install MS-MPI and ensure C:\Program Files\Microsoft MPI\Bin is on PATH.

OOM during eval → increase --eval_block or reduce --hidden.

OOM while loading → ensure arrays saved as float32; loader avoids copies.

P=8 slower than P=4 → set BLAS threads to 1 per rank; increase --batch; run from a local (non-OneDrive) path.

Git hygiene
This repo intentionally excludes large files:

.gitignore excludes: nytaxi2022.csv, nytaxi2022_cleaned.npz, *.npz, *.npy, sample_data/, model_final.npz

Only keep small results (results/*.csv) and plots (plots/*.png) in Git.

If you need to version huge artifacts, use Git LFS (watch quotas) or publish data via releases/cloud storage and link here.

License
MIT — feel free to reuse with attribution.