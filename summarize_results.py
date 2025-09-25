# summarize_results.py — rebuild sweep tables & plots (robust to path/header)
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PLOTS = ROOT / "plots"
RESULTS = ROOT / "results"
PLOTS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

# --- find run_summary.csv ---
candidates = [RESULTS / "run_summary.csv", ROOT / "run_summary.csv"]
for c in candidates:
    if c.exists():
        summary_path = c
        break
else:
    raise FileNotFoundError("run_summary.csv not found in results/ or repo root")

expected = ["activation","batch","hidden","lr","procs","train_time","rmse_train","rmse_test"]

# --- read with header if present; else assign names ---
try:
    df = pd.read_csv(summary_path)
    if not set(expected).issubset(df.columns):
        raise ValueError("columns don't match; retry without header")
except Exception:
    df = pd.read_csv(summary_path, header=None, names=expected)

# --- types ---
for col in ["batch","hidden","procs"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
for col in ["train_time","rmse_train","rmse_test"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# drop bad rows if any
df = df.dropna(subset=["activation","batch","hidden","lr","procs","train_time","rmse_test"])

# --- best per (activation, batch): choose lowest rmse_test then time ---
df["rank"] = df.groupby(["activation","batch"])\
               .apply(lambda g: g[["rmse_test","train_time"]]
                      .rank(method="first").sum(axis=1))\
               .reset_index(level=[0,1], drop=True)
best_ab = df.sort_values(["activation","batch","rmse_test","train_time"])\
            .groupby(["activation","batch"], as_index=False).first()[expected]

# --- top-5 overall (lowest rmse_test, then time) ---
top5 = df.sort_values(["rmse_test","train_time"]).head(5)[expected]

# save tables
(best_ab)[expected].to_csv(RESULTS / "best_per_activation_batch.csv", index=False)
(top5)[expected].to_csv(RESULTS / "top5_overall.csv", index=False)

# --- plots: rmse vs batch, time vs batch (pick best per activation/batch) ---
plot_df = df.sort_values(["activation","batch","rmse_test","train_time"])\
            .groupby(["activation","batch"], as_index=False).first()

def lineplot(x, y, hue, outpng, ylabel, title):
    plt.figure(figsize=(7.5,4.5))
    for act, g in plot_df.groupby(hue):
        gg = g.sort_values(x)
        plt.plot(gg[x], gg[y], marker="o", label=str(act))
    plt.xlabel(x)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS / outpng, dpi=140)
    plt.close()

lineplot("batch", "rmse_test", "activation",
         "rmse_vs_batch.png", "Test RMSE ↓", "RMSE vs Batch (best per σ,M)")

lineplot("batch", "train_time", "activation",
         "time_vs_batch.png", "Time (s) ↓", "Time vs Batch (best per σ,M)")

print(f"✔ Wrote: {RESULTS/'best_per_activation_batch.csv'}, {RESULTS/'top5_overall.csv'}")
print(f"✔ Saved: {PLOTS/'rmse_vs_batch.png'}, {PLOTS/'time_vs_batch.png'}")
