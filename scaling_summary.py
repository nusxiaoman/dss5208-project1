# scaling_summary.py — strong-scaling plots from scaling_table.csv
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
PLOTS = ROOT / "plots"
RESULTS = ROOT / "results"
PLOTS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

candidates = [RESULTS / "scaling_table.csv", ROOT / "scaling_table.csv"]
for c in candidates:
    if c.exists():
        sc_path = c
        break
else:
    raise FileNotFoundError("scaling_table.csv not found in results/ or repo root")

df = pd.read_csv(sc_path)

# normalize columns
df.columns = [c.strip().lower() for c in df.columns]
needed = {"procs","train_time"}
if not needed.issubset(df.columns):
    raise ValueError(f"scaling_table.csv must contain {needed}")

# ensure numeric
df["procs"] = pd.to_numeric(df["procs"], errors="coerce")
df["train_time"] = pd.to_numeric(df["train_time"], errors="coerce")
df = df.dropna(subset=["procs","train_time"])

# compute speedup/efficiency if missing
if "speedup" not in df.columns or df["speedup"].isna().all():
    t1 = df.loc[df["procs"]==1, "train_time"]
    if len(t1)==0:
        raise ValueError("No P=1 baseline found to compute speedup/efficiency.")
    t1 = float(t1.iloc[0])
    df["speedup"] = t1 / df["train_time"]
if "efficiency" not in df.columns or df["efficiency"].isna().all():
    df["efficiency"] = df["speedup"] / df["procs"]

df = df.sort_values("procs")

def simple_plot(x, y, ylabel, title, outpng):
    plt.figure(figsize=(6.5,4.2))
    plt.plot(df[x], df[y], marker="o")
    plt.xlabel(x)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS / outpng, dpi=140)
    plt.close()

simple_plot("procs","train_time","Time (s) ↓","Scaling: Time vs Processes","scaling_time.png")
simple_plot("procs","speedup","Speedup ↑","Scaling: Speedup","scaling_speedup.png")
simple_plot("procs","efficiency","Efficiency (×) ↑","Scaling: Efficiency","scaling_efficiency.png")

print(f"✔ Saved: {PLOTS/'scaling_time.png'}, {PLOTS/'scaling_speedup.png'}, {PLOTS/'scaling_efficiency.png'}")
