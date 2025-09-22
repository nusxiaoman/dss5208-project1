# summarize_results.py
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

Path("results").mkdir(exist_ok=True)
Path("plots").mkdir(exist_ok=True)

# read sweep rows
df = pd.read_csv("run_summary.csv", header=None,
                 names=["activation","batch","hidden","lr","procs","train_time","rmse_train","rmse_test"])

# 2.1 best per (activation,batch)
best = (df.sort_values(["rmse_test","train_time"])
          .groupby(["activation","batch"], as_index=False)
          .first())
best.to_csv("results/best_per_activation_batch.csv", index=False)

# 2.2 top-5 overall (lowest test RMSE, tie-break by time)
top5 = df.sort_values(["rmse_test","train_time"]).head(5)
top5.to_csv("results/top5_overall.csv", index=False)

# 2.3 RMSE vs batch by activation
plt.figure()
for act, g in best.groupby("activation"):
    g = g.sort_values("batch")
    plt.plot(g["batch"], g["rmse_test"], marker="o", label=act)
plt.xlabel("Batch size M"); plt.ylabel("Test RMSE")
plt.title("Test RMSE vs batch size (best hidden per pair)")
plt.legend()
plt.tight_layout(); plt.savefig("plots/rmse_vs_batch.png", dpi=150)

# 2.4 Train time vs batch (P fixed in sweep)
plt.figure()
for act, g in best.groupby("activation"):
    g = g.sort_values("batch")
    plt.plot(g["batch"], g["train_time"], marker="o", label=act)
plt.xlabel("Batch size M"); plt.ylabel("Train time (s)")
plt.title("Training time vs batch size (P fixed)")
plt.legend()
plt.tight_layout(); plt.savefig("plots/time_vs_batch.png", dpi=150)

# 2.5 If scaling rows exist later, make a scaling curve for the best triple
# (the scaling script you’ll run next will append more rows with different 'procs')
best_row = df.sort_values(["rmse_test","train_time"]).iloc[0]
mask = (df.activation==best_row.activation)&(df.batch==best_row.batch)&(df.hidden==best_row.hidden)
sc = df[mask].sort_values("procs")
if sc["procs"].nunique() > 1:
    sc.to_csv("results/scaling_table.csv", index=False)
    plt.figure()
    plt.plot(sc["procs"], sc["train_time"], marker="o")
    plt.xlabel("Processes (P)"); plt.ylabel("Training time (s)")
    plt.title(f"Scaling: {best_row.activation}, M={best_row.batch}, n={best_row.hidden}")
    plt.tight_layout(); plt.savefig("plots/scaling_curve.png", dpi=150)

print("Wrote: results/best_per_activation_batch.csv, results/top5_overall.csv")
print("Saved: plots/rmse_vs_batch.png, plots/time_vs_batch.png", 
      "(plus plots/scaling_curve.png after scaling runs)")
