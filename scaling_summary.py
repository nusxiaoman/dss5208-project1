import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

Path("results").mkdir(exist_ok=True)
Path("plots").mkdir(exist_ok=True)

cols = ["activation","batch","hidden","lr","procs","train_time","rmse_train","rmse_test"]
df = pd.read_csv("run_summary.csv", header=None, names=cols)

# de-dup repeated configs, keep the last occurrence
df = df.drop_duplicates(subset=["activation","batch","hidden","lr","procs"], keep="last")

# pick the overall-best triple (lowest test RMSE, tie-break by train_time)
best = df.sort_values(["rmse_test","train_time"]).iloc[0]
mask = (df.activation==best.activation) & (df.batch==best.batch) & (df.hidden==best.hidden) & (df.lr==best.lr)
sc = df[mask].sort_values("procs").copy()

# compute speedup & efficiency
if (sc["procs"]==1).any():
    t1 = float(sc.loc[sc["procs"]==1,"train_time"].iloc[0])
    sc["speedup"] = t1 / sc["train_time"]
else:
    p0 = int(sc["procs"].min()); t0 = float(sc.loc[sc["procs"]==p0,"train_time"].iloc[0])
    sc["speedup"] = (t0 / sc["train_time"]) * p0  # normalize to an implied 1-proc baseline

sc["efficiency"] = sc["speedup"] / sc["procs"]

sc.to_csv("results/scaling_table.csv", index=False)

# plots
plt.figure()
plt.plot(sc["procs"], sc["train_time"], marker="o")
plt.xlabel("Processes (P)"); plt.ylabel("Train time (s)")
plt.title(f"Scaling time: {best.activation}, M={int(best.batch)}, n={int(best.hidden)}")
plt.tight_layout(); plt.savefig("plots/scaling_time.png", dpi=150)

plt.figure()
plt.plot(sc["procs"], sc["speedup"], marker="o")
plt.xlabel("Processes (P)"); plt.ylabel("Speedup (T1/Tp)")
plt.title(f"Speedup: {best.activation}, M={int(best.batch)}, n={int(best.hidden)}")
plt.tight_layout(); plt.savefig("plots/scaling_speedup.png", dpi=150)

plt.figure()
plt.plot(sc["procs"], sc["efficiency"]*100, marker="o")
plt.xlabel("Processes (P)"); plt.ylabel("Parallel efficiency (%)")
plt.title(f"Efficiency: {best.activation}, M={int(best.batch)}, n={int(best.hidden)}")
plt.tight_layout(); plt.savefig("plots/scaling_efficiency.png", dpi=150)

print("Wrote results/scaling_table.csv and plots/scaling_*.png")
