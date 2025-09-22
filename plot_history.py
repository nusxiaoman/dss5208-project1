# plot_history.py
import argparse, numpy as np, matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--in_csv", required=True)
ap.add_argument("--out_png", required=True)
ap.add_argument("--title", default="Training history")
args = ap.parse_args()

H = np.loadtxt(args.in_csv, delimiter=",", skiprows=1)
if H.ndim == 1:
    H = H.reshape(1, -1)
iters, R = H[:,0], H[:,1]
plt.figure()
plt.plot(iters, R)
plt.xlabel("Iteration"); plt.ylabel("R(theta)")
plt.title(args.title)
plt.tight_layout()
plt.savefig(args.out_png, dpi=150)
print(f"Saved {args.out_png}")

