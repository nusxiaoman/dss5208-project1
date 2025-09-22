# train_mpi.py — optimized, with --eval_sample, chunked RMSE, and no-copy loads
import argparse
import numpy as np
from mpi4py import MPI

# ---------- activations ----------
def relu(x):    return np.maximum(x, 0)
def d_relu(x):  return (x > 0).astype(x.dtype)
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1.0 - s)
def tanh(x):    return np.tanh(x)
def d_tanh(x):  return 1.0 - np.tanh(x) ** 2

ACTS = {"relu": (relu, d_relu), "sigmoid": (sigmoid, d_sigmoid), "tanh": (tanh, d_tanh)}

# ---------- init & sharding ----------
def init_params(m, n, rng):
    # He/Xavier-like init in float32
    W1 = rng.normal(0.0, 1.0/np.sqrt(m), size=(n, m)).astype(np.float32)
    b1 = np.zeros((n,), dtype=np.float32)
    w2 = rng.normal(0.0, 1.0/np.sqrt(n), size=(n,)).astype(np.float32)
    b2 = np.float32(0.0)
    return W1, b1, w2, b2

def shard(N, rank, size):
    base = N // size
    rem  = N %  size
    start = rank * base + min(rank, rem)
    end   = start + base + (1 if rank < rem else 0)
    return start, end

# ---------- helpers ----------
def as_float32_no_copy(a: np.ndarray) -> np.ndarray:
    """Avoid huge duplicate allocations if array is already float32."""
    return a if a.dtype == np.float32 else a.astype(np.float32)

# Chunked RMSE to keep memory bounded (works for full or subset eval)
def rmse_parallel(comm, X, y, W1, b1, w2, b2, act, idx=None, block=100_000):
    rank = comm.Get_rank()
    size = comm.Get_size()
    Ntot = len(X)

    if idx is None:
        a0, a1 = shard(Ntot, rank, size)
        Xs, ys = X[a0:a1], y[a0:a1]
    else:
        a0, a1 = shard(Ntot, rank, size)
        mask = (idx >= a0) & (idx < a1)
        loc  = (idx[mask] - a0).astype(np.int64)
        Xs, ys = X[a0:a1][loc], y[a0:a1][loc]

    sse_local = 0.0
    cnt_local = 0
    for i in range(0, len(Xs), block):
        Xi = Xs[i:i+block]
        yi = ys[i:i+block]
        if len(Xi) == 0:
            continue
        Z = Xi @ W1.T
        Z += b1
        H = act(Z)
        yhat = H @ w2 + b2
        diff = (yhat - yi).astype(np.float64, copy=False)
        sse_local += float(np.dot(diff, diff))
        cnt_local += int(len(diff))

    local = np.array([sse_local, cnt_local], dtype=np.float64)
    global_ = np.zeros_like(local)
    comm.Allreduce(local, global_, op=MPI.SUM)
    sse, cnt = global_
    cnt = int(cnt)
    return 0.0 if cnt == 0 else float(np.sqrt(sse / cnt))

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="nytaxi2022_cleaned.npz")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--activation", choices=list(ACTS), default="relu")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--eval_sample", type=int, default=0,
                    help="Rows of TRAIN set to sample (global) for periodic R(theta) checks; 0 = full train.")
    ap.add_argument("--eval_block", type=int, default=100_000,
                    help="Block size for RMSE evaluation to limit memory use.")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    rng = np.random.default_rng(args.seed + rank * 31)

    # ---- load data (no unnecessary copies) ----
    D = np.load(args.data, allow_pickle=True)
    Xtr = as_float32_no_copy(D["X_train"])
    ytr = as_float32_no_copy(D["y_train"])
    Xte = as_float32_no_copy(D["X_test"])
    yte = as_float32_no_copy(D["y_test"])
    N, m = Xtr.shape

    # local shard view
    s0, s1 = shard(N, rank, size)
    X, y = Xtr[s0:s1], ytr[s0:s1]

    if rank == 0:
        print(f"N={N}, m={m}, hidden={args.hidden}, batch={args.batch}, act={args.activation}, "
              f"lr={args.lr}, epochs={args.epochs}, P={size}", flush=True)

    # init params
    W1, b1, w2, b2 = init_params(m, args.hidden, rng)
    act, d_act = ACTS[args.activation]

    # evaluation subsample (global indices)
    eval_idx = None
    if args.eval_sample and args.eval_sample > 0:
        if rank == 0:
            k = min(args.eval_sample, N)
            eval_idx = np.random.default_rng(args.seed).choice(N, size=k, replace=False).astype(np.int64)
        eval_idx = comm.bcast(eval_idx, root=0)

    history = []  # (iter, R)
    patience = 5
    best_R = np.inf
    bad = 0
    it = 0

    t0 = MPI.Wtime()
    for ep in range(args.epochs):
        # shuffle on root only
        if rank == 0:
            order = np.arange(N, dtype=np.int64)
            np.random.default_rng(args.seed + ep).shuffle(order)

        for off in range(0, N, args.batch):
            it += 1

            # broadcast only this minibatch indices
            batch_idx = order[off: off + args.batch].astype(np.int32, copy=False) if rank == 0 else None
            batch_idx = comm.bcast(batch_idx, root=0)

            # local rows
            mask = (batch_idx >= s0) & (batch_idx < s1)
            loc_idx = (batch_idx[mask] - s0).astype(np.int64)
            Xb = X[loc_idx] if len(loc_idx) else np.empty((0, m), np.float32)
            yb = y[loc_idx] if len(loc_idx) else np.empty((0,), np.float32)

            if len(Xb):
                Z = Xb @ W1.T + b1            # (B, n)
                H = act(Z)
                yhat = H @ w2 + b2
                M = max(len(batch_idx), 1)    # global minibatch size
                diff = (yhat - yb) / M
                # local grads
                gw2_local = H.T @ diff
                gb2_local = np.array([diff.sum()], np.float32)
                dH = diff[:, None] * w2[None, :]
                dZ = dH * d_act(Z)
                gW1_local = dZ.T @ Xb
                gb1_local = dZ.sum(axis=0).astype(np.float32)
            else:
                gw2_local = np.zeros_like(w2)
                gb2_local = np.zeros((1,), dtype=np.float32)
                gW1_local = np.zeros_like(W1)
                gb1_local = np.zeros_like(b1)

            # allreduce grads
            gw2 = np.zeros_like(w2);         comm.Allreduce(gw2_local, gw2, op=MPI.SUM)
            gb2 = np.zeros_like(gb2_local);  comm.Allreduce(gb2_local, gb2, op=MPI.SUM)
            gW1 = np.zeros_like(W1);         comm.Allreduce(gW1_local, gW1, op=MPI.SUM)
            gb1 = np.zeros_like(b1);         comm.Allreduce(gb1_local, gb1, op=MPI.SUM)

            # SGD update
            W1 -= args.lr * gW1
            b1 -= args.lr * gb1
            w2 -= args.lr * gw2
            b2 -= args.lr * gb2[0]

            # periodic R(theta): sample or full (both chunked)
            if it % args.eval_every == 0:
                rmse = rmse_parallel(comm, Xtr, ytr, W1, b1, w2, b2, act,
                                     idx=eval_idx, block=args.eval_block)
                R = 0.5 * (rmse ** 2)
                if rank == 0:
                    history.append((it, float(R)))
                    print(f"[iter {it}] R(train)={R:.6f}  "
                          f"(eval={'sample' if eval_idx is not None else 'full'}, block={args.eval_block})",
                          flush=True)
                    if R + 1e-6 < best_R: best_R, bad = R, 0
                    else: bad += 1
                bad = comm.bcast(bad, root=0)
                if bad >= patience:
                    break
        if bad >= patience:
            break

    train_time = MPI.Wtime() - t0

    # final RMSEs (full sets, chunked)
    tr_rmse = rmse_parallel(comm, Xtr, ytr, W1, b1, w2, b2, act, block=args.eval_block)
    te_rmse = rmse_parallel(comm, Xte, yte, W1, b1, w2, b2, act, block=args.eval_block)

    if rank == 0:
        print(f"Done | time={train_time:.2f}s  RMSE train={tr_rmse:.4f}  test={te_rmse:.4f}", flush=True)

        hist = np.array(history if history else [[0, 0.5 * (tr_rmse ** 2)]], dtype=np.float64)
        np.savetxt("history.csv", hist, delimiter=",", header="iter,R", comments="")

        np.savez("model_final.npz", W1=W1, b1=b1, w2=w2, b2=b2,
                 meta=dict(hidden=args.hidden, act=args.activation, batch=args.batch, lr=args.lr,
                           epochs=args.epochs, procs=size, train_time=float(train_time),
                           rmse_train=float(tr_rmse), rmse_test=float(te_rmse)))

        with open("run_summary.csv", "a", encoding="utf-8") as f:
            f.write(f"{args.activation},{args.batch},{args.hidden},{args.lr},{size},"
                    f"{train_time:.3f},{tr_rmse:.4f},{te_rmse:.4f}\n")

if __name__ == "__main__":
    main()
