#!/usr/bin/env python
# train_mpi.py — 1-hidden-layer NN with mpi4py
# Supports:
#   --data <NPZ>          (legacy mode: loads full arrays)
#   --npy_root <folder>   (memmap mode: X_train.npy, y_train.npy, X_test.npy, y_test.npy)
# Periodic sample eval: --eval_every, --eval_sample
# Chunked full RMSE eval: --eval_block
#
# Output on rank 0:
#   - history.csv  (columns: iter,R)   R = MAE on sampled train split
#   - model_final.npz

import argparse, time, math
from pathlib import Path
import numpy as np
from mpi4py import MPI

# ---------- utils ----------

def get_activation(name):
    name = name.lower()
    if name == "relu":
        f  = lambda z: np.maximum(z, 0.0, dtype=z.dtype)
        df = lambda z: (z > 0).astype(z.dtype, copy=False)
    elif name == "tanh":
        f  = np.tanh
        df = lambda z: (1.0 - np.tanh(z)**2).astype(z.dtype, copy=False)
    elif name == "sigmoid":
        def sig(z): return 1.0 / (1.0 + np.exp(-z, dtype=z.dtype))
        f  = sig
        df = lambda z: (sig(z) * (1.0 - sig(z))).astype(z.dtype, copy=False)
    else:
        raise ValueError("activation must be one of: relu, tanh, sigmoid")
    return f, df

def rank_slice(n, size, rank):
    per = (n + size - 1) // size  # ceil(n/size)
    s = min(rank * per, n)
    e = min(s + per, n)
    return s, e

def rmse_parallel_mem(comm, Xmem, ymem, W1, b1, w2, b2, act, block=100_000, slice_range=None):
    """Compute RMSE over a memmap (or ndarray) in blocks; split across ranks by slice_range; allreduce."""
    if slice_range is None:
        s, e = 0, Xmem.shape[0]
    else:
        s, e = slice_range
    SSE = 0.0
    n = 0
    for i in range(s, e, block):
        j = min(i + block, e)
        Xb = Xmem[i:j]
        yb = ymem[i:j].reshape(-1, 1)
        Z = Xb @ W1.T + b1
        H = act(Z)
        yhat = H @ w2 + b2
        d = (yhat - yb).ravel()
        SSE += float(d @ d)
        n   += (j - i)
    SSE = comm.allreduce(SSE, op=MPI.SUM)
    n   = comm.allreduce(n,   op=MPI.SUM)
    return math.sqrt(SSE / max(n, 1))

def mae_sample_parallel(comm, Xsrc, ysrc, act, W1, b1, w2, b2, sample_per_rank, block=100_000, rng=None, slice_range=None):
    """Approximate training metric: MAE on a sample. Each rank evaluates sample_per_rank rows locally, then allreduce."""
    if sample_per_rank <= 0:
        return float("nan")
    if slice_range is None:
        n_local = Xsrc.shape[0]
        offset  = 0
    else:
        s, e = slice_range
        n_local = e - s
        offset  = s
    sample_per_rank = min(sample_per_rank, n_local)
    idx = np.arange(n_local) if rng is None else rng.integers(0, n_local, size=sample_per_rank, endpoint=False)
    if idx.ndim == 0:  # corner case
        idx = np.array([int(idx)])
    if idx.size > sample_per_rank:
        idx = idx[:sample_per_rank]

    SAE = 0.0
    n   = 0
    for k in range(0, idx.size, block):
        ids = idx[k:k+block]
        rows = offset + ids
        Xb = Xsrc[rows]
        yb = ysrc[rows].reshape(-1, 1)
        Z = Xb @ W1.T + b1
        H = act(Z)
        yhat = H @ w2 + b2
        SAE += float(np.abs((yhat - yb).ravel()).sum())
        n   += yb.shape[0]
    SAE = comm.allreduce(SAE, op=MPI.SUM)
    n   = comm.allreduce(n,   op=MPI.SUM)
    return SAE / max(n, 1)

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None, help="NPZ with X_train,y_train,X_test,y_test (legacy full-load mode)")
    ap.add_argument("--npy_root", type=str, default=None, help="Folder with X_train.npy,y_train.npy,X_test.npy,y_test.npy (memmap shard mode)")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--activation", type=str, choices=["relu","sigmoid","tanh"], default="relu")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--eval_sample", type=int, default=2_000_000)
    ap.add_argument("--eval_block", type=int, default=100_000)
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    rng = np.random.default_rng(args.seed + rank)

    act, dact = get_activation(args.activation)

    # ----- Data loading modes -----
    npy_mode = args.npy_root is not None
    if npy_mode:
        root = Path(args.npy_root)
        Xtr_mem = np.load(root / "X_train.npy", mmap_mode="r")
        ytr_mem = np.load(root / "y_train.npy", mmap_mode="r")
        Xte_mem = np.load(root / "X_test.npy",  mmap_mode="r")
        yte_mem = np.load(root / "y_test.npy",  mmap_mode="r")

        N_total, m = Xtr_mem.shape
        s_tr, e_tr = rank_slice(N_total, size, rank)
        Xtr = Xtr_mem[s_tr:e_tr]
        ytr = ytr_mem[s_tr:e_tr]
        N_local = Xtr.shape[0]

        Nte_total = Xte_mem.shape[0]
        s_te, e_te = rank_slice(Nte_total, size, rank)

        # compute true global N via SUM of local counts
        N_global = comm.allreduce(N_local, op=MPI.SUM)
    else:
        if args.data is None:
            if rank == 0:
                print("ERROR: provide either --data <npz> or --npy_root <folder>")
            return
        D = np.load(args.data, allow_pickle=True)
        Xtr = D["X_train"].astype(np.float32, copy=False)
        ytr = D["y_train"].astype(np.float32, copy=False)
        Xte_mem = D["X_test"].astype(np.float32, copy=False)
        yte_mem = D["y_test"].astype(np.float32, copy=False)
        N_global, m = Xtr.shape
        N_local = Xtr.shape[0] if rank == 0 else 0  # so SUM gives the true N once
        N_global = comm.allreduce(N_local, op=MPI.SUM)
        Nte_total = Xte_mem.shape[0]
        s_te, e_te = rank_slice(Nte_total, size, rank)

    # ----- Model init -----
    h = args.hidden
    # He/Xavier-ish scaling
    W1 = (rng.normal(0, 1.0, size=(h, m)).astype(np.float32) / np.sqrt(m)).astype(np.float32)
    b1 = np.zeros((h,), dtype=np.float32)
    w2 = (rng.normal(0, 1.0, size=(h, 1)).astype(np.float32) / np.sqrt(h)).astype(np.float32)
    b2 = np.zeros((1,), dtype=np.float32)

    if rank == 0:
        print(f"N={N_global}, m={m}, hidden={h}, batch={args.batch}, act={args.activation}, lr={args.lr}, epochs={args.epochs}, P={size}")

    # ----- Training loop -----
    steps_per_epoch = (N_global + args.batch - 1) // args.batch
    total_steps = steps_per_epoch * args.epochs

    # history file on rank 0
    if rank == 0:
        with open("history.csv", "w") as f:
            f.write("iter,R\n")

    t0 = time.perf_counter()
    for step in range(1, total_steps + 1):
        # sample local mini-batch indices
        if npy_mode:
            nloc = Xtr.shape[0]
            idx = rng.integers(0, nloc, size=args.batch, endpoint=False)
            Xb = Xtr[idx]
            yb = ytr[idx].reshape(-1, 1)
        else:
            # legacy full-load mode: each rank samples from full data (simple & OK)
            idx = rng.integers(0, Xtr.shape[0], size=args.batch, endpoint=False)
            Xb = Xtr[idx]
            yb = ytr[idx].reshape(-1, 1)

        # forward
        Z = Xb @ W1.T + b1
        H = act(Z)
        yhat = H @ w2 + b2
        e = yhat - yb  # (B,1)

        # grads (average over batch locally)
        B = Xb.shape[0]
        gw2 = (H.T @ e) / B                 # (h,1)
        gb2 = e.mean(axis=0)                # (1,)
        dH  = e @ w2.T                      # (B,h)
        dZ  = dH * dact(Z)                  # (B,h)
        gW1 = (dZ.T @ Xb) / B               # (h,m)
        gb1 = dZ.mean(axis=0)               # (h,)

        # allreduce (sum) -> average across ranks
        comm.Allreduce(MPI.IN_PLACE, gW1, op=MPI.SUM); gW1 *= (1.0/size)
        comm.Allreduce(MPI.IN_PLACE, gb1, op=MPI.SUM); gb1 *= (1.0/size)
        comm.Allreduce(MPI.IN_PLACE, gw2, op=MPI.SUM); gw2 *= (1.0/size)
        comm.Allreduce(MPI.IN_PLACE, gb2, op=MPI.SUM); gb2 *= (1.0/size)

        # SGD update
        W1 -= args.lr * gW1
        b1 -= args.lr * gb1
        w2 -= args.lr * gw2
        b2 -= args.lr * gb2

        # periodic sample eval (MAE) — cheap & stable
        if args.eval_every > 0 and (step % args.eval_every == 0):
            # distribute sample budget across ranks
            sample_per_rank = max(args.eval_sample // size, 1)
            if npy_mode:
                R = mae_sample_parallel(comm,
                                        Xsrc=(Xtr_mem if npy_mode else Xtr),
                                        ysrc=(ytr_mem if npy_mode else ytr),
                                        act=act, W1=W1, b1=b1, w2=w2, b2=b2,
                                        sample_per_rank=sample_per_rank,
                                        block=args.eval_block, rng=rng,
                                        slice_range=(s_tr, e_tr) if npy_mode else None)
            else:
                R = mae_sample_parallel(comm,
                                        Xsrc=Xtr, ysrc=ytr,
                                        act=act, W1=W1, b1=b1, w2=w2, b2=b2,
                                        sample_per_rank=sample_per_rank,
                                        block=args.eval_block, rng=rng,
                                        slice_range=None)
            if rank == 0:
                print(f"[iter {step}] R(train)={R:.6f}  (eval=sample, block={args.eval_block})")
                with open("history.csv", "a") as f:
                    f.write(f"{step},{R:.6f}\n")

    # ----- Final evals -----
    # full-train RMSE (over local shard in npy_mode; over full set in legacy)
    if npy_mode:
        tr_rmse = rmse_parallel_mem(comm, Xtr_mem, ytr_mem, W1, b1, w2, b2, act,
                                    block=args.eval_block, slice_range=(s_tr, e_tr))
    else:
        tr_rmse = rmse_parallel_mem(comm, Xtr, ytr, W1, b1, w2, b2, act, block=args.eval_block, slice_range=None)

    # full-test RMSE (distributed across ranks)
    te_rmse = rmse_parallel_mem(comm, Xte_mem, yte_mem, W1, b1, w2, b2, act,
                                block=args.eval_block, slice_range=(s_te, e_te))

    t1 = time.perf_counter()

    if rank == 0:
        print(f"Done | time={t1 - t0:.2f}s  RMSE train={tr_rmse:.4f}  test={te_rmse:.4f}")
        np.savez_compressed("model_final.npz", W1=W1, b1=b1, w2=w2, b2=b2,
                            meta=dict(hidden=args.hidden, act=args.activation,
                                      batch=args.batch, lr=args.lr, epochs=args.epochs,
                                      train_rmse=float(tr_rmse), test_rmse=float(te_rmse),
                                      time_sec=float(t1 - t0), P=size, m=m, N=N_global))

if __name__ == "__main__":
    main()
