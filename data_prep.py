# data_prep.py
import pandas as pd, numpy as np
from pathlib import Path

IN = "nytaxi2022.csv"
OUT = "nytaxi_prepared.npz"
RNG = np.random.default_rng(42)

use_cols = [
    "tpep_pickup_datetime","tpep_dropoff_datetime","passenger_count","trip_distance",
    "RatecodeID","PULocationID","DOLocationID","payment_type","extra","total_amount"
]

df = pd.read_csv(IN, usecols=use_cols, low_memory=True)

# Parse datetimes
df["tpep_pickup_datetime"]  = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"], errors="coerce")

# Basic filtering
df = df.dropna(subset=["tpep_pickup_datetime","tpep_dropoff_datetime","total_amount"])
df["trip_duration_min"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds()/60.0
df = df[(df["trip_distance"]>0) & (df["trip_distance"]<200) &
        (df["trip_duration_min"]>0) & (df["trip_duration_min"]<600) &
        (df["total_amount"]>0) & (df["total_amount"]<500)]

# Time features
df["hour"] = df["tpep_pickup_datetime"].dt.hour
df["dow"]  = df["tpep_pickup_datetime"].dt.dayofweek
df["is_weekend"] = (df["dow"]>=5).astype(int)

# Categorical (small-cardinality) one-hots
for col, cats in [("payment_type", sorted(df["payment_type"].dropna().unique())),
                  ("RatecodeID",  sorted(df["RatecodeID"].dropna().unique()))]:
    dummies = pd.get_dummies(df[col].fillna(-1).astype(int), prefix=col)
    df = pd.concat([df, dummies], axis=1)

# Keep numeric/simple encodings
num_cols = [
    "passenger_count","trip_distance","extra","trip_duration_min","hour","dow","is_weekend",
    "PULocationID","DOLocationID"
]
one_hot_cols = [c for c in df.columns if c.startswith("payment_type_") or c.startswith("RatecodeID_")]
X_cols = num_cols + one_hot_cols
y_col = "total_amount"

df = df.dropna(subset=X_cols+[y_col])
X = df[X_cols].to_numpy(dtype=np.float32)
y = df[y_col].to_numpy(dtype=np.float32)

# Scale numerics by train stats later; split first (stratification not needed)
N = len(X)
idx = RNG.permutation(N)
cut = int(0.7*N)
train_idx, test_idx = idx[:cut], idx[cut:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Standardize numerics using train stats
num_idx = [X_cols.index(c) for c in num_cols]
mu = X_train[:, num_idx].mean(axis=0)
sd = X_train[:, num_idx].std(axis=0) + 1e-8

def apply_scale(A):
    A = A.copy()
    A[:, num_idx] = (A[:, num_idx]-mu)/sd
    # Scale location IDs into [0,1] to help stability
    pu_i, do_i = X_cols.index("PULocationID"), X_cols.index("DOLocationID")
    for ii in [pu_i, do_i]:
        A[:, ii] = (A[:, ii]-A[:, ii].min())/(A[:, ii].max()-A[:, ii].min()+1e-8)
    return A

X_train = apply_scale(X_train)
X_test  = apply_scale(X_test)

Path("data").mkdir(exist_ok=True)
np.savez("data/nytaxi_prepared.npz",
         X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
         X_cols=np.array(X_cols, dtype=object), mu=mu, sd=sd, num_idx=np.array(num_idx))
print("Saved data/nytaxi_prepared.npz")
