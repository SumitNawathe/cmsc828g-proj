"""Predict per-prompt rollout-length Log-t parameters by retrieval (top-k pooled).

For each query sample_idx:
  1. find the top-k most cosine-similar training prompts
  2. pool their log-rollout-lengths
  3. fit (mu, sigma) by Student-t IRLS at the global nu*

CLI:
    python predict.py 42                 # default k=10
    python predict.py 42 --k 5
    python predict.py 42 137 9 --k 5
    python predict.py --all --k 10

Library:
    from predict import predict
    predict(42, k=5)
    predict([42, 137], k=10)

Returned (mu, sigma, nu) parameterize a Student-t over log(L):
    log L ~ t(df=nu, loc=mu, scale=sigma)

Self-matches are excluded — if the query is a training prompt, it is dropped
from the candidate pool before top-k.
"""
from __future__ import annotations
import argparse, json, pickle
from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACT = Path(__file__).resolve().parent / "retrieval_logt.pkl"
ALLOWED_K = (5, 10)

_CACHE: dict = {}


def _load():
    if "blob" in _CACHE:
        return _CACHE
    with open(ARTIFACT, "rb") as f:
        blob = pickle.load(f)
    embed_df = pd.read_parquet(blob["embed_path"]).set_index("sample_idx")
    Xt = blob["X_train"].astype(np.float32)
    Xt_n = Xt / (np.linalg.norm(Xt, axis=1, keepdims=True) + 1e-9)
    _CACHE.update({"blob": blob, "embed_df": embed_df, "Xt_n": Xt_n})
    return _CACHE


def _fit_mu_sigma_fixed_nu(y: np.ndarray, nu: float, n_iter: int = 25, tol: float = 1e-6):
    mu = float(np.median(y))
    sigma = float(np.std(y) + 1e-6)
    for _ in range(n_iter):
        z2 = ((y - mu) / sigma) ** 2
        w = (nu + 1.0) / (nu + z2)
        mu_new = float(np.sum(w * y) / np.sum(w))
        sigma2_new = float(np.sum(w * (y - mu_new) ** 2) / len(y))
        sigma_new = float(np.sqrt(max(sigma2_new, 1e-12)))
        if abs(mu_new - mu) < tol and abs(sigma_new - sigma) < tol:
            return mu_new, sigma_new
        mu, sigma = mu_new, sigma_new
    return mu, sigma


def predict(sample_idx, k: int = 10):
    if k not in ALLOWED_K:
        raise ValueError(f"k must be one of {ALLOWED_K}, got {k}")
    c = _load()
    blob, embed_df, Xt_n = c["blob"], c["embed_df"], c["Xt_n"]
    train_ids = blob["train_ids"]
    pools = blob["train_log_lengths"]
    nu = float(blob["nu_star"])
    cols = blob["embed_dim_cols"]

    single = isinstance(sample_idx, (int, np.integer))
    ids = [int(sample_idx)] if single else [int(i) for i in sample_idx]

    Xq = embed_df.loc[ids, cols].values.astype(np.float32)
    Xq_n = Xq / (np.linalg.norm(Xq, axis=1, keepdims=True) + 1e-9)
    sim = Xq_n @ Xt_n.T  # (Q, N_train)

    train_id_arr = np.asarray(train_ids)
    out = []
    for i, sid in enumerate(ids):
        s = sim[i].copy()
        if sid in pools:  # self-match: exclude
            j = train_ids.index(sid)
            s[j] = -np.inf
        top = np.argpartition(-s, k)[:k]
        top = top[np.argsort(-s[top])]
        pooled = np.concatenate([pools[int(train_id_arr[j])] for j in top])
        mu, sigma = _fit_mu_sigma_fixed_nu(pooled, nu)
        out.append({"sample_idx": sid, "mu": mu, "sigma": sigma, "nu": nu, "k": k})
    return out[0] if single else out


def _main():
    ap = argparse.ArgumentParser(description="Retrieval Log-t predictor.")
    ap.add_argument("sample_idx", nargs="*", type=int)
    ap.add_argument("--k", type=int, default=10, choices=ALLOWED_K)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    c = _load()
    ids = c["embed_df"].index.tolist() if args.all else args.sample_idx
    if not ids:
        ap.error("provide sample_idx or --all")
    rows = predict(ids, k=args.k)
    if args.json:
        for r in rows:
            print(json.dumps(r))
    else:
        print("sample_idx\tmu\tsigma\tnu\tk")
        for r in rows:
            print(f"{r['sample_idx']}\t{r['mu']:.4f}\t{r['sigma']:.4f}\t{r['nu']:.4f}\t{r['k']}")


if __name__ == "__main__":
    _main()
