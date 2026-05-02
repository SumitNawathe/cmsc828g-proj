"""Predict per-prompt rollout-length Log-t parameters using the trained XGBoost.

CLI:
    python predict.py 42
    python predict.py 42 137 9
    python predict.py --all

Library:
    from predict import predict
    predict(42)
    predict([42, 137])

Returned (mu, sigma, nu) parameterize a Student-t over log(L):
    log L ~ t(df=nu, loc=mu, scale=sigma)
"""
from __future__ import annotations
import argparse, json, pickle
from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACT = Path(__file__).resolve().parent / "xgb_logt.pkl"

_CACHE: dict = {}


def _load():
    if "model" in _CACHE:
        return _CACHE
    with open(ARTIFACT, "rb") as f:
        blob = pickle.load(f)
    embed_df = pd.read_parquet(blob["embed_path"]).set_index("sample_idx")
    _CACHE.update({"model": blob["model"], "embed_df": embed_df, "blob": blob})
    return _CACHE


def predict(sample_idx):
    c = _load()
    single = isinstance(sample_idx, (int, np.integer))
    ids = [int(sample_idx)] if single else [int(i) for i in sample_idx]
    cols = c["blob"]["embed_dim_cols"]
    X = c["embed_df"].loc[ids, cols].values.astype(np.float32)
    Y = c["model"].predict(X).astype(np.float32)
    nu = float(c["blob"]["nu_star"])
    out = [{"sample_idx": sid,
            "mu": float(Y[i, 0]),
            "sigma": float(np.exp(Y[i, 1])),
            "nu": nu} for i, sid in enumerate(ids)]
    return out[0] if single else out


def _main():
    ap = argparse.ArgumentParser(description="XGBoost Log-t predictor.")
    ap.add_argument("sample_idx", nargs="*", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    c = _load()
    ids = c["embed_df"].index.tolist() if args.all else args.sample_idx
    if not ids:
        ap.error("provide sample_idx or --all")
    rows = predict(ids)
    if args.json:
        for r in rows:
            print(json.dumps(r))
    else:
        print("sample_idx\tmu\tsigma\tnu")
        for r in rows:
            print(f"{r['sample_idx']}\t{r['mu']:.4f}\t{r['sigma']:.4f}\t{r['nu']:.4f}")


if __name__ == "__main__":
    _main()
