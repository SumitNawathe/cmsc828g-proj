"""Predict per-prompt rollout-length Log-t parameters using the trained MLP.

CLI:
    python predict.py 42                  # single sample
    python predict.py 42 137 9            # multiple samples
    python predict.py --all               # all samples in the embedding parquet

Library:
    from predict import predict
    predict(42)            -> {"sample_idx": 42, "mu": ..., "sigma": ..., "nu": ...}
    predict([42, 137])     -> [ {...}, {...} ]

The returned (mu, sigma, nu) parameterize a Student-t over log(L):
    log L ~ t(df=nu, loc=mu, scale=sigma)

To convert to length-space tail/quantile:
    P(L > tau) = 1 - t.cdf((log(tau) - mu) / sigma, df=nu)
    L_q        = exp( t.ppf(q, df=nu, loc=mu, scale=sigma) )
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ARTIFACT = Path(__file__).resolve().parent / "mlp_logt.pt"


class MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, d_out),
        )

    def forward(self, x):
        return self.net(x)


_CACHE: dict = {}


def _load():
    if "model" in _CACHE:
        return _CACHE
    blob = torch.load(ARTIFACT, map_location="cpu", weights_only=False)
    model = MLP(blob["d_in"])
    model.load_state_dict(blob["state_dict"])
    model.eval()
    embed_df = pd.read_parquet(blob["embed_path"]).set_index("sample_idx")
    _CACHE.update({"model": model, "embed_df": embed_df, "blob": blob})
    return _CACHE


def predict(sample_idx):
    """Predict (mu, sigma, nu) for one or many sample_idx values."""
    c = _load()
    single = isinstance(sample_idx, (int, np.integer))
    ids = [int(sample_idx)] if single else [int(i) for i in sample_idx]
    cols = c["blob"]["embed_dim_cols"]
    X = c["embed_df"].loc[ids, cols].values.astype(np.float32)
    with torch.no_grad():
        Yn = c["model"](torch.from_numpy(X)).numpy()
    Y = Yn * c["blob"]["sd_y"] + c["blob"]["mu_y"]
    nu = float(c["blob"]["nu_star"])
    out = [{"sample_idx": sid,
            "mu": float(Y[i, 0]),
            "sigma": float(np.exp(Y[i, 1])),
            "nu": nu} for i, sid in enumerate(ids)]
    return out[0] if single else out


def _main():
    ap = argparse.ArgumentParser(description="MLP Log-t predictor.")
    ap.add_argument("sample_idx", nargs="*", type=int)
    ap.add_argument("--all", action="store_true", help="predict every sample in the embedding file")
    ap.add_argument("--json", action="store_true", help="emit JSONL instead of TSV")
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
