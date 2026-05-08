# Log-t Rollout-Length Predictors

Each predictor takes a `sample_idx` (an index into the Qwen3-Embedding-4B parquet
of MATH500 prompts) and returns Log-t parameters `(mu, sigma, nu)` such that

```
log(rollout_length)  ~  Student-t(df=nu, loc=mu, scale=sigma)
```

Use these to derive whatever the scheduler needs:

```python
from scipy.stats import t
P_over_tau = 1 - t.cdf((np.log(tau) - mu) / sigma, df=nu)        # tail prob
L_q        = float(np.exp(t.ppf(q, df=nu, loc=mu, scale=sigma))) # length quantile
```

## Held-out test results (100 MATH500 prompts)


| model          | NLL  | Wasserstein | Tail-MAE  |
| -------------- | ---- | ----------- | --------- |
| MLP            | 1.55 | **2102**    | **0.095** |
| XGBoost        | 2.15 | 2268        | 0.109     |
| Retrieval k=10 | 0.81 | 2900        | 0.124     |


Recommended for scheduling: **MLP** (lowest tail error and Wasserstein).

## One-time setup

Run all cells of `len_logt_pred.ipynb` once. Cell 13 saves three artifacts:

```
mlp-inference/mlp_logt.pt
xgb-inference/xgb_logt.pkl
retrieval-inference/retrieval_logt.pkl
```

Python deps: `numpy pandas scipy scikit-learn xgboost torch pyarrow`.

## Usage (same for all 3)

```bash
# single sample
python mlp-inference/predict.py 42
python xgb-inference/predict.py 42
python retrieval-inference/predict.py 42 --k 10   # or --k 5

# multiple samples
python mlp-inference/predict.py 42 137 9

# every sample in the embedding parquet
python mlp-inference/predict.py --all

# machine-readable
python mlp-inference/predict.py 42 137 --json
```

Output (TSV):

```
sample_idx	mu	sigma	nu
42	8.4123	0.5314	7.8612
```

## Usage (Python)

```python
from predict import predict          # cd into the chosen subdir, or add to sys.path
predict(42)                          # -> {"sample_idx": 42, "mu": ..., "sigma": ..., "nu": ...}
predict([42, 137])                   # -> [ {...}, {...} ]
predict(42, k=5)                     # retrieval only
```

The first call loads the artifact + embedding parquet; subsequent calls hit an
in-process cache.

## Notes

- `nu` is fixed globally (median of per-prompt fits on the train set). MLP and
XGB predict `(mu, log sigma)` from the prompt embedding; the script
exponentiates `log sigma` for you.
- Retrieval excludes self-matches automatically (safe to query a training id).
- Inference runs on CPU; a single sample takes a few ms after warm-up.

