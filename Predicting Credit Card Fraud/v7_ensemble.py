"""v7 - Ensemble: blend the out-of-fold predictions of earlier iterations.

AUC only cares about ranking, so each model's predictions are converted to ranks
(0..1) before blending. Blend weights are found on the out-of-fold predictions.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from common import OOF_DIR, load, log_result, save_outputs
import os

MODELS = ["v4_lightgbm_tuned", "v5_xgboost", "v6_catboost", "v1_logistic_regression"]
_, _, y = load()


def rank01(a):
    return (rankdata(a) - 1) / (len(a) - 1)


oofs = np.column_stack([rank01(np.load(os.path.join(OOF_DIR, f"{m}_oof.npy"))) for m in MODELS])
tests = np.column_stack([rank01(np.load(os.path.join(OOF_DIR, f"{m}_test.npy"))) for m in MODELS])
for m, col in zip(MODELS, oofs.T):
    print(f"  {m:25s} OOF AUC {roc_auc_score(y, col):.5f}")

equal = roc_auc_score(y, oofs.mean(axis=1))
print(f"  equal-weight blend        OOF AUC {equal:.5f}")


def neg_auc(w):
    w = np.abs(w) / np.abs(w).sum()
    return -roc_auc_score(y, oofs @ w)


res = minimize(neg_auc, np.ones(len(MODELS)) / len(MODELS), method="Nelder-Mead", options={"maxiter": 300})
w = np.abs(res.x) / np.abs(res.x).sum()
auc = -res.fun
print("  weights:", dict(zip(MODELS, np.round(w, 3))))
print(f"v7_ensemble: OOF AUC {auc:.5f}")
save_outputs("v7_ensemble", oofs @ w, tests @ w)
log_result("v7_ensemble", auc, float("nan"), 0,
           "Rank-average blend: " + ", ".join(f"{m} {wi:.2f}" for m, wi in zip(MODELS, w)))
