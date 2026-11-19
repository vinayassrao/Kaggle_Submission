"""v15 - Leak-free stacking of the strongest, most diverse models.

Each model's OOF prediction is converted to within-fold standardised log-odds (and test
predictions are already stored that way by run_cv_ff). A small logistic regression (the
meta-model) is trained on 4 folds' OOF predictions and scored on the 5th, rotating, so the
reported AUC is honest. A non-negative hill-climbing blend is reported for comparison.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import DATA, N_FOLDS, OOF_DIR, SEED, SUB_DIR, TARGET, load, log_result
from fe_v9 import to_std_logit

DEFAULT = ["v9_lightgbm_woe", "v10_spline_logreg", "v11_ebm", "v12_xgboost_woe", "v13_catboost_woe", "v14_lightgbm_extratrees"]
models = [m for m in (sys.argv[1:] or DEFAULT) if os.path.exists(os.path.join(OOF_DIR, f"{m}_oof.npy"))]
_, _, y = load()
folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))


def oof_z(name):
    o = np.load(os.path.join(OOF_DIR, f"{name}_oof.npy")); z = np.zeros_like(o)
    for _, va in folds:
        z[va] = to_std_logit(o[va])
    return z


def test_z(name):
    t = np.load(os.path.join(OOF_DIR, f"{name}_test.npy"))
    # older runs (common.run_cv) saved probabilities; run_cv_ff saved std log-odds already
    return to_std_logit(t) if t.min() >= 0 and t.max() <= 1 else t


Z = np.column_stack([oof_z(m) for m in models]); T = np.column_stack([test_z(m) for m in models])
print("models:", models)
for m, col in zip(models, Z.T):
    print(f"  {m:28s} OOF AUC {roc_auc_score(y, col):.5f}")
print("  Spearman correlations:\n", pd.DataFrame(Z, columns=models).corr(method="spearman").round(3).to_string())

# 1. Logistic-regression stack, nested over the same folds
meta_oof = np.zeros(len(y))
for tr, va in folds:
    lr = LogisticRegression(C=0.05, max_iter=1000).fit(Z[tr], y[tr])
    meta_oof[va] = lr.decision_function(Z[va])
stack_auc = roc_auc_score(y, meta_oof)
lr_full = LogisticRegression(C=0.05, max_iter=1000).fit(Z, y)
print(f"LogReg stack (nested CV): AUC {stack_auc:.5f}  coefs {dict(zip(models, np.round(lr_full.coef_[0], 3)))}")

# 2. Hill climbing with non-negative weights (greedy, with replacement), also nested
def hill(Ztr, ytr, iters=60):
    w = np.zeros(Ztr.shape[1]); best = -1
    for _ in range(iters):
        cand = [(roc_auc_score(ytr, Ztr @ (w + np.eye(len(w))[j])), j) for j in range(len(w))]
        s, j = max(cand)
        if s <= best + 1e-7:
            break
        best = s; w[j] += 1
    return w / w.sum()


hill_oof = np.zeros(len(y))
for tr, va in folds:
    hill_oof[va] = Z[va] @ hill(Z[tr], y[tr])
hill_auc = roc_auc_score(y, hill_oof)
w_full = hill(Z, y)
print(f"Hill-climb blend (nested CV): AUC {hill_auc:.5f}  weights {dict(zip(models, np.round(w_full, 3)))}")

use_stack = stack_auc >= hill_auc
final_oof = meta_oof if use_stack else hill_oof
final_test = lr_full.decision_function(T) if use_stack else T @ w_full
name = "v15_stack"
np.save(os.path.join(OOF_DIR, f"{name}_oof.npy"), final_oof)
np.save(os.path.join(OOF_DIR, f"{name}_test.npy"), final_test)
ids = pd.read_csv(os.path.join(DATA, "test.csv"), usecols=["id"])["id"]
pd.DataFrame({"id": ids, TARGET: 1 / (1 + np.exp(-final_test))}).to_csv(os.path.join(SUB_DIR, f"{name}.csv"), index=False)
desc = ("LogReg stack" if use_stack else "Hill-climb blend") + " of " + ", ".join(models)
log_result(name, max(stack_auc, hill_auc), np.nan, 0, desc)
print(f"{name}: {desc} -> AUC {max(stack_auc, hill_auc):.5f}")
