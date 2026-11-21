"""Read-only check: does v25_gpt2 add to the final v31 pool? Counts folds up across all 3 stacker splits (15 folds).
Same inputs and combiner as v31_stack.py (LogReg C=0.05 on native-fold-standardised rank-logits). Writes nothing."""
import os

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import OOF_DIR, load
from fe_keys import rank_logit, safe_logit

POOL = ["v17_glm_lgb10", "v17b_glm_lgb10_s100", "v28_glm_xgb10_gpu", "v19_sparse_glm_10f", "v16_glm_lgb",
        "v26_realmlp", "v27_digit_cnn", "v11_ebm", "v22_lgb4_raw", "v10_spline_logreg", "v28_glm_only"]
NATIVE10 = ("lgb10", "xgb10", "_10f", "glm_only")
_, _, y = load(); n = len(y)
r01 = lambda v: (rankdata(v) - 1) / (len(v) - 1)


def member(m):
    z = safe_logit(np.load(os.path.join(OOF_DIR, f"{m}_oof.npy"))); zs = np.zeros_like(z)
    k = 10 if any(t in m for t in NATIVE10) else 5
    for _, va in StratifiedKFold(k, shuffle=True, random_state=42).split(np.zeros(n), y):
        zs[va] = (z[va] - z[va].mean()) / (z[va].std() + 1e-12)
    return rank_logit(zs)


Zb = np.column_stack([member(m) for m in POOL]); g = member("v25_gpt2")
for C in (0.03, 0.05):
    tot = {"base": [], "+v25": []}; up = 0; diffs = []
    for s in (43, 44, 45):
        pb, pg = np.zeros(n), np.zeros(n)
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=s).split(np.zeros(n), y):
            for Z, p in ((Zb, pb), (np.column_stack([Zb, g]), pg)):
                p[va] = r01(LogisticRegression(C=C, solver="newton-cholesky", max_iter=200).fit(Z[tr], y[tr]).decision_function(Z[va]))
            d = roc_auc_score(y[va], pg[va]) - roc_auc_score(y[va], pb[va]); diffs.append(d); up += d > 0
        tot["base"].append(roc_auc_score(y, pb)); tot["+v25"].append(roc_auc_score(y, pg))
    gain = np.mean(tot["+v25"]) - np.mean(tot["base"])
    print(f"C={C}: v31 pool {np.mean(tot['base']):.5f} | + v25 {np.mean(tot['+v25']):.5f} | gain {gain:+.5f} "
          f"| per-split {np.round(np.array(tot['+v25']) - np.array(tot['base']), 5)} | folds up {up}/15 | mean fold diff {np.mean(diffs):+.6f}")
Zf = np.column_stack([Zb, g])
lr = LogisticRegression(C=0.05, solver="newton-cholesky", max_iter=200).fit(Zf, y)
print("coef v25 with full pool (C=0.05):", round(float(lr.coef_[0][-1]), 3))
