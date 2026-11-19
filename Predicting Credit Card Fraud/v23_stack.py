"""v23 - Phase 2 stack: global rank-logit inputs, combiners compared by nested CV on 3 fresh splits
(seeds 43, 44, 45), then a leak-free check of the two zero-fraud zones.

Inputs per model: OOF probabilities and test predictions (full-data refit if available, else fold mean).
Every vector -> rank_logit (same marginal distribution for OOF and test, whatever the model's scale).
Combiners: LogReg C=0.01, LogReg C=0.05, non-negative hill climbing (rank01), and mix01 = 50/50 rank mix
of LogReg C=0.01 and hill climbing.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import DATA, OOF_DIR, SUB_DIR, TARGET, load, log_result
from fe_keys import rank_logit

DEFAULT = ["v16_glm_lgb", "v17_glm_lgb10", "v18_glm_xgb", "v19_sparse_glm", "v20_ebm_digits",
           "v21_mlp_bag", "v22_lgb4_raw", "v11_ebm", "v10_spline_logreg"]
train, test, y = load()
names = [m for m in (sys.argv[1:] or DEFAULT) if os.path.exists(os.path.join(OOF_DIR, f"{m}_oof.npy"))]


def load_model(m):
    oof = np.load(os.path.join(OOF_DIR, f"{m}_oof.npy"))
    for suffix in ("_full.npy", "_test.npy"):
        p = os.path.join(OOF_DIR, m + suffix)
        if os.path.exists(p):
            return oof, np.load(p), suffix
    raise FileNotFoundError(m)


rows = []; O = []; T = []
for m in names:
    o, t, src = load_model(m)
    O.append(rank_logit(o)); T.append(rank_logit(t)); rows.append((m, roc_auc_score(y, o), src))
O, T = np.column_stack(O), np.column_stack(T)
print("members:"); [print(f"  {m:24s} OOF {a:.5f}  test from {s}") for m, a, s in rows]
r01 = lambda v: (rankdata(v) - 1) / (len(v) - 1)


def hill(Z, yy, steps=80):
    R = np.column_stack([r01(c) for c in Z.T]); w = np.zeros(Z.shape[1]); best = -1
    for _ in range(steps):
        s, j = max((roc_auc_score(yy, R @ (w + np.eye(len(w))[j])), j) for j in range(len(w)))
        if s <= best + 1e-7:
            break
        best = s; w[j] += 1
    return w / w.sum()


def combiners(Ztr, ytr, Zev):
    out = {}
    for C in (0.01, 0.05):
        lr = LogisticRegression(C=C, solver="newton-cholesky", max_iter=200).fit(Ztr, ytr)
        out[f"lr_C{C}"] = lr.decision_function(Zev)
    w = hill(Ztr, ytr)
    out["hill"] = np.column_stack([r01(c) for c in Zev.T]) @ w
    out["mix01"] = 0.5 * r01(out["lr_C0.01"]) + 0.5 * r01(out["hill"])
    return out, w


scores = {}
nested = {}
for s in (43, 44, 45):
    pred = {}
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=s).split(O, y):
        out, _ = combiners(O[tr], y[tr], O[va])
        for k, v in out.items():
            pred.setdefault(k, np.zeros(len(y)))[va] = r01(v)   # rank within the validation fold
    for k, v in pred.items():
        scores.setdefault(k, []).append(roc_auc_score(y, v))
        if s == 43:
            nested[k] = v
print("\nnested CV AUC (splits 43/44/45):")
for k, v in scores.items():
    print(f"  {k:9s} {np.mean(v):.5f}  {np.round(v, 5)}")
best = max(scores, key=lambda k: np.mean(scores[k]))
best_auc = float(np.mean(scores[best]))
print(f"best combiner: {best} ({best_auc:.5f})")

full_out, w = combiners(O, y, T)
lr01 = LogisticRegression(C=0.01, solver="newton-cholesky", max_iter=200).fit(O, y)
print("LogReg C=0.01 coefs:", dict(zip(names, np.round(lr01.coef_[0], 3))))
print("hill weights:", dict(zip(names, np.round(w, 3))))
final_test = r01(full_out[best])

# ---- zero-fraud zones, verified leak-free on the base folds (seed 42) using the nested stack OOF ----
mrs, vel = train["merchant_risk_score"].to_numpy(), train["velocity_score"].to_numpy()
st = nested[best]; gains = []
for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(O, y):
    thr = mrs[tr][y[tr] == 1].min()
    z = (mrs[va] < thr) | ((vel[va] <= 4.1) & (mrs[va] <= 20.0))
    zero_in_train = y[tr][(mrs[tr] < thr) | ((vel[tr] <= 4.1) & (mrs[tr] <= 20.0))].sum()
    adj = st[va].copy(); adj[z] -= 10.0
    gains.append(roc_auc_score(y[va], adj) - roc_auc_score(y[va], st[va]))
    print(f"  zone fold: thr {thr:.1f}, zone rows {z.sum()}, frauds in train zone {zero_in_train}, gain {gains[-1]:+.6f}")
use_zones = all(g > 0 for g in gains)
print(f"zones: mean gain {np.mean(gains):+.6f}, positive in {sum(g > 0 for g in gains)}/5 -> {'APPLY' if use_zones else 'skip'}")
if use_zones:
    thr = mrs[y == 1].min()
    tm, tv = test["merchant_risk_score"].to_numpy(), test["velocity_score"].to_numpy()
    zt = (tm < thr) | ((tv <= 4.1) & (tm <= 20.0))
    final_test = final_test.copy(); final_test[zt] -= 10.0; final_test = r01(final_test)
    print(f"  applied to {zt.sum()} test rows (thr {thr:.1f})")

name = "v23_stack"
np.save(os.path.join(OOF_DIR, f"{name}_oof.npy"), nested[best]); np.save(os.path.join(OOF_DIR, f"{name}_test.npy"), final_test)
ids = pd.read_csv(os.path.join(DATA, "test.csv"), usecols=["id"])["id"]
pd.DataFrame({"id": ids, TARGET: final_test}).to_csv(os.path.join(SUB_DIR, f"{name}.csv"), index=False)
log_result(name, best_auc, float(np.std(scores[best])), 0, f"{best} on rank-logits of {', '.join(names)}"
           + (f"; zero-fraud zones applied (+{np.mean(gains):.6f})" if use_zones else ""))
