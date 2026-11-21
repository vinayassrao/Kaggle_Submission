"""v32 - v31 pool + v25 GPT-2 (Kaggle 2x T4 run). Same stacker code as v31; the fold check now counts folds over
all 3 stacker splits (15 folds, need >= 12 up) instead of split 43 only (5 folds).

Original v31 description: Phase 4 stack with a strict marginal-gain gate and a non-negative stacker (P4.5).

1. Base pool = the 10 v30 members. Candidates = v28_glm_only, v25_gpt2, v26b_realmlp_keys, v27b_digit_cnn_seg (if on disk).
2. Gate: each candidate is added to the base pool alone; mean nested AUC over stacker splits 43/44/45 must rise by
   >= +0.00004 and the per-fold AUC (split 43) must rise in >= 4/5 folds. Passing candidates are then added together.
3. Final pool compared with: unconstrained LogReg (C grid) vs non-negative L2 logistic regression (L-BFGS-B, weights >= 0).
   Prefer the non-negative one if within 0.00002 of the best.
4. Zero-fraud zones verified leak-free on the seed-42 folds, then applied.
Outputs: submissions/v32_stack.csv (chosen), submissions/v32_conservative.csv (non-negative weights, test inputs from
5-fold means instead of full refits).
"""
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import DATA, OOF_DIR, SUB_DIR, TARGET, load, log_result
from fe_keys import rank_logit, safe_logit

BASE = ["v17_glm_lgb10", "v17b_glm_lgb10_s100", "v28_glm_xgb10_gpu", "v19_sparse_glm_10f", "v16_glm_lgb",
        "v26_realmlp", "v27_digit_cnn", "v11_ebm", "v22_lgb4_raw", "v10_spline_logreg", "v28_glm_only"]   # = final v31 pool
CANDS = ["v25_gpt2"]
NATIVE10 = ("lgb10", "xgb10", "_10f", "glm_only")
train, test, y = load()
n = len(y)
ex = lambda m: os.path.exists(os.path.join(OOF_DIR, f"{m}_oof.npy"))
CANDS = [c for c in CANDS if ex(c)]
r01 = lambda v: (rankdata(v) - 1) / (len(v) - 1)


def load_member(m, prefer_full=True):
    o = np.load(os.path.join(OOF_DIR, f"{m}_oof.npy"))
    k = 10 if any(t in m for t in NATIVE10) else 5
    z = safe_logit(o); zs = np.zeros_like(z)
    for _, va in StratifiedKFold(k, shuffle=True, random_state=42).split(np.zeros(n), y):
        zs[va] = (z[va] - z[va].mean()) / (z[va].std() + 1e-12)
    order = ("_full.npy", "_test.npy") if prefer_full else ("_test.npy", "_full.npy")
    for suf in order:
        p = os.path.join(OOF_DIR, m + suf)
        if os.path.exists(p):
            return rank_logit(zs), rank_logit(np.load(p)), suf
    raise FileNotFoundError(m)


MEM = {m: load_member(m) for m in BASE + CANDS}
SPLITS = [list(StratifiedKFold(5, shuffle=True, random_state=s).split(np.zeros(n), y)) for s in (43, 44, 45)]


def nn_logreg(Z, yy, lam):
    """Non-negative L2 logistic regression: weights >= 0, free intercept."""
    M = Z.shape[1]

    def f(w):
        s = Z @ w[:M] + w[M]
        loss = np.mean(np.logaddexp(0, s) - yy * s) + lam * np.sum(w[:M] ** 2)
        p = expit(s); g = np.r_[Z.T @ (p - yy) / len(yy) + 2 * lam * w[:M], np.mean(p - yy)]
        return loss, g
    w0 = np.r_[np.full(M, 1.0 / M), -4.0]
    res = minimize(f, w0, jac=True, method="L-BFGS-B", bounds=[(0, None)] * M + [(None, None)])
    return res.x


def combiners(Ztr, ytr, Zev):
    out = {}
    for C in (0.005, 0.01, 0.03, 0.05):
        out[f"lr_C{C}"] = LogisticRegression(C=C, solver="newton-cholesky", max_iter=200).fit(Ztr, ytr).decision_function(Zev)
    for lam in (1e-4, 1e-3):
        w = nn_logreg(Ztr, ytr, lam); out[f"nnlr_l{lam:g}"] = Zev @ w[:-1] + w[-1]
    return out


def nested(names, only=None):
    """Mean nested AUC per combiner over splits 43/44/45, plus split-43 per-fold AUCs."""
    Z = np.column_stack([MEM[m][0] for m in names])
    res, fold43 = {}, {}
    for si, sp in enumerate(SPLITS):
        pred = {}
        for tr, va in sp:
            for k, v in combiners(Z[tr], y[tr], Z[va]).items():
                if only and k != only:
                    continue
                pred.setdefault(k, np.zeros(n))[va] = r01(v)
        for k, v in pred.items():
            res.setdefault(k, []).append(roc_auc_score(y, v))
            fold43.setdefault(k, []).extend(roc_auc_score(y[va], v[va]) for _, va in sp)
    return {k: float(np.mean(v)) for k, v in res.items()}, {k: np.array(v) for k, v in fold43.items()}, res


print("candidates on disk:", CANDS)
base_scores, base_f43, _ = nested(BASE)
REF = "lr_C0.05"
print(f"base pool (v31 members) nested AUC [{REF}]: {base_scores[REF]:.5f}")
kept = []
for c in CANDS:
    sc, f43, _ = nested(BASE + [c], only=REF)
    gain = sc[REF] - base_scores[REF]; up = int((f43[REF] > base_f43[REF]).sum())
    ok = gain >= 4e-5 and up >= 12
    print(f"  + {c:22s} nested {sc[REF]:.5f}  gain {gain:+.5f}  folds up {up}/15  -> {'KEEP' if ok else 'drop'}")
    if ok:
        kept.append(c)
POOL = BASE + kept
final_scores, _, raw = nested(POOL)
print("\nfinal pool:", POOL)
for k, v in final_scores.items():
    print(f"  {k:12s} {v:.5f}  {np.round(raw[k], 5)}")
best = max(final_scores, key=final_scores.get)
nn_best = max((k for k in final_scores if k.startswith("nnlr")), key=final_scores.get)
chosen = nn_best if final_scores[best] - final_scores[nn_best] <= 2e-5 else best
print(f"best {best} {final_scores[best]:.5f} | best non-negative {nn_best} {final_scores[nn_best]:.5f} -> chosen {chosen}")

Z = np.column_stack([MEM[m][0] for m in POOL]); T = np.column_stack([MEM[m][1] for m in POOL])
full_out = combiners(Z, y, T)
lam = float(nn_best.split("_l")[1]); w = nn_logreg(Z, y, lam)
print("non-negative weights:", dict(zip(POOL, np.round(w[:-1], 3))))
lr = LogisticRegression(C=0.05, solver="newton-cholesky", max_iter=200).fit(Z, y)
print("LogReg C=0.05 coefs:", dict(zip(POOL, np.round(lr.coef_[0], 3))))

# zero-fraud zones, verified on the seed-42 folds with the chosen combiner's nested OOF (split 43)
_, _, _ = None, None, None
oof43 = np.zeros(n)
for tr, va in SPLITS[0]:
    oof43[va] = r01(combiners(Z[tr], y[tr], Z[va])[chosen])
mrs, vel = train["merchant_risk_score"].to_numpy(), train["velocity_score"].to_numpy()
gains = []
for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n), y):
    thr = mrs[tr][y[tr] == 1].min()
    zmask = (mrs[va] < thr) | ((vel[va] <= 4.1) & (mrs[va] <= 20.0))
    adj = oof43[va].copy(); adj[zmask] -= 10
    gains.append(roc_auc_score(y[va], adj) - roc_auc_score(y[va], oof43[va]))
use_z = all(g > 0 for g in gains)
print(f"zones: gains {np.round(gains, 6)} -> {'APPLY' if use_z else 'skip'}")
thr = mrs[y == 1].min(); tm, tv = test["merchant_risk_score"].to_numpy(), test["velocity_score"].to_numpy()
zt = (tm < thr) | ((tv <= 4.1) & (tm <= 20.0))


def finish(score):
    s = r01(score)
    if use_z:
        s = s.copy(); s[zt] -= 10; s = r01(s)
    return s


ids = pd.read_csv(os.path.join(DATA, "test.csv"), usecols=["id"])["id"]
main = finish(full_out[chosen])
pd.DataFrame({"id": ids, TARGET: main}).to_csv(os.path.join(SUB_DIR, "v32_stack.csv"), index=False)
np.save(os.path.join(OOF_DIR, "v32_stack_oof.npy"), oof43); np.save(os.path.join(OOF_DIR, "v32_stack_test.npy"), main)
# conservative: non-negative weights, 5-fold-mean test inputs instead of full refits
Tc = np.column_stack([load_member(m, prefer_full=False)[1] for m in POOL])
cons = finish(Tc @ w[:-1] + w[-1])
pd.DataFrame({"id": ids, TARGET: cons}).to_csv(os.path.join(SUB_DIR, "v32_conservative.csv"), index=False)
print(f"Spearman(main, conservative) on test: {pd.Series(main).corr(pd.Series(cons), method='spearman'):.5f}")
log_result("v32_stack", final_scores[chosen], float(np.std(raw[chosen])), 0,
           f"{chosen} on {len(POOL)} members (kept: {', '.join(kept) or 'none'}); zones {'on' if use_z else 'off'}")
print(f"v32_stack: {chosen} nested {final_scores[chosen]:.5f}")
