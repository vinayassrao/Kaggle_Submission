"""v16/v17/v18 - "GLM-boost": a dense logistic regression (GLM) on the 73 target-encoded keys gives a
starting score; tiny trees (4-leaf LightGBM or depth-2 XGBoost) then learn only the corrections
(the GLM log-odds are passed as init_score / base_margin).

Usage: python v16_glm_boost.py NAME ENGINE N_FOLDS SEEDS [refit]
  ENGINE: lgb | xgb | xgbgpu (XGBoost on CUDA)
  SEEDS: comma list of teseed:modelseed[:smooth[:variant]]  (smooth default 20; variant for XGBoost: d2 = max_depth 2 (default), lg = lossguide max_leaves 4)
  e.g. python v16_glm_boost.py v16_glm_lgb lgb 5 0:42,1:137,2:2026 refit
Saves: oof/<NAME>_oof.npy (probabilities), oof/<NAME>_test.npy (fold-mean probabilities),
       oof/<NAME>_full.npy (full-data refit probabilities, if refit), oof/<NAME>_glm_oof.npy (GLM alone).
"""
import os
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from common import DATA, OOF_DIR, SUB_DIR, TARGET, load, log_result, raw_features
from fe_keys import N_TREE_KEYS, build_key_store, encode_keys, glm_base_matrix, safe_logit, tree_frame

warnings.filterwarnings("ignore")
NAME, ENGINE, NF = sys.argv[1], sys.argv[2], int(sys.argv[3])
def _parse(spec):
    p = spec.split(":")
    return int(p[0]), int(p[1]), float(p[2]) if len(p) > 2 else 20.0, p[3] if len(p) > 3 else "d2"


SEEDS = [_parse(s) for s in sys.argv[4].split(",")]
REFIT = len(sys.argv) > 5 and sys.argv[5] == "refit"

train, test, y = load()
n = len(train)
raw_tr, raw_te = raw_features(train, test)
raw_all = pd.concat([raw_tr, raw_te], ignore_index=True)
knames, kmat, kcnt, ncat, cents = build_key_store(train, test)
gbase = glm_base_matrix(train, test)
test_rows = np.arange(n, n + len(test))

LGB_P = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=4, min_child_samples=100,
             feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255,
             verbosity=-1, num_threads=-1)
XGB_P = dict(objective="binary:logistic", eval_metric="auc", learning_rate=0.02, max_depth=2, min_child_weight=20,
             subsample=0.8, colsample_bytree=0.5, reg_lambda=2.0, tree_method="hist", enable_categorical=True,
             max_cat_to_onehot=16, n_jobs=-1)


def glm_offsets(a_rows, Ea, Eb, Et, b_rows):
    sc = StandardScaler().fit(safe_logit(Ea))
    Xa = np.hstack([sc.transform(safe_logit(Ea)), gbase[a_rows]])
    m = LogisticRegression(C=0.3, max_iter=300, tol=1e-7, solver="newton-cholesky").fit(Xa, y[a_rows])
    ga = m.decision_function(Xa)
    gt = m.decision_function(np.hstack([sc.transform(safe_logit(Et)), gbase[n:]]))
    gb = m.decision_function(np.hstack([sc.transform(safe_logit(Eb)), gbase[b_rows]])) if Eb is not None else None
    return ga, gb, gt


def boost(Xa, ya, ga, Xb, yb, gb, Xt, gt, mseed, n_rounds=None, variant="d2"):
    """Returns (val_margin or None, test_margin, best_iter)."""
    if ENGINE == "lgb":
        dtr = lgb.Dataset(Xa, ya, init_score=ga)
        p = {**LGB_P, "seed": mseed}
        if n_rounds is None:
            dva = lgb.Dataset(Xb, yb, init_score=gb, reference=dtr)
            bst = lgb.train(p, dtr, 30000, valid_sets=[dva], callbacks=[lgb.early_stopping(300, verbose=False)])
            it = bst.best_iteration
        else:
            bst = lgb.train(p, dtr, n_rounds); it = n_rounds
        vm = bst.predict(Xb, raw_score=True, num_iteration=it) + gb if Xb is not None else None
        return vm, bst.predict(Xt, raw_score=True, num_iteration=it) + gt, it
    xp = dict(XGB_P)
    if variant == "lg":
        xp.update(grow_policy="lossguide", max_depth=0, max_leaves=4)
    if ENGINE == "xgbgpu":
        xp["device"] = "cuda"
    if n_rounds is None:
        m = xgb.XGBClassifier(**xp, n_estimators=30000, early_stopping_rounds=300, random_state=mseed)
        m.fit(Xa, ya, base_margin=ga, eval_set=[(Xb, yb)], base_margin_eval_set=[gb], verbose=False)
        it = m.best_iteration + 1
    else:
        m = xgb.XGBClassifier(**xp, n_estimators=n_rounds, random_state=mseed)
        m.fit(Xa, ya, base_margin=ga, verbose=False); it = n_rounds
    vm = m.predict(Xb, output_margin=True, base_margin=gb, iteration_range=(0, it)) if Xb is not None else None
    return vm, m.predict(Xt, output_margin=True, base_margin=gt, iteration_range=(0, it)), it


folds = list(StratifiedKFold(NF, shuffle=True, random_state=42).split(np.zeros(n), y))
oof = np.zeros(n); glm_oof = np.zeros(n); test_m = np.zeros(len(test)); iters = []; t0 = time.time()
for k, (tr, va) in enumerate(folds):
    vm_sum = np.zeros(len(va)); gb_sum = np.zeros(len(va))
    for teseed, mseed, smooth, variant in SEEDS:
        Ea, Eb, Et = encode_keys(kmat, ncat, y, n, tr, va, smooth=smooth, teseed=teseed)
        ga, gb, gt = glm_offsets(tr, Ea, Eb, Et, va)
        Xa = tree_frame(raw_all, cents, kcnt, knames, tr, Ea[:, :N_TREE_KEYS])
        Xb = tree_frame(raw_all, cents, kcnt, knames, va, Eb[:, :N_TREE_KEYS])
        Xt = tree_frame(raw_all, cents, kcnt, knames, test_rows, Et[:, :N_TREE_KEYS])
        vm, tm, it = boost(Xa, y[tr], ga, Xb, y[va], gb, Xt, gt, mseed, variant=variant)
        vm_sum += vm / len(SEEDS); gb_sum += gb / len(SEEDS); test_m += tm / (len(SEEDS) * NF); iters.append(it)
        print(f"  fold {k + 1} seed ({teseed},{mseed},m{smooth:g},{variant}): GLM {roc_auc_score(y[va], gb):.5f} -> boost {roc_auc_score(y[va], vm):.5f} ({it} trees)", flush=True)
    oof[va] = expit(vm_sum); glm_oof[va] = expit(gb_sum)
    print(f"  fold {k + 1}: AUC {roc_auc_score(y[va], oof[va]):.5f}  (GLM {roc_auc_score(y[va], glm_oof[va]):.5f})  {time.time() - t0:.0f}s", flush=True)
fa = [roc_auc_score(y[va], oof[va]) for _, va in folds]
auc, gauc = roc_auc_score(y, oof), roc_auc_score(y, glm_oof)
print(f"{NAME}: CV AUC {auc:.5f} | GLM alone {gauc:.5f} | folds {np.round(fa, 5)} | mean trees {np.mean(iters):.0f} | {time.time() - t0:.0f}s", flush=True)
np.save(os.path.join(OOF_DIR, f"{NAME}_oof.npy"), oof); np.save(os.path.join(OOF_DIR, f"{NAME}_glm_oof.npy"), glm_oof)
np.save(os.path.join(OOF_DIR, f"{NAME}_test.npy"), expit(test_m))
final_test = expit(test_m)
if REFIT:   # full-data refit: encodings use 100% of train labels; tree count = mean CV iterations x multiplier
    mult = 1.20 if NF == 5 else 1.11
    n_rounds = int(round(np.mean(iters) * mult)); full_m = np.zeros(len(test))
    for teseed, mseed, smooth, variant in SEEDS:
        Ea, _, Et = encode_keys(kmat, ncat, y, n, None, None, smooth=smooth, teseed=teseed)
        ga, _, gt = glm_offsets(np.arange(n), Ea, None, Et, None)
        Xa = tree_frame(raw_all, cents, kcnt, knames, np.arange(n), Ea[:, :N_TREE_KEYS])
        Xt = tree_frame(raw_all, cents, kcnt, knames, test_rows, Et[:, :N_TREE_KEYS])
        _, tm, _ = boost(Xa, y, ga, None, None, None, Xt, gt, mseed, n_rounds=n_rounds, variant=variant)
        full_m += tm / len(SEEDS)
    final_test = expit(full_m); np.save(os.path.join(OOF_DIR, f"{NAME}_full.npy"), final_test)
    print(f"  full refit done ({n_rounds} trees x {len(SEEDS)} seeds)", flush=True)
ids = pd.read_csv(os.path.join(DATA, "test.csv"), usecols=["id"])["id"]
pd.DataFrame({"id": ids, TARGET: final_test}).to_csv(os.path.join(SUB_DIR, f"{NAME}.csv"), index=False)
log_result(NAME, auc, np.std(fa), time.time() - t0,
           f"GLM-boost {ENGINE} {NF}-fold, seeds {sys.argv[4]}, 73 TE keys + digits (GLM alone {gauc:.5f}){', full refit' if REFIT else ''}")
