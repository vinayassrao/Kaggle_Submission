"""v8 - Do imbalance techniques help? Class weights vs undersampling vs SMOTE, on the v4 LightGBM.

Every resampling is done INSIDE each fold, on the training part only (the validation
fold keeps the real 1.67% fraud rate), otherwise the scores would be leaked.
Early stopping watches AUC only: with class weights the (unweighted) log-loss gets worse
from the first trees, and LightGBM's default would stop training almost immediately.
Reports ROC AUC (the competition metric) and PR-AUC (average precision) for each.
"""
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import CATS, FLAGS, N_FOLDS, OOF_DIR, SEED, load, log_result, raw_features, save_outputs
from v4_lightgbm_tuned import PARAMS

train, test, y = load()
X, X_test = raw_features(train, test)
NEG_POS = (y == 0).sum() / (y == 1).sum()   # ~58.9 normal transactions per fraud


def undersample(X_tr, y_tr, ratio=10, seed=SEED):
    """Keep every fraud, keep only `ratio` normal transactions per fraud."""
    rng = np.random.default_rng(seed)
    pos = np.where(y_tr == 1)[0]
    neg = rng.choice(np.where(y_tr == 0)[0], size=len(pos) * ratio, replace=False)
    idx = np.concatenate([pos, neg])
    return X_tr.iloc[idx], y_tr[idx]


def smote(X_tr, y_tr, target_share=0.10):
    """Create synthetic frauds until frauds are `target_share` of the normal count (SMOTE-NC handles categories)."""
    Xn = X_tr.copy()
    for c in CATS:
        Xn[c] = Xn[c].cat.codes
    cat_idx = [Xn.columns.get_loc(c) for c in CATS + FLAGS]
    sm = SMOTENC(categorical_features=cat_idx, sampling_strategy=target_share, k_neighbors=5, random_state=SEED)
    Xr, yr = sm.fit_resample(Xn, y_tr)
    Xr = pd.DataFrame(Xr, columns=X_tr.columns)
    for c in CATS:
        Xr[c] = pd.Categorical.from_codes(Xr[c].astype(int), categories=X_tr[c].cat.categories)
    for c in X_tr.columns:
        if c not in CATS:
            Xr[c] = Xr[c].astype(X_tr[c].dtype)
    return Xr, np.asarray(yr)


METHODS = {
    "none": ("No imbalance handling (same settings, AUC-only early stopping)", {}, None),
    "class_weights": ("Class weights: each fraud counts ~59x in training", dict(scale_pos_weight=NEG_POS), None),
    "class_weights_sqrt": ("Softer class weights: each fraud counts ~7.7x", dict(scale_pos_weight=np.sqrt(NEG_POS)), None),
    "undersample": ("Random undersampling to 1 fraud : 10 normal", {}, undersample),
    "smote": ("SMOTE-NC: synthetic frauds up to 10% of normals", {}, smote),
}

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = list(skf.split(X, y))
rows = []
base = np.load(os.path.join(OOF_DIR, "v4_lightgbm_tuned_oof.npy"))
rows.append(("v4 (no imbalance handling)", roc_auc_score(y, base), average_precision_score(y, base)))
print(f"v4 baseline: AUC {rows[0][1]:.5f}  PR-AUC {rows[0][2]:.5f}", flush=True)

only = sys.argv[1:] or list(METHODS)
for key in only:
    desc, extra, resampler = METHODS[key]
    oof, test_pred = np.zeros(len(X)), np.zeros(len(X_test))
    for k, (tr, va) in enumerate(folds):
        X_tr, y_tr = X.iloc[tr], y[tr]
        if resampler is not None:
            X_tr, y_tr = resampler(X_tr, y_tr)
        model = lgb.LGBMClassifier(**{**PARAMS, "metric": "auc", **extra})  # stop on AUC only
        model.fit(X_tr, y_tr, eval_set=[(X.iloc[va], y[va])], eval_metric="auc",
                  callbacks=[lgb.early_stopping(300, verbose=False)])
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / N_FOLDS
        print(f"  {key} fold {k + 1}: AUC {roc_auc_score(y[va], oof[va]):.5f}  rows trained {len(y_tr):,}  trees {model.best_iteration_}", flush=True)
    auc, ap = roc_auc_score(y, oof), average_precision_score(y, oof)
    print(f"{key}: AUC {auc:.5f}  PR-AUC {ap:.5f}", flush=True)
    name = f"v8_{key}"
    save_outputs(name, oof, test_pred)
    log_result(name, auc, np.nan, 0, desc)
    rows.append((desc, auc, ap))

out = pd.DataFrame(rows, columns=["method", "roc_auc", "pr_auc"])
out.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "v8_imbalance_results.csv"), index=False)
print(out.to_string(index=False))
