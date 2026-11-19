"""Diagnostics for the Appendix A target-encoding gain.

1. Re-runs the TE model (fast LightGBM, same folds) and SAVES its OOF + per-fold info.
2. Gain % and permutation AUC drop of every TE column (fold-1 model, on fold-1 validation).
3. Leave-one-out: drop each TE column in turn and re-run the 5-fold CV.
4. Group runs: TE on high-signal numerics only / noise numerics only.
Writes diag_te_results.json.
"""
import json
import os
import warnings

import lightgbm as lgb
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

from common import HERE, NUMS, OOF_DIR, SEED, load, raw_features

warnings.filterwarnings("ignore")
train, test, y = load()
X, _ = raw_features(train, test)
PARAMS = dict(objective="binary", metric="auc", n_estimators=10000, learning_rate=0.05, max_depth=4, num_leaves=31,
              min_child_samples=300, colsample_bytree=0.4, subsample=0.7, subsample_freq=1, reg_lambda=10,
              cat_smooth=50, random_state=SEED, verbose=-1, n_jobs=-1)
folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
OUT = os.path.join(HERE, "diag_te_results.json")
res = json.load(open(OUT)) if os.path.exists(OUT) else {}


def te_frames(tr, va, cols):
    Xtr, Xva = X.iloc[tr].copy(), X.iloc[va].copy()
    if cols:
        te = TargetEncoder(target_type="binary", cv=5, shuffle=True, random_state=SEED)  # smooth="auto"
        names = [f"{c}_te" for c in cols]
        Xtr[names] = te.fit_transform(Xtr[cols].astype(str), y[tr])   # cross-fitted inside the training fold
        Xva[names] = te.transform(Xva[cols].astype(str))              # full-training-fold encoding
    return Xtr, Xva


def cv(cols, save_name=None, keep_fold1=False):
    oof = np.zeros(len(X)); iters = []; fold1 = None
    for k, (tr, va) in enumerate(folds):
        Xtr, Xva = te_frames(tr, va, cols)
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(Xtr, y[tr], eval_set=[(Xva, y[va])], callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va] = m.predict_proba(Xva)[:, 1]; iters.append(int(m.best_iteration_))
        if k == 0 and keep_fold1:
            fold1 = (m, Xva, y[va])
    fa = [roc_auc_score(y[va], oof[va]) for _, va in folds]
    if save_name:
        np.save(os.path.join(OOF_DIR, f"{save_name}_oof.npy"), oof)
    return roc_auc_score(y, oof), fa, iters, fold1


if "full" not in res:
    auc, fa, iters, (m, Xva, yva) = cv(NUMS, save_name="screen_te_nums", keep_fold1=True)
    gain = m.booster_.feature_importance("gain"); gain = dict(zip(Xva.columns, (gain / gain.sum() * 100).round(3).tolist()))
    pi = permutation_importance(m, Xva, yva, scoring="roc_auc", n_repeats=3, random_state=SEED, n_jobs=1)
    perm = dict(zip(Xva.columns, pi.importances_mean.round(5).tolist()))
    res["full"] = dict(auc=auc, folds=fa, iters=iters, gain=gain, perm=perm)
    print("full", auc, flush=True); json.dump(res, open(OUT, "w"), indent=1)
if "base" not in res:
    auc, fa, iters, _ = cv([], save_name="screen_base")
    res["base"] = dict(auc=auc, folds=fa, iters=iters); print("base", auc, flush=True); json.dump(res, open(OUT, "w"), indent=1)

groups = {"high_signal_only": ["merchant_risk_score", "velocity_score", "amount_usd", "hours_since_last_txn"],
          "noise_only": ["account_balance_usd", "distance_from_home_km", "card_age_months"]}
for g, cols in groups.items():
    if g not in res:
        auc, fa, iters, _ = cv(cols); res[g] = dict(auc=auc, folds=fa, cols=cols)
        print(g, auc, flush=True); json.dump(res, open(OUT, "w"), indent=1)
for c in NUMS:
    k = f"drop_{c}"
    if k not in res:
        auc, fa, iters, _ = cv([n for n in NUMS if n != c]); res[k] = dict(auc=auc, folds=fa)
        print(k, auc, flush=True); json.dump(res, open(OUT, "w"), indent=1)
print("DONE")
