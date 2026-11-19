"""Quick screening of improvement ideas with a faster LightGBM (lr 0.05), same 5 folds.

Ideas:
  base      : raw features (reference for this screen)
  te_nums   : + in-fold target encoding of every numeric column's exact value (Playground trick)
  cat_nums  : low-cardinality numbers treated as categories
  both      : te_nums + cat_nums
Target encoding uses sklearn's TargetEncoder, which cross-fits inside the training fold
(each training row's encoding comes from other rows), so there is no leakage.
"""
import sys
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

from common import NUMS, SEED, load, raw_features

warnings.filterwarnings("ignore")
train, test, y = load()
X, _ = raw_features(train, test)
LOWCARD = ["txn_count_last_24h", "cvv_retry_count", "prior_disputes", "card_age_months", "customer_age",
           "time_of_day_hour", "day_of_week"]
PARAMS = dict(objective="binary", metric="auc", n_estimators=10000, learning_rate=0.05, max_depth=4, num_leaves=31,
              min_child_samples=300, colsample_bytree=0.4, subsample=0.7, subsample_freq=1, reg_lambda=10,
              cat_smooth=50, random_state=SEED, verbose=-1, n_jobs=-1)
folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))


def run(name, use_te=False, use_cat=False):
    oof = np.zeros(len(X))
    for tr, va in folds:
        Xtr, Xva = X.iloc[tr].copy(), X.iloc[va].copy()
        if use_te:
            te = TargetEncoder(target_type="binary", cv=5, shuffle=True, random_state=SEED)
            cols = [f"te_{c}" for c in NUMS]
            Xtr[cols] = te.fit_transform(Xtr[NUMS].astype(str), y[tr])
            Xva[cols] = te.transform(Xva[NUMS].astype(str))
        if use_cat:
            for c in LOWCARD:
                cat = pd.CategoricalDtype(sorted(X[c].unique()))
                Xtr[f"cat_{c}"] = Xtr[c].astype(cat); Xva[f"cat_{c}"] = Xva[c].astype(cat)
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(Xtr, y[tr], eval_set=[(Xva, y[va])], callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va] = m.predict_proba(Xva)[:, 1]
    fa = [roc_auc_score(y[va], oof[va]) for _, va in folds]
    print(f"{name:10s} AUC {roc_auc_score(y, oof):.5f}  folds {np.round(fa, 5)}", flush=True)
    return oof


ideas = {"base": {}, "te_nums": dict(use_te=True), "cat_nums": dict(use_cat=True), "both": dict(use_te=True, use_cat=True)}
for k in (sys.argv[1:] or ideas):
    run(k, **ideas[k])
