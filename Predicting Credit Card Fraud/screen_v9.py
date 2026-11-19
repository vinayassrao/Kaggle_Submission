"""Screen the reviewer's Experiment 2 ideas with the fast LightGBM (lr 0.05), same folds.
Reference: base 0.94716, sklearn TE on all 13 numerics 0.94791 (screen_ideas.py)."""
import sys
import warnings

import lightgbm as lgb
import pandas as pd

from common import SEED, load, raw_features
from fe_v9 import NOISE, deterministic_features, run_cv_ff

warnings.filterwarnings("ignore")
train, test, y = load()
X0, Xt0 = raw_features(train, test)
D, Dt = deterministic_features(train, test)
BASE = dict(objective="binary", metric="auc", n_estimators=10000, learning_rate=0.05, max_depth=4, num_leaves=31,
            min_child_samples=300, colsample_bytree=0.4, subsample=0.7, subsample_freq=1, reg_lambda=10,
            cat_smooth=50, random_state=SEED, verbose=-1, n_jobs=-1)


def lgb_fp(params):
    def fp(X_tr, y_tr, X_va, y_va, X_te):
        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(200, verbose=False)])
        print(f"    trees {m.best_iteration_}", flush=True)
        return m.predict_proba(X_va)[:, 1], m.predict_proba(X_te)[:, 1]
    return fp


def with_det(drop=()):
    X = pd.concat([X0, D], axis=1).drop(columns=list(drop))
    Xt = pd.concat([Xt0, Dt], axis=1).drop(columns=list(drop))
    return X, Xt


V = {
    "maxbin":        lambda: (X0, Xt0, None, {"max_bin": 1023}),
    "det":           lambda: (*with_det(), None, {"max_bin": 1023}),
    "woe":           lambda: (*with_det(), dict(m=15.0), {"max_bin": 1023}),
    "woe_noresid":   lambda: (*with_det(), dict(m=15.0, resid=False), {"max_bin": 1023}),
    "woe_m40_nodet": lambda: (X0, Xt0, dict(m=40.0), {"max_bin": 1023}),
    "woe_m40":       lambda: (*with_det(), dict(m=40.0), {"max_bin": 1023}),
    "woe_dropnoise": lambda: (*with_det(NOISE), dict(m=15.0), {"max_bin": 1023}),
    "woe_pairs":     lambda: (*with_det(), dict(m=15.0, pairs=[("auth_method", "customer_age"), ("merchant_category", "auth_method"),
                                                              ("auth_method", "cvv_retry_count")]), {"max_bin": 1023}),
}
for k in (sys.argv[1:] or V):
    X, Xt, wk, extra = V[k]()
    run_cv_ff(f"s9_{k}", lgb_fp({**BASE, **extra}), X, Xt, y, train, test, woe_kwargs=wk, save=False)
