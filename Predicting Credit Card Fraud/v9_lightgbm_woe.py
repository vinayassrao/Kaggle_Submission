"""v9 - Tuned LightGBM (v4 settings, lr 0.02) + finer histogram (max_bin 1023)
+ in-fold exact-value WoE / residual WoE / memo sum (smoothing m=40) + value frequencies + velocity-zero flag.
Chosen by screen_v9.py (fast lr 0.05 screen): 0.94842 vs 0.94716 base."""
import lightgbm as lgb
import pandas as pd

from common import SEED, load, raw_features
from fe_v9 import deterministic_features, run_cv_ff

PARAMS = dict(objective="binary", metric="auc", n_estimators=20000, learning_rate=0.02, max_depth=4, num_leaves=31,
              min_child_samples=300, colsample_bytree=0.4, subsample=0.7, subsample_freq=1, reg_lambda=10,
              cat_smooth=50, max_bin=1023, random_state=SEED, verbose=-1, n_jobs=-1)


def fit_predict(X_tr, y_tr, X_va, y_va, X_te, params=PARAMS):
    m = lgb.LGBMClassifier(**params)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(400, verbose=False)])
    print(f"    trees {m.best_iteration_}", flush=True)
    return m.predict_proba(X_va)[:, 1], m.predict_proba(X_te)[:, 1]


if __name__ == "__main__":
    train, test, y = load()
    X0, Xt0 = raw_features(train, test)
    D, Dt = deterministic_features(train, test)
    run_cv_ff("v9_lightgbm_woe", fit_predict, pd.concat([X0, D], axis=1), pd.concat([Xt0, Dt], axis=1), y, train, test,
              woe_kwargs=dict(m=40.0), description="Tuned LightGBM + max_bin 1023 + in-fold exact-value WoE (m=40) + value frequencies + velocity-zero")
