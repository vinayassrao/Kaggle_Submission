"""v12 - XGBoost on the v9 feature set: max_bin 1024, one-hot splits for the small categoricals."""
import pandas as pd
import xgboost as xgb

from common import SEED, load, raw_features
from fe_v9 import deterministic_features, run_cv_ff

PARAMS = dict(objective="binary:logistic", eval_metric="auc", learning_rate=0.02, max_depth=4, min_child_weight=10,
              subsample=0.7, colsample_bytree=0.4, reg_lambda=10, n_estimators=20000, early_stopping_rounds=400,
              tree_method="hist", max_bin=1024, enable_categorical=True, max_cat_to_onehot=16, random_state=SEED, n_jobs=-1)


def fit_predict(X_tr, y_tr, X_va, y_va, X_te):
    m = xgb.XGBClassifier(**PARAMS)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    print(f"    trees {m.best_iteration}", flush=True)
    return m.predict_proba(X_va)[:, 1], m.predict_proba(X_te)[:, 1]


train, test, y = load()
X0, Xt0 = raw_features(train, test)
D, Dt = deterministic_features(train, test)
run_cv_ff("v12_xgboost_woe", fit_predict, pd.concat([X0, D], axis=1), pd.concat([Xt0, Dt], axis=1), y, train, test,
          woe_kwargs=dict(m=40.0), description="XGBoost + v9 features (max_bin 1024, one-hot categoricals)")
