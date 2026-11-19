"""v5 - XGBoost, regularised like v4, on the same features as v4."""
import sys

import xgboost as xgb

from common import SEED, engineered_features, load, raw_features, run_cv

FEATURES = sys.argv[1] if len(sys.argv) > 1 else "raw"
train, test, y = load()
X, X_test = (engineered_features if FEATURES == "eng" else raw_features)(train, test)

PARAMS = dict(objective="binary:logistic", eval_metric="auc", learning_rate=0.02, max_depth=4,
              min_child_weight=10, subsample=0.7, colsample_bytree=0.5, reg_lambda=10,
              n_estimators=10000, early_stopping_rounds=300, tree_method="hist",
              enable_categorical=True, max_cat_to_onehot=1, random_state=SEED, n_jobs=-1)


def fit_predict(X_tr, y_tr, X_va, y_va, X_te):
    model = xgb.XGBClassifier(**PARAMS)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    print(f"    best trees: {model.best_iteration}")
    return model.predict_proba(X_va)[:, 1], model.predict_proba(X_te)[:, 1]


run_cv("v5_xgboost", fit_predict, X, X_test, y, f"XGBoost, regularised, {FEATURES} features")
