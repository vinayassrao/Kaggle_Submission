"""v4 - LightGBM with strong regularisation (best setting from tune_lightgbm.py) on raw features.

Engineered features (v3) did not help, so v4 goes back to the raw 24 columns.
"""
import lightgbm as lgb

from common import SEED, load, raw_features, run_cv

PARAMS = dict(objective="binary", n_estimators=10000, learning_rate=0.02, num_leaves=31, max_depth=4,
              min_child_samples=300, colsample_bytree=0.4, subsample=0.7, subsample_freq=1,
              reg_lambda=10, cat_smooth=50, random_state=SEED, verbose=-1, n_jobs=-1)


def fit_predict(X_tr, y_tr, X_va, y_va, X_te):
    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
              callbacks=[lgb.early_stopping(300, verbose=False)])
    print(f"    best trees: {model.best_iteration_}")
    return model.predict_proba(X_va)[:, 1], model.predict_proba(X_te)[:, 1]


if __name__ == "__main__":
    train, test, y = load()
    X, X_test = raw_features(train, test)
    run_cv("v4_lightgbm_tuned", fit_predict, X, X_test, y,
           "LightGBM, strong regularisation (depth 4, lr 0.02, feature/row sampling), raw features")
