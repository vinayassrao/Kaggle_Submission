"""v2 - LightGBM (gradient-boosted trees) on the same raw features, near-default settings."""
import lightgbm as lgb

from common import SEED, load, raw_features, run_cv

train, test, y = load()
X, X_test = raw_features(train, test)

PARAMS = dict(objective="binary", learning_rate=0.05, n_estimators=5000, num_leaves=31,
              random_state=SEED, verbose=-1, n_jobs=-1)


def fit_predict(X_tr, y_tr, X_va, y_va, X_te):
    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
              callbacks=[lgb.early_stopping(200, verbose=False)])
    print(f"    best trees: {model.best_iteration_}")
    return model.predict_proba(X_va)[:, 1], model.predict_proba(X_te)[:, 1]


if __name__ == "__main__":
    run_cv("v2_lightgbm", fit_predict, X, X_test, y, "LightGBM, raw features, default-ish settings")
