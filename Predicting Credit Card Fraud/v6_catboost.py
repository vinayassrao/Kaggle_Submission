"""v6 - CatBoost: boosting that handles text categories natively with ordered target statistics."""
import sys

from catboost import CatBoostClassifier

from common import CATS, SEED, engineered_features, load, raw_features, run_cv

FEATURES = sys.argv[1] if len(sys.argv) > 1 else "raw"
train, test, y = load()
X, X_test = (engineered_features if FEATURES == "eng" else raw_features)(train, test)
X, X_test = X.astype({c: str for c in CATS}), X_test.astype({c: str for c in CATS})

PARAMS = dict(iterations=10000, learning_rate=0.05, depth=6, l2_leaf_reg=10, eval_metric="AUC",
              od_type="Iter", od_wait=300, random_seed=SEED, thread_count=-1, verbose=False)


def fit_predict(X_tr, y_tr, X_va, y_va, X_te):
    model = CatBoostClassifier(**PARAMS, cat_features=CATS)
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
    print(f"    best trees: {model.get_best_iteration()}")
    return model.predict_proba(X_va)[:, 1], model.predict_proba(X_te)[:, 1]


run_cv("v6_catboost", fit_predict, X, X_test, y, f"CatBoost, native categories, {FEATURES} features")
