"""v13 - CatBoost on the v9 feature set, better regularised than v6 (depth 4, rsm 0.4, Bernoulli subsample),
plus hours_since_last_txn and customer_age as extra categorical features (CatBoost ordered target statistics)."""
import pandas as pd
from catboost import CatBoostClassifier

from common import CATS, SEED, load, raw_features
from fe_v9 import deterministic_features, run_cv_ff

EXTRA_CATS = ["hours_since_last_txn_str", "customer_age_str"]
PARAMS = dict(iterations=20000, learning_rate=0.03, depth=4, l2_leaf_reg=15, rsm=0.4, bootstrap_type="Bernoulli",
              subsample=0.75, border_count=254, max_ctr_complexity=1, eval_metric="AUC", od_type="Iter", od_wait=300,
              random_seed=SEED, thread_count=-1, verbose=False)


def fit_predict(X_tr, y_tr, X_va, y_va, X_te):
    m = CatBoostClassifier(**PARAMS, cat_features=CATS + EXTRA_CATS)
    m.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
    print(f"    trees {m.get_best_iteration()}", flush=True)
    return m.predict_proba(X_va)[:, 1], m.predict_proba(X_te)[:, 1]


train, test, y = load()
X0, Xt0 = raw_features(train, test)
D, Dt = deterministic_features(train, test)
X, Xt = pd.concat([X0, D], axis=1), pd.concat([Xt0, Dt], axis=1)
for df, src in [(X, train), (Xt, test)]:
    df[CATS] = df[CATS].astype(str)
    df["hours_since_last_txn_str"] = src["hours_since_last_txn"].astype(str).values
    df["customer_age_str"] = src["customer_age"].astype(str).values
run_cv_ff("v13_catboost_woe", fit_predict, X, Xt, y, train, test, woe_kwargs=dict(m=40.0),
          description="CatBoost depth 4, rsm 0.4 + v9 features + exact hours/age as categorical")
