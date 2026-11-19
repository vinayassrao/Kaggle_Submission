"""v11 - Explainable Boosting Machine (EBM): an additive model of fine-grained 1-D step functions
plus a few pairwise interactions. max_bins=1024 gives most exact values of merchant_risk_score,
velocity_score and customer_age their own bin. Inputs: raw features + label-free features + in-fold WoE."""
import sys

import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier

from common import CATS, load, raw_features
from fe_v9 import deterministic_features, run_cv_ff

train, test, y = load()
X0, Xt0 = raw_features(train, test)
D, Dt = deterministic_features(train, test)
X, Xt = pd.concat([X0, D], axis=1), pd.concat([Xt0, Dt], axis=1)
X[CATS], Xt[CATS] = X[CATS].astype(str), Xt[CATS].astype(str)
BAGS = int(sys.argv[1]) if len(sys.argv) > 1 else 4


def fit_predict(X_tr, y_tr, X_va, y_va, X_te):
    ebm = ExplainableBoostingClassifier(max_bins=1024, max_interaction_bins=64, interactions=10, outer_bags=BAGS,
                                        inner_bags=0, learning_rate=0.015, min_samples_leaf=100,
                                        early_stopping_rounds=100, random_state=42, n_jobs=-1)
    ebm.fit(X_tr, y_tr)
    return ebm.predict_proba(X_va)[:, 1], ebm.predict_proba(X_te)[:, 1]


run_cv_ff("v11_ebm", fit_predict, X, Xt, y, train, test, woe_kwargs=dict(m=40.0),
          description=f"Explainable Boosting Machine (max_bins 1024, 10 interactions, {BAGS} bags) + WoE features")
