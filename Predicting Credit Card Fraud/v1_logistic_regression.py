"""v1 - Baseline: logistic regression on the raw features.

Text columns are one-hot encoded, numbers are standardised.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import CATS, FLAGS, NUMS, load, raw_features, run_cv

train, test, y = load()
X, X_test = raw_features(train, test)
X, X_test = X.astype({c: str for c in CATS}), X_test.astype({c: str for c in CATS})


def fit_predict(X_tr, y_tr, X_va, y_va, X_te):
    pre = ColumnTransformer([("num", StandardScaler(), NUMS + FLAGS),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CATS)])
    model = make_pipeline(pre, LogisticRegression(C=1.0, max_iter=2000))
    model.fit(X_tr, y_tr)
    return model.predict_proba(X_va)[:, 1], model.predict_proba(X_te)[:, 1]


run_cv("v1_logistic_regression", fit_predict, X, X_test, y,
       "Logistic regression, raw features (one-hot + scaling)")
