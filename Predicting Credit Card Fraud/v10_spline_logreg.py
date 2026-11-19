"""v10 - Upgraded scorecard: spline logistic regression (a GAM-like additive model).

Fixes every gap the diagnostics found between v1 and the GBDTs:
  - splines (smooth bendy curves) for the main numbers instead of straight lines
  - exact values of small-integer columns as categories (customer_age jump at 34, cvv 2+)
  - is_velocity_zero point mass
  - in-fold exact-value WoE features + value frequencies
  - auth_method x [is_velocity_zero, merchant_risk_score, cvv_retry_count, flag_count] interactions
"""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from common import CATS, FLAGS, load, raw_features
from fe_v9 import deterministic_features, run_cv_ff

train, test, y = load()


def base_frame(df, det):
    out = pd.DataFrame({
        "merchant_risk_score": df["merchant_risk_score"].values, "velocity_score": df["velocity_score"].values,
        "log_hours": np.log1p(df["hours_since_last_txn"].values), "log_amount": np.log1p(df["amount_usd"].values),
        "log_balance": np.log1p(df["account_balance_usd"].values), "distance_from_home_km": df["distance_from_home_km"].values,
    })
    for c in ["customer_age", "cvv_retry_count", "txn_count_last_24h", "prior_disputes"] + CATS:
        out[c] = df[c].astype(str).values
    for c in FLAGS:
        out[c] = df[c].values
    out["flag_count"] = df[FLAGS].sum(axis=1).values
    return pd.concat([out, det.reset_index(drop=True)], axis=1)


D, Dt = deterministic_features(train, test)
X, Xt = base_frame(train, D), base_frame(test, Dt)
SPL = ["merchant_risk_score", "velocity_score", "log_hours", "log_amount", "log_balance", "distance_from_home_km"]
EXACT = ["customer_age", "cvv_retry_count", "txn_count_last_24h", "prior_disputes"] + CATS


def fit_predict(X_tr, y_tr, X_va, y_va, X_te):
    woe_cols = [c for c in X_tr.columns if c.endswith("woe") or c == "woe_memo_sum"]
    scale = FLAGS + ["flag_count", "is_velocity_zero"] + [c for c in X_tr.columns if c.endswith("_logfreq")] + woe_cols
    pre = ColumnTransformer([
        ("spl", SplineTransformer(n_knots=12, degree=3, knots="quantile", include_bias=False), SPL),
        ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=10), EXACT),
        ("sc", StandardScaler(), scale),
    ], sparse_threshold=1.0)

    def inter(df):   # auth_method dummies x key risk drivers
        auth = pd.get_dummies(df["auth_method"]).reindex(columns=sorted(train["auth_method"].unique()), fill_value=0).values.astype(float)
        drivers = np.column_stack([df["is_velocity_zero"], df["merchant_risk_score"] / 100, df["cvv_retry_count"].astype(float), df["flag_count"]])
        return np.column_stack([auth[:, i] * drivers[:, j] for i in range(auth.shape[1]) for j in range(drivers.shape[1])])

    A_tr = pre.fit_transform(X_tr)
    sc = StandardScaler().fit(inter(X_tr))
    mk = lambda A, df: sparse.hstack([sparse.csr_matrix(A), sparse.csr_matrix(sc.transform(inter(df)))]).tocsr()
    M_tr, M_va, M_te = mk(A_tr, X_tr), mk(pre.transform(X_va), X_va), mk(pre.transform(X_te), X_te)
    clf = LogisticRegression(C=0.2, max_iter=3000, solver="lbfgs")
    clf.fit(M_tr, y_tr)
    return clf.predict_proba(M_va)[:, 1], clf.predict_proba(M_te)[:, 1]


if __name__ == "__main__":
    run_cv_ff("v10_spline_logreg", fit_predict, X, Xt, y, train, test, woe_kwargs=dict(m=40.0),
              description="Spline logistic regression + exact-value categories + velocity-zero + in-fold WoE + auth interactions")
