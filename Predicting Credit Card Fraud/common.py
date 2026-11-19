"""Shared helpers: data loading, features, a fixed 5-fold CV runner, and result logging.

Every iteration script calls run_cv(), which:
  - trains on 4 folds and predicts the 5th (out-of-fold, "OOF") for every fold,
  - averages the 5 fold models' predictions on the test set,
  - saves OOF/test predictions to oof/, the submission to submissions/,
  - appends the CV AUC to results.csv.
"""
import os
import warnings

warnings.filterwarnings("ignore")
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OOF_DIR = os.path.join(HERE, "oof")
SUB_DIR = os.path.join(HERE, "submissions")
RESULTS = os.path.join(HERE, "results.csv")
os.makedirs(OOF_DIR, exist_ok=True)
os.makedirs(SUB_DIR, exist_ok=True)

TARGET = "is_fraud"
SEED = 42
N_FOLDS = 5

CATS = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
FLAGS = ["is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch",
         "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]
NUMS = ["amount_usd", "hours_since_last_txn", "txn_count_last_24h", "distance_from_home_km",
        "card_age_months", "customer_age", "account_balance_usd", "cvv_retry_count", "velocity_score",
        "time_of_day_hour", "day_of_week", "merchant_risk_score", "prior_disputes"]


def load():
    train = pd.read_csv(os.path.join(DATA, "train.csv"))
    test = pd.read_csv(os.path.join(DATA, "test.csv"))
    for df in (train, test):
        df[FLAGS] = df[FLAGS].astype(int)
    y = train[TARGET].astype(int).values
    return train, test, y


def raw_features(train, test):
    """The original 24 columns, with text columns as pandas 'category' (LightGBM/XGBoost/CatBoost read these)."""
    cols = NUMS + FLAGS + CATS
    X, X_test = train[cols].copy(), test[cols].copy()
    for c in CATS:
        cat_type = pd.CategoricalDtype(sorted(train[c].unique()))
        X[c], X_test[c] = X[c].astype(cat_type), X_test[c].astype(cat_type)
    return X, X_test


def engineered_features(train, test):
    """Raw features + new clues (see guide, iteration v3)."""
    X, X_test = raw_features(train, test)
    full = pd.concat([X, X_test], ignore_index=True)
    out = pd.DataFrame(index=full.index)
    out["flag_count"] = full[FLAGS].sum(axis=1)                       # how many red flags at once
    out["log_amount"] = np.log1p(full["amount_usd"])
    out["amount_to_balance"] = full["amount_usd"] / (full["account_balance_usd"] + 1)
    cat_mean_amount = full.groupby("merchant_category", observed=True)["amount_usd"].transform("mean")
    out["amount_vs_category"] = full["amount_usd"] / cat_mean_amount  # unusually big for this shop type?
    out["is_night"] = full["time_of_day_hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    out["risk_x_velocity"] = full["merchant_risk_score"] * full["velocity_score"]
    out["no_auth"] = (full["auth_method"] == "No Authentication").astype(int)
    out["weak_auth_risky"] = out["no_auth"] * out["flag_count"]
    out["txn_per_hour"] = full["txn_count_last_24h"] / (full["hours_since_last_txn"] + 0.1)
    # Frequency encoding: how common is this exact value? (synthetic-data trick)
    for c in ["amount_usd", "velocity_score", "merchant_risk_score", "account_balance_usd"]:
        out[f"{c}_freq"] = full[c].map(full[c].value_counts())
    full = pd.concat([full, out], axis=1)
    n = len(X)
    return full.iloc[:n].reset_index(drop=True), full.iloc[n:].reset_index(drop=True)


def run_cv(name, fit_predict, X, X_test, y, description=""):
    """fit_predict(X_tr, y_tr, X_va, y_va, X_te) -> (pred_va, pred_te)."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_aucs = []
    t0 = time.time()
    for k, (tr, va) in enumerate(skf.split(X, y)):
        p_va, p_te = fit_predict(X.iloc[tr], y[tr], X.iloc[va], y[va], X_test)
        oof[va] = p_va
        test_pred += np.asarray(p_te) / N_FOLDS
        fold_aucs.append(roc_auc_score(y[va], p_va))
        print(f"  fold {k + 1}: AUC {fold_aucs[-1]:.5f}", flush=True)
    auc = roc_auc_score(y, oof)
    secs = time.time() - t0
    print(f"{name}: CV AUC {auc:.5f} (folds {np.mean(fold_aucs):.5f} +/- {np.std(fold_aucs):.5f}) in {secs:.0f}s")
    save_outputs(name, oof, test_pred, X_test.index if "id" not in X_test else None)
    log_result(name, auc, np.std(fold_aucs), secs, description)
    return oof, test_pred


def save_outputs(name, oof, test_pred, _=None):
    np.save(os.path.join(OOF_DIR, f"{name}_oof.npy"), oof)
    np.save(os.path.join(OOF_DIR, f"{name}_test.npy"), test_pred)
    ids = pd.read_csv(os.path.join(DATA, "test.csv"), usecols=["id"])["id"]
    pd.DataFrame({"id": ids, TARGET: test_pred}).to_csv(os.path.join(SUB_DIR, f"{name}.csv"), index=False)


def log_result(name, auc, std, secs, description):
    row = pd.DataFrame([{"iteration": name, "cv_auc": round(auc, 5), "fold_std": round(std, 5),
                         "seconds": round(secs), "description": description}])
    if os.path.exists(RESULTS):
        old = pd.read_csv(RESULTS)
        row = pd.concat([old[old.iteration != name], row], ignore_index=True)
    row.to_csv(RESULTS, index=False)
