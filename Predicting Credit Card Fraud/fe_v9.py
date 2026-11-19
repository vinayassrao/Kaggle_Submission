"""Feature engineering + CV runner for the v9+ experiments (based on the reviewer's improvement plan).

Label-free features (computed once on train+test):
  <col>_logfreq     log(1 + how often this exact value occurs in train+test)
  is_velocity_zero  velocity_score == 0 (zero-inflated point mass, 0.04% fraud)

Label-based features (computed INSIDE each CV fold, leak-free):
  <col>_woe         smoothed exact-value Weight of Evidence (log-odds vs prior).
                    Training rows get inner-5-fold out-of-fold values; validation/test rows
                    get the value fitted on the whole training fold.
  <col>_resid_woe   exact-value WoE minus coarse-bin WoE (only value-level spikes remain)
  woe_memo_sum      sum of the sparse float columns' WoE (naive-Bayes style score)

run_cv_ff(): like common.run_cv but the model function also receives fold-specific features,
reports concatenated OOF AUC (our standard) and mean fold AUC, and averages the 5 fold test
predictions in standardised log-odds space (removes fold-to-fold scale drift).
"""
import os
import time

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import DATA, N_FOLDS, OOF_DIR, SEED, SUB_DIR, TARGET, log_result

HIGH_CARD = ["hours_since_last_txn", "customer_age", "account_balance_usd", "amount_usd", "distance_from_home_km"]
RESID = ["merchant_risk_score", "velocity_score"]
MEMO = ["account_balance_usd_woe", "amount_usd_woe", "distance_from_home_km_woe", "hours_since_last_txn_woe",
        "merchant_risk_score_resid_woe", "velocity_score_resid_woe"]
FREQ_COLS = ["account_balance_usd", "amount_usd", "hours_since_last_txn", "merchant_risk_score", "velocity_score"]
NOISE = ["channel", "device_type", "card_type", "time_of_day_hour", "day_of_week", "card_age_months"]


def to_std_logit(p, eps=1e-6):
    z = logit(np.clip(p, eps, 1 - eps))
    return (z - z.mean()) / (z.std() + 1e-12)


def deterministic_features(train, test):
    full = pd.concat([train, test], ignore_index=True)
    out = pd.DataFrame(index=full.index)
    for c in FREQ_COLS:
        out[f"{c}_logfreq"] = np.log1p(full[c].map(full[c].value_counts())).astype(np.float32)
    out["is_velocity_zero"] = (full["velocity_score"] == 0).astype(np.int8)
    n = len(train)
    return out.iloc[:n].reset_index(drop=True), out.iloc[n:].reset_index(drop=True)


def _woe_map(keys, y, m):
    prior = y.mean()
    st = pd.DataFrame({"k": keys, "y": y}).groupby("k", sort=False)["y"].agg(["count", "sum"])
    p = ((st["sum"] + m * prior) / (st["count"] + m)).clip(1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)) - np.log(prior / (1 - prior))


def _encode(keys_tr, y_tr, keys_other, inner, m):
    """OOF encoding for training rows, full-fit encoding for each array in keys_other."""
    oof = np.zeros(len(keys_tr), dtype=np.float32)
    for itr, iva in inner:
        wm = _woe_map(keys_tr[itr], y_tr[itr], m)
        oof[iva] = pd.Series(keys_tr[iva]).map(wm).fillna(0.0).to_numpy(np.float32)
    wm = _woe_map(keys_tr, y_tr, m)
    return oof, [pd.Series(k).map(wm).fillna(0.0).to_numpy(np.float32) for k in keys_other]


def fold_woe_features(raw_tr, y_tr, raw_others, m=15.0, m_coarse=50.0, resid=True, pairs=None, seed=SEED):
    """raw_tr / raw_others: DataFrames with the original columns. Returns (feat_tr, [feat_other...])."""
    inner = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y_tr)), y_tr))
    f_tr = pd.DataFrame(index=range(len(raw_tr)))
    f_ot = [pd.DataFrame(index=range(len(r))) for r in raw_others]
    key = lambda df, c: df[c].astype(str).to_numpy()
    for c in HIGH_CARD + RESID:
        oof, others = _encode(key(raw_tr, c), y_tr, [key(r, c) for r in raw_others], inner, m)
        if c in RESID and resid:
            ck = lambda df: np.floor(df[c].to_numpy() / 2.0).astype(int).astype(str)
            coof, cothers = _encode(ck(raw_tr), y_tr, [ck(r) for r in raw_others], inner, m_coarse)
            f_tr[f"{c}_resid_woe"] = oof - coof
            for f, o, co in zip(f_ot, others, cothers):
                f[f"{c}_resid_woe"] = o - co
        else:
            name = f"{c}_resid_woe" if c in RESID else f"{c}_woe"   # keep MEMO names stable
            f_tr[name] = oof
            for f, o in zip(f_ot, others):
                f[name] = o
    for a, b in (pairs or []):
        pk = lambda df: (df[a].astype(str) + "_" + df[b].astype(str)).to_numpy()
        oof, others = _encode(pk(raw_tr), y_tr, [pk(r) for r in raw_others], inner, m)
        f_tr[f"{a}_x_{b}_woe"] = oof
        for f, o in zip(f_ot, others):
            f[f"{a}_x_{b}_woe"] = o
    f_tr["woe_memo_sum"] = f_tr[MEMO].sum(axis=1)
    for f in f_ot:
        f["woe_memo_sum"] = f[MEMO].sum(axis=1)
    return f_tr, f_ot


def run_cv_ff(name, fit_predict, X, X_test, y, raw_train, raw_test, woe_kwargs=None, description="", save=True):
    """fit_predict(X_tr, y_tr, X_va, y_va, X_te) -> (p_va, p_te). Fold WoE features are appended when woe_kwargs is not None."""
    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y))
    oof = np.zeros(len(X)); test_z = np.zeros(len(X_test)); fa = []; t0 = time.time()
    for k, (tr, va) in enumerate(folds):
        X_tr, X_va, X_te = X.iloc[tr].reset_index(drop=True), X.iloc[va].reset_index(drop=True), X_test.reset_index(drop=True)
        if woe_kwargs is not None:
            f_tr, (f_va, f_te) = fold_woe_features(raw_train.iloc[tr].reset_index(drop=True), y[tr],
                                                   [raw_train.iloc[va].reset_index(drop=True), raw_test], **woe_kwargs)
            X_tr, X_va, X_te = (pd.concat([a, b], axis=1) for a, b in [(X_tr, f_tr), (X_va, f_va), (X_te, f_te)])
        p_va, p_te = fit_predict(X_tr, y[tr], X_va, y[va], X_te)
        oof[va] = p_va; test_z += to_std_logit(np.asarray(p_te)) / N_FOLDS
        fa.append(roc_auc_score(y[va], p_va))
        print(f"  {name} fold {k + 1}: AUC {fa[-1]:.5f}", flush=True)
    auc = roc_auc_score(y, oof)
    print(f"{name}: CV AUC {auc:.5f} | mean fold {np.mean(fa):.5f} | folds {np.round(fa, 5)} | {time.time() - t0:.0f}s", flush=True)
    if save:
        np.save(os.path.join(OOF_DIR, f"{name}_oof.npy"), oof)
        np.save(os.path.join(OOF_DIR, f"{name}_test.npy"), test_z)
        test_p = 1 / (1 + np.exp(-test_z))   # monotone transform: AUC-equivalent, valid probabilities
        ids = pd.read_csv(os.path.join(DATA, "test.csv"), usecols=["id"])["id"]
        pd.DataFrame({"id": ids, TARGET: test_p}).to_csv(os.path.join(SUB_DIR, f"{name}.csv"), index=False)
        log_result(name, auc, np.std(fa), time.time() - t0, description)
    return oof, test_z, fa
