"""Phase 2 shared feature engine (P2.0).

Label-free key store (train+test), cached to keys_cache.npz:
  - 24 exact-value keys (every column)
  - 36 truncated digit keys: for each of the 6 float columns -> integer part, tens, tenths, cents, lead1, lead2
  - 13 curated pair keys (used only by the dense GLM)
  - key frequencies (train+test counts) and the 6 numeric 'cents' columns
Nested target encoding with np.bincount: training rows are encoded out-of-fold by an inner
StratifiedKFold; validation/test rows are encoded from all training rows. Leak-free.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import CATS, FLAGS, HERE, NUMS

FLOATS = ["amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd", "velocity_score", "merchant_risk_score"]
FEATS = NUMS + FLAGS + CATS
N_TREE_KEYS = 60   # 24 exact + 36 digit keys; the 13 pair keys (60..72) are GLM-only
GLM_PAIRS = [("merchant_category", "merchant_risk_score"), ("is_foreign_transaction", "ip_country_mismatch"),
             ("merchant_category", "q:amount_usd"), ("txn_count_last_24h", "q:velocity_score"),
             ("merchant_category", "customer_age"), ("q:amount_usd", "q:merchant_risk_score"),
             ("q:hours_since_last_txn", "q:velocity_score"), ("customer_age", "q:merchant_risk_score"),
             ("is_new_merchant", "day_of_week"), ("is_ai_generated_scam_attempt", "q:velocity_score"),
             ("customer_age", "card_age_months"), ("time_of_day_hour", "day_of_week"), ("txn_count_last_24h", "customer_age")]
CACHE = os.path.join(HERE, "keys_cache.npz")


def safe_logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


def rank_logit(x, eps=1e-6):
    return safe_logit(rankdata(np.asarray(x, dtype=np.float64)) / (len(x) + 1.0), eps)


def build_key_store(train, test, use_cache=True):
    if use_cache and os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        return list(z["knames"]), z["kmat"], z["kcnt"], z["ncat"], z["cents"]
    both = pd.concat([train[FEATS], test[FEATS]], ignore_index=True).copy()
    both[FLAGS] = both[FLAGS].astype(np.int8)
    keys = {c: both[c] for c in FEATS}
    cents = np.empty((len(both), len(FLOATS)), dtype=np.float32)
    for j, c in enumerate(FLOATS):
        units = np.rint(both[c].to_numpy(np.float64) * 100).astype(np.int64)   # exact integer cents
        integer = units // 100                                                 # truncate, never round
        s = pd.Series(integer).astype(str)
        keys[f"{c}__int"] = integer
        keys[f"{c}__int10"] = units // 1000
        keys[f"{c}__tenths"] = units // 10
        keys[f"{c}__cents"] = units % 100
        keys[f"{c}__lead1"] = (s.str.len() * 10 + s.str[0].astype(int)).to_numpy()
        keys[f"{c}__lead2"] = (s.str.len() * 100 + s.str[:2].astype(int)).to_numpy()
        cents[:, j] = (units % 100).astype(np.float32)

    def part(col):
        return pd.qcut(both[col[2:]], 10, labels=False, duplicates="drop") if col.startswith("q:") else both[col]

    for a, b in GLM_PAIRS:
        keys[f"pair_{a}_{b}".replace("q:", "")] = part(a).astype(str) + "|" + part(b).astype(str)
    knames = list(keys)
    kmat = np.empty((len(both), len(knames)), dtype=np.int32)
    kcnt = np.empty((len(both), len(knames)), dtype=np.float32)
    ncat = np.empty(len(knames), dtype=np.int32)
    for j, n in enumerate(knames):
        codes, uniq = pd.factorize(pd.Series(keys[n]), sort=False)
        kmat[:, j] = codes; ncat[j] = len(uniq)
        kcnt[:, j] = np.bincount(codes, minlength=len(uniq))[codes]
    np.savez(CACHE, knames=np.array(knames, dtype=object), kmat=kmat, kcnt=kcnt, ncat=ncat, cents=cents)
    return knames, kmat, kcnt, ncat, cents


def _rate(codes, yf, nl, prior, smooth):
    s = np.bincount(codes, weights=yf, minlength=nl); n = np.bincount(codes, minlength=nl)
    return (s + prior * smooth) / (n + smooth)


def encode_keys(kmat, ncat, y, n_train, tr_idx=None, va_idx=None, smooth=20.0, inner_folds=5, teseed=0, n_cols=None):
    """Returns (E_train_rows_oof, E_val_rows, E_test_rows) as smoothed fraud rates."""
    use = kmat.shape[1] if n_cols is None else n_cols
    a = np.arange(n_train) if tr_idx is None else np.asarray(tr_idx)
    ya = y[a].astype(np.float64)
    Ka, Kt = kmat[a, :use], kmat[n_train:, :use]
    Kb = kmat[va_idx, :use] if va_idx is not None else None
    Ea = np.empty((len(a), use), np.float32); Et = np.empty((len(Kt), use), np.float32)
    Eb = np.empty((len(va_idx), use), np.float32) if va_idx is not None else None
    inner = list(StratifiedKFold(inner_folds, shuffle=True, random_state=teseed).split(np.zeros(len(a)), ya))
    pa = float(ya.mean())
    for j in range(use):
        nl, col = int(ncat[j]), Ka[:, j]
        for fi, vi in inner:
            Ea[vi, j] = _rate(col[fi], ya[fi], nl, float(ya[fi].mean()), smooth)[col[vi]]
        r = _rate(col, ya, nl, pa, smooth)
        Et[:, j] = r[Kt[:, j]]
        if Kb is not None:
            Eb[:, j] = r[Kb[:, j]]
    return Ea, Eb, Et


def glm_base_matrix(train, test):
    full = pd.concat([train[FEATS], test[FEATS]], ignore_index=True)
    num = full[NUMS].astype(np.float64).copy()
    for c in ("amount_usd", "account_balance_usd", "hours_since_last_txn"):
        num[c] = np.log1p(num[c].clip(lower=0))
    Z = StandardScaler().fit_transform(num).astype(np.float32)
    OH = OneHotEncoder(sparse_output=False, dtype=np.float32).fit_transform(full[CATS].astype(str))
    return np.hstack([Z, full[FLAGS].astype(np.float32).to_numpy(), OH])


def tree_frame(raw_all, cents, kcnt, knames, rows, E):
    df = raw_all.iloc[rows].reset_index(drop=True)
    extra = pd.DataFrame({**{f"cents_{c}": cents[rows, j] for j, c in enumerate(FLOATS)},
                          **{f"te_{knames[j]}": E[:, j] for j in range(N_TREE_KEYS)},
                          **{f"cnt_{knames[j]}": kcnt[rows, j] for j in range(N_TREE_KEYS)}})
    return pd.concat([df, extra], axis=1)
