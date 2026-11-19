"""Phase 2 diversity models sharing the fe_keys engine. Usage: python p2_models.py v19|v20|v21|v22

v19_sparse_glm  joint sparse L2 logistic regression: one-hot of every exact numeric value and digit key
                (seen >= 5 times) + splines + standardised numerics + 60 TE logits.
v20_ebm_digits  EBM (v11 settings) on raw 24 + 6 cents + 60 TE logits.
v21_mlp_bag     scikit-learn MLP (256,128), fixed 12 epochs (no early stopping), 3 seeds.
v22_lgb4_raw    4-leaf LightGBM on raw 24 + 6 cents only (no target encoding): a zero-TE anchor.
All: 5 folds (seed 42), OOF probabilities, test = fold-mean probabilities, plus a full-data refit.
"""
import os
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, SplineTransformer, StandardScaler

from common import CATS, DATA, FLAGS, NUMS, OOF_DIR, SUB_DIR, TARGET, load, log_result, raw_features
from fe_keys import FEATS, FLOATS, N_TREE_KEYS, build_key_store, encode_keys, safe_logit

warnings.filterwarnings("ignore")
WHICH = sys.argv[1]
train, test, y = load()
n = len(train)
knames, kmat, kcnt, ncat, cents = build_key_store(train, test)
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n), y))
ALL_TR, TEST_ROWS = np.arange(n), np.arange(n, n + len(test))


def run(name, fit_predict, needs_te, desc):
    """fit_predict(tr_rows, va_rows_or_None, Ea, Eb, Et) -> (p_va or None, p_te, info)."""
    oof = np.zeros(n); tmean = np.zeros(len(test)); t0 = time.time(); infos = []
    for k, (tr, va) in enumerate(folds):
        Ea, Eb, Et = encode_keys(kmat, ncat, y, n, tr, va, smooth=20.0, teseed=0, n_cols=N_TREE_KEYS) if needs_te else (None, None, None)
        p_va, p_te, info = fit_predict(tr, va, Ea, Eb, Et)
        oof[va] = p_va; tmean += p_te / 5; infos.append(info)
        print(f"  {name} fold {k + 1}: AUC {roc_auc_score(y[va], p_va):.5f} ({info}) {time.time() - t0:.0f}s", flush=True)
    fa = [roc_auc_score(y[va], oof[va]) for _, va in folds]
    auc = roc_auc_score(y, oof)
    print(f"{name}: CV AUC {auc:.5f} | folds {np.round(fa, 5)} | {time.time() - t0:.0f}s", flush=True)
    np.save(os.path.join(OOF_DIR, f"{name}_oof.npy"), oof); np.save(os.path.join(OOF_DIR, f"{name}_test.npy"), tmean)
    Ea, _, Et = encode_keys(kmat, ncat, y, n, None, None, smooth=20.0, teseed=0, n_cols=N_TREE_KEYS) if needs_te else (None, None, None)
    _, full, _ = fit_predict(ALL_TR, None, Ea, None, Et, refit_info=infos)
    np.save(os.path.join(OOF_DIR, f"{name}_full.npy"), full)
    ids = pd.read_csv(os.path.join(DATA, "test.csv"), usecols=["id"])["id"]
    pd.DataFrame({"id": ids, TARGET: full}).to_csv(os.path.join(SUB_DIR, f"{name}.csv"), index=False)
    log_result(name, auc, np.std(fa), time.time() - t0, desc)


full_df = pd.concat([train[FEATS], test[FEATS]], ignore_index=True)

if WHICH == "v19":
    num = full_df[NUMS].astype(np.float64).copy()
    for c in ("amount_usd", "account_balance_usd", "hours_since_last_txn"):
        num[c] = np.log1p(num[c].clip(lower=0))
    Z = StandardScaler().fit_transform(num)
    M = sparse.hstack([
        sparse.csr_matrix(Z, dtype=np.float32), sparse.csr_matrix(full_df[FLAGS].to_numpy(np.float32)),
        OneHotEncoder(handle_unknown="ignore", dtype=np.float32).fit_transform(full_df[CATS].astype(str)),
        sparse.csr_matrix(SplineTransformer(n_knots=8, degree=3, knots="quantile").fit_transform(Z), dtype=np.float32),
        OneHotEncoder(handle_unknown="ignore", min_frequency=5, dtype=np.float32).fit_transform(kmat[:, :len(NUMS)]),
        OneHotEncoder(handle_unknown="ignore", min_frequency=5, dtype=np.float32).fit_transform(kmat[:, 24:60]),
    ]).tocsr()
    print("sparse matrix", M.shape, flush=True)

    def fp(tr, va, Ea, Eb, Et, refit_info=None):
        sc = StandardScaler().fit(safe_logit(Ea))
        D = lambda rows, E: sparse.hstack([M[rows], sparse.csr_matrix(sc.transform(safe_logit(E)), dtype=np.float32)]).tocsr()
        clf = LogisticRegression(C=0.01, solver="lbfgs", max_iter=3000, tol=1e-6).fit(D(tr, Ea), y[tr])
        pv = clf.predict_proba(D(va, Eb))[:, 1] if va is not None else None
        return pv, clf.predict_proba(D(TEST_ROWS, Et))[:, 1], f"n_iter {clf.n_iter_[0]}"
    run("v19_sparse_glm", fp, True, f"Joint sparse L2 LogReg C=0.01 on {M.shape[1]} cols (exact+digit one-hots, splines) + 60 TE logits")

elif WHICH == "v22":
    raw_tr, raw_te = raw_features(train, test)
    R = pd.concat([raw_tr, raw_te], ignore_index=True)
    for j, c in enumerate(FLOATS):
        R[f"cents_{c}"] = cents[:, j]
    P = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=4, min_child_samples=100, feature_fraction=0.5,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbosity=-1, num_threads=-1, seed=42)

    def fp(tr, va, Ea, Eb, Et, refit_info=None):
        dtr = lgb.Dataset(R.iloc[tr], y[tr])
        if va is not None:
            b = lgb.train(P, dtr, 30000, valid_sets=[lgb.Dataset(R.iloc[va], y[va], reference=dtr)],
                          callbacks=[lgb.early_stopping(300, verbose=False)])
            return b.predict(R.iloc[va], num_iteration=b.best_iteration), b.predict(R.iloc[TEST_ROWS], num_iteration=b.best_iteration), b.best_iteration
        nr = int(round(np.mean(refit_info) * 1.2)); b = lgb.train(P, dtr, nr)
        return None, b.predict(R.iloc[TEST_ROWS]), nr
    run("v22_lgb4_raw", fp, False, "4-leaf LightGBM on raw 24 + 6 cents, no target encoding")

elif WHICH == "v20":
    from interpret.glassbox import ExplainableBoostingClassifier
    raw_tr, raw_te = raw_features(train, test)
    R = pd.concat([raw_tr, raw_te], ignore_index=True)
    R[CATS] = R[CATS].astype(str)
    for j, c in enumerate(FLOATS):
        R[f"cents_{c}"] = cents[:, j]

    def fp(tr, va, Ea, Eb, Et, refit_info=None):
        te = lambda rows, E: pd.concat([R.iloc[rows].reset_index(drop=True),
                                        pd.DataFrame(safe_logit(E), columns=[f"te_{k}" for k in knames[:N_TREE_KEYS]])], axis=1)
        ebm = ExplainableBoostingClassifier(max_bins=1024, max_interaction_bins=64, interactions=10, outer_bags=4, inner_bags=0,
                                            learning_rate=0.015, min_samples_leaf=100, early_stopping_rounds=100, random_state=42, n_jobs=-1)
        ebm.fit(te(tr, Ea), y[tr])
        pv = ebm.predict_proba(te(va, Eb))[:, 1] if va is not None else None
        return pv, ebm.predict_proba(te(TEST_ROWS, Et))[:, 1], "ebm"
    run("v20_ebm_digits", fp, True, "EBM (v11 settings) on raw + cents + 60 exact/digit TE logits")

elif WHICH == "v21":
    from sklearn.neural_network import MLPClassifier
    numc = np.column_stack([full_df[NUMS].to_numpy(np.float64), cents])
    ohe = OneHotEncoder(sparse_output=False, dtype=np.float32).fit_transform(
        np.column_stack([full_df[CATS].astype(str).to_numpy(), full_df[FLAGS].astype(int).astype(str).to_numpy(),
                         (full_df["velocity_score"] == 0).astype(int).astype(str).to_numpy()]))
    SEEDS = [42, 137, 2026]

    def fp(tr, va, Ea, Eb, Et, refit_info=None):
        qt = QuantileTransformer(n_quantiles=2000, output_distribution="normal", random_state=42).fit(numc[tr])
        sc = StandardScaler().fit(safe_logit(Ea))
        X = lambda rows, E: np.hstack([qt.transform(numc[rows]), ohe[rows], sc.transform(safe_logit(E))]).astype(np.float32)
        Xa, Xt = X(tr, Ea), X(TEST_ROWS, Et); Xb = X(va, Eb) if va is not None else None
        pv = np.zeros(len(va)) if va is not None else None; pt = np.zeros(len(TEST_ROWS))
        for s in SEEDS:
            m = MLPClassifier(hidden_layer_sizes=(256, 128), activation="relu", solver="adam", alpha=0.02, batch_size=1024,
                              learning_rate_init=1e-3, max_iter=12, early_stopping=False, random_state=s).fit(Xa, y[tr])
            if va is not None:
                pv += m.predict_proba(Xb)[:, 1] / len(SEEDS)
            pt += m.predict_proba(Xt)[:, 1] / len(SEEDS)
        return pv, pt, f"{len(SEEDS)} seeds x 12 epochs"
    run("v21_mlp_bag", fp, True, "sklearn MLP (256,128) fixed 12 epochs, 3 seeds, quantile nums + one-hots + 60 TE logits")
