"""v26 - RealMLP (pytabkit) on the GPU (run with E:\\envs\\kaggle\\Scripts\\python.exe).

Inputs (no target encoding): the 13 raw numerics (periodic/piecewise-linear embeddings inside RealMLP),
plus categorical columns with learned embeddings: the 5 categoricals, the 6 flags, the 6 float 'cents'
(0..99), and exact customer_age, card_age_months, txn_count_last_24h, cvv_retry_count.
Early stopping uses a 5% stratified holdout taken from the outer TRAINING fold only.
Usage: python v26_realmlp.py [pilot]
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from pytabkit import RealMLP_TD_Classifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from common import CATS, DATA, FLAGS, NUMS, OOF_DIR, SUB_DIR, TARGET, load, log_result
from fe_keys import FLOATS

warnings.filterwarnings("ignore")
PILOT = len(sys.argv) > 1 and sys.argv[1] == "pilot"
NAME = "v26_realmlp"
P = dict(n_ens=8, n_epochs=16, batch_size=512, lr=0.03, lr_sched="lin_cos_log_15", num_emb_type="pblrd", plr_sigma=16.0,
         plr_hidden_1=16, plr_hidden_2=8, plr_lr_factor=0.10, hidden_sizes=[512, 256, 128], act="mish", p_drop=0.05,
         wd=0.02, ls_eps=0.005, embedding_size=6, max_one_hot_cat_size=18, val_metric_name="1-auc_ovr",
         use_early_stopping=True, verbosity=0, device="cuda", random_state=42)

train, test, y = load()
n = len(train)
full = pd.concat([train, test], ignore_index=True)
X = full[NUMS].astype(np.float32).copy()
cat_cols = []
for c in CATS + FLAGS:
    X[f"cat_{c}"] = full[c].astype(str); cat_cols.append(f"cat_{c}")
for c in FLOATS:
    X[f"cents_{c}"] = (np.rint(full[c].to_numpy() * 100).astype(np.int64) % 100).astype(str); cat_cols.append(f"cents_{c}")
for c in ["customer_age", "card_age_months", "txn_count_last_24h", "cvv_retry_count"]:
    X[f"exact_{c}"] = full[c].astype(str); cat_cols.append(f"exact_{c}")
for c in cat_cols:
    X[c] = X[c].astype("category")
Xtr_all, Xte = X.iloc[:n].reset_index(drop=True), X.iloc[n:].reset_index(drop=True)


def fit_predict(rows, X_evals, seed=42):
    fit_rows, ho_rows = train_test_split(rows, test_size=0.05, stratify=y[rows], random_state=seed)
    m = RealMLP_TD_Classifier(**{**P, "random_state": seed})
    m.fit(Xtr_all.iloc[fit_rows], y[fit_rows], X_val=Xtr_all.iloc[ho_rows], y_val=y[ho_rows], cat_col_names=cat_cols)
    return [m.predict_proba(Xe)[:, 1] for Xe in X_evals]


folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n), y))
oof = np.zeros(n); tmean = np.zeros(len(test)); t0 = time.time()
for k, (tr, va) in enumerate(folds):
    if PILOT and k > 0:
        break
    pv, pt = fit_predict(tr, [Xtr_all.iloc[va], Xte])
    oof[va] = pv; tmean += pt / 5
    print(f"  fold {k + 1}: AUC {roc_auc_score(y[va], pv):.5f}  {time.time() - t0:.0f}s", flush=True)
if PILOT:
    sys.exit(0)
fa = [roc_auc_score(y[va], oof[va]) for _, va in folds]
auc = roc_auc_score(y, oof)
print(f"{NAME}: CV AUC {auc:.5f} | folds {np.round(fa, 5)} | {time.time() - t0:.0f}s", flush=True)
np.save(os.path.join(OOF_DIR, f"{NAME}_oof.npy"), oof); np.save(os.path.join(OOF_DIR, f"{NAME}_test.npy"), tmean)
ids = pd.read_csv(os.path.join(DATA, "test.csv"), usecols=["id"])["id"]
pd.DataFrame({"id": ids, TARGET: tmean}).to_csv(os.path.join(SUB_DIR, f"{NAME}.csv"), index=False)
log_result(NAME, auc, np.std(fa), time.time() - t0, "RealMLP (pytabkit, GPU, n_ens 8, PBLD embeddings) on raw + categorical cents/ages, no TE, 5% inner holdout")
print("done", flush=True)
