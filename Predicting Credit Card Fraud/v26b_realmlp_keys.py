"""v26b - RealMLP with key categoricals (P4.3). Run with E:\\envs\\kaggle\\Scripts\\python.exe.

Changes vs v26:
  * categorical inputs = all 24 exact-value keys + the 36 digit keys from keys_cache (values seen < 5 times in
    train+test pooled into one 'rare' code per column), learned embeddings; plus the 13 raw numerics (PBLD).
    Still no target encoding.
  * label-free fold alignment: each fold's logits are standardised with that fold's TEST mean/std before the
    OOF is stored and the test predictions are averaged.
Usage: python v26b_realmlp_keys.py [pilot]   (pilot = fold 1 only, for the gate)
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from pytabkit import RealMLP_TD_Classifier
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from common import DATA, NUMS, OOF_DIR, SUB_DIR, TARGET, load, log_result
from fe_keys import N_TREE_KEYS, build_key_store, safe_logit

warnings.filterwarnings("ignore")
PILOT = len(sys.argv) > 1 and sys.argv[1] == "pilot"
NAME = "v26b_realmlp_keys"
P = dict(n_ens=8, n_epochs=16, batch_size=512, lr=0.03, lr_sched="lin_cos_log_15", num_emb_type="pblrd", plr_sigma=16.0,
         plr_hidden_1=16, plr_hidden_2=8, plr_lr_factor=0.10, hidden_sizes=[512, 256, 128], act="mish", p_drop=0.05,
         wd=0.02, ls_eps=0.005, embedding_size=6, max_one_hot_cat_size=18, val_metric_name="1-auc_ovr",
         use_early_stopping=True, verbosity=0, device="cuda", random_state=42)

train, test, y = load()
n = len(train)
knames, kmat, kcnt, ncat, cents = build_key_store(train, test)
full = pd.concat([train, test], ignore_index=True)
X = full[NUMS].astype(np.float32).copy()
cat_cols = []
for j in range(N_TREE_KEYS):                     # 24 exact + 36 digit keys
    codes = np.where(kcnt[:, j] >= 5, kmat[:, j], -1)   # pool rare values (<5 occurrences) into one code
    col = f"k_{knames[j]}"
    X[col] = pd.Categorical(codes.astype(str)); cat_cols.append(col)
Xtr_all, Xte = X.iloc[:n].reset_index(drop=True), X.iloc[n:].reset_index(drop=True)
print("categorical key columns:", len(cat_cols), "| max cardinality", max(X[c].cat.categories.size for c in cat_cols), flush=True)


def fit_predict(rows, X_evals, seed=42):
    fit_rows, ho_rows = train_test_split(rows, test_size=0.05, stratify=y[rows], random_state=seed)
    m = RealMLP_TD_Classifier(**{**P, "random_state": seed})
    m.fit(Xtr_all.iloc[fit_rows], y[fit_rows], X_val=Xtr_all.iloc[ho_rows], y_val=y[ho_rows], cat_col_names=cat_cols)
    return [m.predict_proba(Xe)[:, 1] for Xe in X_evals]


folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n), y))
oof = np.zeros(n); tz = np.zeros(len(test)); t0 = time.time()
for k, (tr, va) in enumerate(folds):
    if PILOT and k > 0:
        break
    pv, pt = fit_predict(tr, [Xtr_all.iloc[va], Xte])
    zv, zt = safe_logit(pv), safe_logit(pt)
    mu, sd = zt.mean(), zt.std()                 # label-free alignment with this fold's test statistics
    oof[va] = expit((zv - mu) / sd); tz += (zt - mu) / sd / 5
    print(f"  fold {k + 1}: AUC {roc_auc_score(y[va], pv):.5f}  {time.time() - t0:.0f}s", flush=True)
if PILOT:
    sys.exit(0)
fa = [roc_auc_score(y[va], oof[va]) for _, va in folds]
auc = roc_auc_score(y, oof)
print(f"{NAME}: CV AUC {auc:.5f} | folds {np.round(fa, 5)} | {time.time() - t0:.0f}s", flush=True)
np.save(os.path.join(OOF_DIR, f"{NAME}_oof.npy"), oof); np.save(os.path.join(OOF_DIR, f"{NAME}_test.npy"), expit(tz))
ids = pd.read_csv(os.path.join(DATA, "test.csv"), usecols=["id"])["id"]
pd.DataFrame({"id": ids, TARGET: expit(tz)}).to_csv(os.path.join(SUB_DIR, f"{NAME}.csv"), index=False)
log_result(NAME, auc, np.std(fa), time.time() - t0, "RealMLP with 60 exact/digit key categoricals (rare<5 pooled), no TE, test-aligned fold logits")
print("done", flush=True)
