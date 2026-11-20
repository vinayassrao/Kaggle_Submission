"""P4.0 - Save the pure 10-fold dense GLM (the init_score model inside v28) as its own stack member.

Same encodings and GLM as v28: 10 folds (seed 42), 4 encoding seeds (teseed 0,1 with smoothing 20;
teseed 2,3 with smoothing 100), LogisticRegression(C=0.3, newton-cholesky) on 73 TE logits + base matrix.
Outputs oof/v28_glm_only_{oof,test,full}.npy:
  oof  = sigmoid(mean over seeds of the GLM decision function) on each validation fold
  test = sigmoid(mean over folds x seeds of the standardised test decision function)
  full = sigmoid(mean over seeds of a GLM fitted on all training rows)
"""
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from common import DATA, OOF_DIR, SUB_DIR, TARGET, load, log_result
from fe_keys import build_key_store, encode_keys, glm_base_matrix, safe_logit

warnings.filterwarnings("ignore")
NAME = "v28_glm_only"
SEEDS = [(0, 20.0), (1, 20.0), (2, 100.0), (3, 100.0)]
train, test, y = load()
n = len(train)
knames, kmat, kcnt, ncat, cents = build_key_store(train, test)
gbase = glm_base_matrix(train, test)
std = lambda z: (z - z.mean()) / z.std()


def glm(a_rows, Ea, others):
    sc = StandardScaler().fit(safe_logit(Ea))
    m = LogisticRegression(C=0.3, max_iter=300, tol=1e-7, solver="newton-cholesky")
    m.fit(np.hstack([sc.transform(safe_logit(Ea)), gbase[a_rows]]), y[a_rows])
    return [m.decision_function(np.hstack([sc.transform(safe_logit(E)), gbase[r]])) for r, E in others]


TEST = np.arange(n, n + len(test))
folds = list(StratifiedKFold(10, shuffle=True, random_state=42).split(np.zeros(n), y))
oof = np.zeros(n); tz = np.zeros(len(test)); t0 = time.time()
for k, (tr, va) in enumerate(folds):
    v = np.zeros(len(va))
    for teseed, sm in SEEDS:
        Ea, Eb, Et = encode_keys(kmat, ncat, y, n, tr, va, smooth=sm, teseed=teseed)
        gv, gt = glm(tr, Ea, [(va, Eb), (TEST, Et)])
        v += gv / len(SEEDS); tz += std(gt) / (len(SEEDS) * len(folds))
    oof[va] = expit(v)
    print(f"  fold {k + 1}: AUC {roc_auc_score(y[va], oof[va]):.5f}  {time.time() - t0:.0f}s", flush=True)
auc = roc_auc_score(y, oof)
fa = [roc_auc_score(y[va], oof[va]) for _, va in folds]
print(f"{NAME}: CV AUC {auc:.5f} | {time.time() - t0:.0f}s", flush=True)
full = np.zeros(len(test))
for teseed, sm in SEEDS:
    Ea, _, Et = encode_keys(kmat, ncat, y, n, None, None, smooth=sm, teseed=teseed)
    full += glm(np.arange(n), Ea, [(TEST, Et)])[0] / len(SEEDS)
np.save(os.path.join(OOF_DIR, f"{NAME}_oof.npy"), oof)
np.save(os.path.join(OOF_DIR, f"{NAME}_test.npy"), expit(tz))
np.save(os.path.join(OOF_DIR, f"{NAME}_full.npy"), expit(full))
ids = pd.read_csv(os.path.join(DATA, "test.csv"), usecols=["id"])["id"]
pd.DataFrame({"id": ids, TARGET: expit(full)}).to_csv(os.path.join(SUB_DIR, f"{NAME}.csv"), index=False)
log_result(NAME, auc, np.std(fa), time.time() - t0, "Pure 10-fold dense GLM (v28's init_score model), 4 encoding seeds m20/m100, full refit")
print("done", flush=True)
