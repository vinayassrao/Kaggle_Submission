# Phase 4 code appendix: Kaggle Playground S6E11

Exact source of the Phase 4 scripts, generated from the files on disk. `kaggle_v25_gpt2.py` is stored locally byte-for-byte as the version running on Kaggle (Version #1), so it isn't repeated here.

## Answer to the `_full` vs `_test` pairing question

Checked. `v31_stack.py` → `load_member(m, prefer_full=True)` uses `_full.npy` for **every** member that has one, and v17, v17b, v28 **and** v28_glm_only all have `_full.npy`. So in `v31_stack.csv` the −0.474 GLM weight subtracts a full-data GLM from full-data GLM+tree refits. `v31_conservative.csv` uses `prefer_full=False` → `_test.npy` (fold means) for every member, which is also consistent. Members without a `_full.npy` (v26, v11, v10) use their `_test.npy` in both outputs.


## `v31_stack.py`

v31 stack: marginal-gain gate, unconstrained vs non-negative stacker, zones, main + conservative outputs

```python
"""v31 - Phase 4 stack with a strict marginal-gain gate and a non-negative stacker (P4.5).

1. Base pool = the 10 v30 members. Candidates = v28_glm_only, v25_gpt2, v26b_realmlp_keys, v27b_digit_cnn_seg (if on disk).
2. Gate: each candidate is added to the base pool alone; mean nested AUC over stacker splits 43/44/45 must rise by
   >= +0.00004 and the per-fold AUC (split 43) must rise in >= 4/5 folds. Passing candidates are then added together.
3. Final pool compared with: unconstrained LogReg (C grid) vs non-negative L2 logistic regression (L-BFGS-B, weights >= 0).
   Prefer the non-negative one if within 0.00002 of the best.
4. Zero-fraud zones verified leak-free on the seed-42 folds, then applied.
Outputs: submissions/v31_stack.csv (chosen), submissions/v31_conservative.csv (non-negative weights, test inputs from
5-fold means instead of full refits).
"""
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import DATA, OOF_DIR, SUB_DIR, TARGET, load, log_result
from fe_keys import rank_logit, safe_logit

BASE = ["v17_glm_lgb10", "v17b_glm_lgb10_s100", "v28_glm_xgb10_gpu", "v19_sparse_glm_10f", "v16_glm_lgb",
        "v26_realmlp", "v27_digit_cnn", "v11_ebm", "v22_lgb4_raw", "v10_spline_logreg"]
CANDS = ["v28_glm_only", "v25_gpt2", "v26b_realmlp_keys", "v27b_digit_cnn_seg"]
NATIVE10 = ("lgb10", "xgb10", "_10f", "glm_only")
train, test, y = load()
n = len(y)
ex = lambda m: os.path.exists(os.path.join(OOF_DIR, f"{m}_oof.npy"))
CANDS = [c for c in CANDS if ex(c)]
r01 = lambda v: (rankdata(v) - 1) / (len(v) - 1)


def load_member(m, prefer_full=True):
    o = np.load(os.path.join(OOF_DIR, f"{m}_oof.npy"))
    k = 10 if any(t in m for t in NATIVE10) else 5
    z = safe_logit(o); zs = np.zeros_like(z)
    for _, va in StratifiedKFold(k, shuffle=True, random_state=42).split(np.zeros(n), y):
        zs[va] = (z[va] - z[va].mean()) / (z[va].std() + 1e-12)
    order = ("_full.npy", "_test.npy") if prefer_full else ("_test.npy", "_full.npy")
    for suf in order:
        p = os.path.join(OOF_DIR, m + suf)
        if os.path.exists(p):
            return rank_logit(zs), rank_logit(np.load(p)), suf
    raise FileNotFoundError(m)


MEM = {m: load_member(m) for m in BASE + CANDS}
SPLITS = [list(StratifiedKFold(5, shuffle=True, random_state=s).split(np.zeros(n), y)) for s in (43, 44, 45)]


def nn_logreg(Z, yy, lam):
    """Non-negative L2 logistic regression: weights >= 0, free intercept."""
    M = Z.shape[1]

    def f(w):
        s = Z @ w[:M] + w[M]
        loss = np.mean(np.logaddexp(0, s) - yy * s) + lam * np.sum(w[:M] ** 2)
        p = expit(s); g = np.r_[Z.T @ (p - yy) / len(yy) + 2 * lam * w[:M], np.mean(p - yy)]
        return loss, g
    w0 = np.r_[np.full(M, 1.0 / M), -4.0]
    res = minimize(f, w0, jac=True, method="L-BFGS-B", bounds=[(0, None)] * M + [(None, None)])
    return res.x


def combiners(Ztr, ytr, Zev):
    out = {}
    for C in (0.005, 0.01, 0.03, 0.05):
        out[f"lr_C{C}"] = LogisticRegression(C=C, solver="newton-cholesky", max_iter=200).fit(Ztr, ytr).decision_function(Zev)
    for lam in (1e-4, 1e-3):
        w = nn_logreg(Ztr, ytr, lam); out[f"nnlr_l{lam:g}"] = Zev @ w[:-1] + w[-1]
    return out


def nested(names, only=None):
    """Mean nested AUC per combiner over splits 43/44/45, plus split-43 per-fold AUCs."""
    Z = np.column_stack([MEM[m][0] for m in names])
    res, fold43 = {}, {}
    for si, sp in enumerate(SPLITS):
        pred = {}
        for tr, va in sp:
            for k, v in combiners(Z[tr], y[tr], Z[va]).items():
                if only and k != only:
                    continue
                pred.setdefault(k, np.zeros(n))[va] = r01(v)
        for k, v in pred.items():
            res.setdefault(k, []).append(roc_auc_score(y, v))
            if si == 0:
                fold43[k] = np.array([roc_auc_score(y[va], v[va]) for _, va in sp])
    return {k: float(np.mean(v)) for k, v in res.items()}, fold43, res


print("candidates on disk:", CANDS)
base_scores, base_f43, _ = nested(BASE)
REF = "lr_C0.05"
print(f"base pool (v30 members) nested AUC [{REF}]: {base_scores[REF]:.5f}")
kept = []
for c in CANDS:
    sc, f43, _ = nested(BASE + [c], only=REF)
    gain = sc[REF] - base_scores[REF]; up = int((f43[REF] > base_f43[REF]).sum())
    ok = gain >= 4e-5 and up >= 4
    print(f"  + {c:22s} nested {sc[REF]:.5f}  gain {gain:+.5f}  folds up {up}/5  -> {'KEEP' if ok else 'drop'}")
    if ok:
        kept.append(c)
POOL = BASE + kept
final_scores, _, raw = nested(POOL)
print("\nfinal pool:", POOL)
for k, v in final_scores.items():
    print(f"  {k:12s} {v:.5f}  {np.round(raw[k], 5)}")
best = max(final_scores, key=final_scores.get)
nn_best = max((k for k in final_scores if k.startswith("nnlr")), key=final_scores.get)
chosen = nn_best if final_scores[best] - final_scores[nn_best] <= 2e-5 else best
print(f"best {best} {final_scores[best]:.5f} | best non-negative {nn_best} {final_scores[nn_best]:.5f} -> chosen {chosen}")

Z = np.column_stack([MEM[m][0] for m in POOL]); T = np.column_stack([MEM[m][1] for m in POOL])
full_out = combiners(Z, y, T)
lam = float(nn_best.split("_l")[1]); w = nn_logreg(Z, y, lam)
print("non-negative weights:", dict(zip(POOL, np.round(w[:-1], 3))))
lr = LogisticRegression(C=0.05, solver="newton-cholesky", max_iter=200).fit(Z, y)
print("LogReg C=0.05 coefs:", dict(zip(POOL, np.round(lr.coef_[0], 3))))

# zero-fraud zones, verified on the seed-42 folds with the chosen combiner's nested OOF (split 43)
_, _, _ = None, None, None
oof43 = np.zeros(n)
for tr, va in SPLITS[0]:
    oof43[va] = r01(combiners(Z[tr], y[tr], Z[va])[chosen])
mrs, vel = train["merchant_risk_score"].to_numpy(), train["velocity_score"].to_numpy()
gains = []
for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n), y):
    thr = mrs[tr][y[tr] == 1].min()
    zmask = (mrs[va] < thr) | ((vel[va] <= 4.1) & (mrs[va] <= 20.0))
    adj = oof43[va].copy(); adj[zmask] -= 10
    gains.append(roc_auc_score(y[va], adj) - roc_auc_score(y[va], oof43[va]))
use_z = all(g > 0 for g in gains)
print(f"zones: gains {np.round(gains, 6)} -> {'APPLY' if use_z else 'skip'}")
thr = mrs[y == 1].min(); tm, tv = test["merchant_risk_score"].to_numpy(), test["velocity_score"].to_numpy()
zt = (tm < thr) | ((tv <= 4.1) & (tm <= 20.0))


def finish(score):
    s = r01(score)
    if use_z:
        s = s.copy(); s[zt] -= 10; s = r01(s)
    return s


ids = pd.read_csv(os.path.join(DATA, "test.csv"), usecols=["id"])["id"]
main = finish(full_out[chosen])
pd.DataFrame({"id": ids, TARGET: main}).to_csv(os.path.join(SUB_DIR, "v31_stack.csv"), index=False)
np.save(os.path.join(OOF_DIR, "v31_stack_oof.npy"), oof43); np.save(os.path.join(OOF_DIR, "v31_stack_test.npy"), main)
# conservative: non-negative weights, 5-fold-mean test inputs instead of full refits
Tc = np.column_stack([load_member(m, prefer_full=False)[1] for m in POOL])
cons = finish(Tc @ w[:-1] + w[-1])
pd.DataFrame({"id": ids, TARGET: cons}).to_csv(os.path.join(SUB_DIR, "v31_conservative.csv"), index=False)
print(f"Spearman(main, conservative) on test: {pd.Series(main).corr(pd.Series(cons), method='spearman'):.5f}")
log_result("v31_stack", final_scores[chosen], float(np.std(raw[chosen])), 0,
           f"{chosen} on {len(POOL)} members (kept: {', '.join(kept) or 'none'}); zones {'on' if use_z else 'off'}")
print(f"v31_stack: {chosen} nested {final_scores[chosen]:.5f}")
```


## `p4_glm_only.py`

P4.0: pure 10-fold dense GLM member

```python
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
```


## `v26b_realmlp_keys.py`

P4.3: RealMLP with the 60 key categoricals (fold-1 pilot only)

```python
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
```


## `v27b_digit_cnn_seg.py`

P4.4: position-preserving digit CNN (fold-1 pilot only)

```python
"""v27b - Position-preserving digit CNN (P4.4): v27 with segment pooling instead of global pooling.

Original v27 docstring: 1-D character/digit CNN on the GPU, with NO target encoding (run with E:\\envs\\kaggle\\Scripts\\python.exe).

Every row is written as a fixed-width character string (floats with fixed formats, e.g. amount '06872.69',
integers zero-padded), and each categorical/flag value becomes one token. A character embedding plus a
position embedding feeds 3 residual Conv1d blocks, then mean+max pooling and a small head.
Honest training: inside each outer training fold, a 5% stratified holdout picks the best epoch; the outer
validation fold is never used for any decision. 2 seeds per fold; test logits standardised per fold and averaged.
Full refit on all training rows for the mean best epoch count.
Usage: python v27_digit_cnn.py [pilot]   (pilot = fold 1 only)
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from common import CATS, DATA, FLAGS, OOF_DIR, SUB_DIR, TARGET, load, log_result

warnings.filterwarnings("ignore")
DEV = "cuda"
PILOT = len(sys.argv) > 1 and sys.argv[1] == "pilot"
NAME, EPOCHS, BATCH, SEEDS = "v27b_digit_cnn_seg", 12, 2048, [42, 137, 2026]
FMT = [("amount_usd", "{:08.2f}"), ("hours_since_last_txn", "{:05.2f}"), ("distance_from_home_km", "{:05.2f}"),
       ("account_balance_usd", "{:08.2f}"), ("velocity_score", "{:04.1f}"), ("merchant_risk_score", "{:05.1f}"),
       ("txn_count_last_24h", "{:02d}"), ("card_age_months", "{:02d}"), ("customer_age", "{:02d}"),
       ("cvv_retry_count", "{:01d}"), ("time_of_day_hour", "{:02d}"), ("day_of_week", "{:01d}"), ("prior_disputes", "{:01d}")]

train, test, y = load()
n = len(train)
full = pd.concat([train, test], ignore_index=True)

# ---- encode rows as fixed-length token sequences ----
chars = list("0123456789.|")
vocab = {c: i + 1 for i, c in enumerate(chars)}          # 0 = padding (unused: all rows same length)
parts = []
for c, f in FMT:
    s = full[c].map(lambda v: f.format(int(v) if "d" in f else float(v)))
    lens = s.str.len().unique()
    assert len(lens) == 1, (c, lens)                      # fixed width per column
    parts.append(np.array([[vocab[ch] for ch in v] + [vocab["|"]] for v in s], dtype=np.int16))
for c in CATS + FLAGS:                                    # one token per categorical / flag value
    codes, uniq = pd.factorize(full[c].astype(str))
    base = len(vocab) + 1
    for u in uniq:
        vocab[f"{c}={u}"] = base; base += 1
    parts.append(np.array([vocab[f"{c}={u}"] for u in uniq], dtype=np.int16)[codes][:, None])
SEQ = np.hstack(parts).astype(np.int64)
V, L = len(vocab) + 1, SEQ.shape[1]
print(f"sequence length {L}, vocab {V}, device {torch.cuda.get_device_name(0)}", flush=True)


class Res(nn.Module):
    def __init__(self, c, k):
        super().__init__()
        self.f = nn.Sequential(nn.Conv1d(c, c, k, padding=k // 2), nn.BatchNorm1d(c), nn.GELU(),
                               nn.Conv1d(c, c, k, padding=k // 2), nn.BatchNorm1d(c))
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.f(x))


class Net(nn.Module):
    def __init__(self, d=32, c=128):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V, d), nn.Embedding(L, d)
        self.inp = nn.Conv1d(d, c, 1)
        self.blocks = nn.Sequential(Res(c, 3), Res(c, 5), Res(c, 3))
        self.avg, self.mx = nn.AdaptiveAvgPool1d(24), nn.AdaptiveMaxPool1d(24)   # keep WHERE in the row a pattern occurs
        self.head = nn.Sequential(nn.LayerNorm(2 * c * 24), nn.Dropout(0.25), nn.Linear(2 * c * 24, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, x):
        h = self.tok(x) + self.pos(torch.arange(L, device=x.device))[None]
        h = self.blocks(self.inp(h.transpose(1, 2)))
        return self.head(torch.cat([self.avg(h).flatten(1), self.mx(h).flatten(1)], 1)).squeeze(1)


def predict(net, idx):
    net.eval(); out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(idx), 16384):
            out.append(net(torch.as_tensor(SEQ[idx[i:i + 16384]], device=DEV)).float().cpu().numpy())
    return np.concatenate(out)


def fit(rows, seed, epochs=EPOCHS, holdout=True):
    """Train on `rows`; if holdout, keep 5% aside to pick the best epoch. Returns (net, best_epoch)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    if holdout:
        fit_rows, ho_rows = train_test_split(rows, test_size=0.05, stratify=y[rows], random_state=seed)
    else:
        fit_rows, ho_rows = rows, None
    net = Net().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(fit_rows) / BATCH))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler()
    lossf = nn.BCEWithLogitsLoss()
    Xg = torch.as_tensor(SEQ[fit_rows], device=DEV); yg = torch.as_tensor(y[fit_rows], dtype=torch.float32, device=DEV)
    best, best_ep, best_state = -1, epochs, None
    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.as_tensor(rng.permutation(len(fit_rows)), device=DEV)
        for i in range(0, len(fit_rows), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = lossf(net(Xg[b]), yg[b])
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        if holdout:
            a = roc_auc_score(y[ho_rows], predict(net, ho_rows))
            if a > best:
                best, best_ep = a, ep
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    if holdout:
        net.load_state_dict(best_state)
    return net, best_ep


std = lambda z: (z - z.mean()) / (z.std() + 1e-12)
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n), y))
oof = np.full(n, np.nan); test_z = np.zeros(len(test)); eps = []; t0 = time.time()
TEST = np.arange(n, n + len(test))
for k, (tr, va) in enumerate(folds):
    if PILOT and k > 0:
        break
    zv = np.zeros(len(va)); zt = np.zeros(len(test))
    for s in SEEDS:
        net, ep = fit(tr, s); eps.append(ep)
        pv, pt = predict(net, va), predict(net, TEST)
        mu, sd = pt.mean(), pt.std()                       # align val and test with the TEST statistics
        zv += (pv - mu) / sd / len(SEEDS); zt += (pt - mu) / sd / len(SEEDS)
        print(f"  fold {k + 1} seed {s}: AUC {roc_auc_score(y[va], pv):.5f} (best epoch {ep})  {time.time() - t0:.0f}s", flush=True)
    oof[va] = zv; test_z += zt / 5
    print(f"  fold {k + 1}: AUC {roc_auc_score(y[va], zv):.5f}", flush=True)
if PILOT:
    sys.exit(0)
fa = [roc_auc_score(y[va], oof[va]) for _, va in folds]
auc = roc_auc_score(y, oof)   # within-fold standardised logits
print(f"{NAME}: CV AUC {auc:.5f} | folds {np.round(fa, 5)} | best epochs {eps} | {time.time() - t0:.0f}s", flush=True)
sig = lambda z: 1 / (1 + np.exp(-z))
np.save(os.path.join(OOF_DIR, f"{NAME}_oof.npy"), sig(oof)); np.save(os.path.join(OOF_DIR, f"{NAME}_test.npy"), sig(test_z))
ne = int(round(np.mean(eps)))
ft = np.mean([std(predict(fit(np.arange(n), s, epochs=ne, holdout=False)[0], TEST)) for s in SEEDS], axis=0)
np.save(os.path.join(OOF_DIR, f"{NAME}_full.npy"), sig(ft))
ids = pd.read_csv(os.path.join(DATA, "test.csv"), usecols=["id"])["id"]
pd.DataFrame({"id": ids, TARGET: sig(ft)}).to_csv(os.path.join(SUB_DIR, f"{NAME}.csv"), index=False)
log_result(NAME, auc, np.std(fa), time.time() - t0, f"Position-preserving digit CNN (24-segment avg+max pooling), no TE, 3 seeds, test-aligned logits, 5% inner holdout for epoch, full refit {ne} epochs")
print("done", flush=True)
```


## `fe_keys.py`

shared key store + nested bincount target encoding (used by p4_glm_only and v26b)

```python
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
```


## `common.py`

shared helpers: load(), raw_features(), log_result(), constants

```python
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
```
