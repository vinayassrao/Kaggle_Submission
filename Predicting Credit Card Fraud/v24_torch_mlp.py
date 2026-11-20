"""v24 - PyTorch residual MLP on the GPU (run with E:\\envs\\kaggle\\Scripts\\python.exe).

Inputs (built per fold, leak-free):
  - quantile-normalised numerics + 6 'cents' columns (QuantileTransformer fitted on the training rows)
  - one-hot categoricals, flags, velocity==0
  - standardised logits of the 73 target-encoded keys (fe_keys.encode_keys, inner-fold for training rows)
Network: Linear(in,256) -> 2 residual blocks [LayerNorm, Linear, SiLU, Dropout(0.2), Linear] -> Linear(256,1)
Training: AdamW lr 2e-3, weight decay 1e-4, OneCycle schedule, batch 4096, FIXED 10 epochs (no early
stopping, so the validation fold never picks the epoch), BCEWithLogitsLoss. 3 seeds averaged + full refit.
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler

from common import CATS, DATA, FLAGS, NUMS, OOF_DIR, SUB_DIR, TARGET, load, log_result
from fe_keys import FEATS, build_key_store, encode_keys, safe_logit

warnings.filterwarnings("ignore")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, BATCH, SEEDS = 10, 4096, [42, 137, 2026]
NAME = sys.argv[1] if len(sys.argv) > 1 else "v24_torch_mlp"
print("device:", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "", flush=True)

train, test, y = load()
n = len(train)
knames, kmat, kcnt, ncat, cents = build_key_store(train, test)
full = pd.concat([train[FEATS], test[FEATS]], ignore_index=True)
numc = np.column_stack([full[NUMS].to_numpy(np.float64), cents])
ohe = OneHotEncoder(sparse_output=False, dtype=np.float32).fit_transform(
    np.column_stack([full[CATS].astype(str).to_numpy(), full[FLAGS].astype(int).astype(str).to_numpy(),
                     (full["velocity_score"] == 0).astype(int).astype(str).to_numpy()]))
TEST_ROWS = np.arange(n, n + len(test))


class Block(nn.Module):
    def __init__(self, d, p=0.2):
        super().__init__()
        self.f = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.SiLU(), nn.Dropout(p), nn.Linear(d, d))

    def forward(self, x):
        return x + self.f(x)


def make_net(d_in, d=256):
    return nn.Sequential(nn.Linear(d_in, d), nn.SiLU(), Block(d), Block(d), nn.LayerNorm(d), nn.Linear(d, 1))


def train_predict(Xa, ya, X_evals, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xa_t = torch.tensor(Xa, device=DEV); ya_t = torch.tensor(ya, dtype=torch.float32, device=DEV)
    net = make_net(Xa.shape[1]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = EPOCHS * int(np.ceil(len(Xa) / BATCH))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.2)
    lossf = nn.BCEWithLogitsLoss()
    g = torch.Generator(device=DEV); g.manual_seed(seed)
    for _ in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(Xa), device=DEV, generator=g)
        for i in range(0, len(Xa), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad(set_to_none=True)
            loss = lossf(net(Xa_t[idx]).squeeze(1), ya_t[idx])
            loss.backward(); opt.step(); sched.step()
    net.eval(); outs = []
    with torch.no_grad():
        for Xe in X_evals:
            t = torch.tensor(Xe, device=DEV)
            outs.append(torch.cat([net(t[i:i + 65536]).squeeze(1) for i in range(0, len(t), 65536)]).float().cpu().numpy())
    return outs


def features(tr, Ea, rows_list, E_list):
    qt = QuantileTransformer(n_quantiles=2000, output_distribution="normal", random_state=42).fit(numc[tr])
    sc = StandardScaler().fit(safe_logit(Ea))
    mk = lambda rows, E: np.hstack([qt.transform(numc[rows]), ohe[rows], np.clip(sc.transform(safe_logit(E)), -8, 8)]).astype(np.float32)
    return mk(tr, Ea), [mk(r, E) for r, E in zip(rows_list, E_list)]


folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n), y))
oof = np.zeros(n); test_logit = np.zeros(len(test)); t0 = time.time()
for k, (tr, va) in enumerate(folds):
    Ea, Eb, Et = encode_keys(kmat, ncat, y, n, tr, va, smooth=20.0, teseed=0)
    Xa, (Xb, Xt) = features(tr, Ea, [va, TEST_ROWS], [Eb, Et])
    vb = np.zeros(len(va)); vt = np.zeros(len(test))
    for s in SEEDS:
        ob, ot = train_predict(Xa, y[tr], [Xb, Xt], s)
        vb += ob / len(SEEDS); vt += ot / len(SEEDS)
        print(f"  fold {k + 1} seed {s}: AUC {roc_auc_score(y[va], ob):.5f}  {time.time() - t0:.0f}s", flush=True)
    oof[va] = 1 / (1 + np.exp(-vb)); test_logit += vt / 5
    print(f"  fold {k + 1}: AUC {roc_auc_score(y[va], oof[va]):.5f}", flush=True)
fa = [roc_auc_score(y[va], oof[va]) for _, va in folds]
auc = roc_auc_score(y, oof)
print(f"{NAME}: CV AUC {auc:.5f} | folds {np.round(fa, 5)} | {time.time() - t0:.0f}s", flush=True)
np.save(os.path.join(OOF_DIR, f"{NAME}_oof.npy"), oof); np.save(os.path.join(OOF_DIR, f"{NAME}_test.npy"), 1 / (1 + np.exp(-test_logit)))

# full-data refit (same fixed epochs)
Ea, _, Et = encode_keys(kmat, ncat, y, n, None, None, smooth=20.0, teseed=0)
Xa, (Xt,) = features(np.arange(n), Ea, [TEST_ROWS], [Et])
ft = np.mean([train_predict(Xa, y, [Xt], s)[0] for s in SEEDS], axis=0)
full_p = 1 / (1 + np.exp(-ft)); np.save(os.path.join(OOF_DIR, f"{NAME}_full.npy"), full_p)
ids = pd.read_csv(os.path.join(DATA, "test.csv"), usecols=["id"])["id"]
pd.DataFrame({"id": ids, TARGET: full_p}).to_csv(os.path.join(SUB_DIR, f"{NAME}.csv"), index=False)
log_result(NAME, auc, np.std(fa), time.time() - t0, f"PyTorch residual MLP on GPU ({DEV}), 73 TE logits + quantile nums + one-hots, fixed {EPOCHS} epochs, 3 seeds, full refit")
print("done", flush=True)
