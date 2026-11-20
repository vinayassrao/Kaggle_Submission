"""v27 - 1-D character/digit CNN on the GPU, with NO target encoding (run with E:\\envs\\kaggle\\Scripts\\python.exe).

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
NAME, EPOCHS, BATCH, SEEDS = "v27_digit_cnn", 12, 2048, [42, 137]
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
        self.head = nn.Sequential(nn.Linear(2 * c, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, x):
        h = self.tok(x) + self.pos(torch.arange(L, device=x.device))[None]
        h = self.blocks(self.inp(h.transpose(1, 2)))
        return self.head(torch.cat([h.mean(2), h.amax(2)], 1)).squeeze(1)


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
        zv += std(pv) / len(SEEDS); zt += std(pt) / len(SEEDS)
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
log_result(NAME, auc, np.std(fa), time.time() - t0, f"1-D char/digit CNN on GPU, no target encoding, 2 seeds, 5% inner holdout for epoch, full refit {ne} epochs")
print("done", flush=True)
