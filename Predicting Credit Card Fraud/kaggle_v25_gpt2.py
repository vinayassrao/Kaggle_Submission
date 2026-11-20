"""v25 - Label-first GPT-2 generative classifier, compact-vocabulary version. Standalone: runs locally or on Kaggle.

Each row -> [<Y0|Y1>, <C_j>, value tokens of column j, ... (24 columns, random order), EOS].
GPT-2 (124M) is fine-tuned to model P(fields | label); score = log P(row | Y1) - log P(row | Y0), pooled with
log-sum-exp over K column orders. No target encoding.

Speed-ups vs the first version:
  * vocabulary sliced to the ~hundreds of token ids the data actually uses (embedding + tied LM head), so the
    output softmax is tiny instead of 50,283 wide;
  * every field pre-tokenised once into a padded int16 tensor [N, 24, MAX_F] on the GPU; batches with random
    column orders are packed with tensor ops (gather + stable argsort), no Python loops.

Usage:
  python kaggle_v25_gpt2.py bench   [--data DIR] [--out DIR]
  python kaggle_v25_gpt2.py folds   --folds 0,2,4 [--epochs 4 --orders 8 --batch 128]   (writes v25_f{k}_val.npy / _test.npy)
  python kaggle_v25_gpt2.py kaggle  [--epochs 4 --orders 8]  (2 GPUs: launches folds 0,2,4 on GPU0 and 1,3 on GPU1)
  python kaggle_v25_gpt2.py collect [--data DIR] [--out DIR] (assemble v25_gpt2_oof.npy / v25_gpt2_test.npy)
Defaults: --data = auto-discovered under /kaggle/input if it exists, else ./data ; --out = /kaggle/working or ./parts
"""
import argparse
import os
import subprocess
import sys
import time

from pathlib import Path
import numpy as np
import pandas as pd


def _default_data_dir():
    if os.path.isdir("/kaggle/input"):
        hit = next(Path("/kaggle/input").rglob("train.csv"), None)
        if hit is not None:
            return str(hit.parent)
    return "data"


P = argparse.ArgumentParser()
P.add_argument("mode", choices=["bench", "folds", "kaggle", "collect"])
P.add_argument("--data", default=_default_data_dir())
P.add_argument("--out", default="/kaggle/working" if os.path.isdir("/kaggle/working") else "parts")
P.add_argument("--folds", default="0,1,2,3,4")
P.add_argument("--epochs", type=int, default=4)
P.add_argument("--orders", type=int, default=8)
P.add_argument("--batch", type=int, default=128)
P.add_argument("--lr", type=float, default=2e-4)
P.add_argument("--steps", type=int, default=200, help="bench: training steps to time")
A = P.parse_args()
os.makedirs(A.out, exist_ok=True)
FEATS = ["amount_usd", "hours_since_last_txn", "txn_count_last_24h", "distance_from_home_km", "card_age_months",
         "customer_age", "account_balance_usd", "cvv_retry_count", "velocity_score", "time_of_day_hour", "day_of_week",
         "merchant_risk_score", "prior_disputes", "merchant_category", "card_type", "auth_method", "channel",
         "device_type", "is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch",
         "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]
DEC = {"amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd", "velocity_score", "merchant_risk_score"}
NC = len(FEATS)
train = pd.read_csv(os.path.join(A.data, "train.csv"))
test = pd.read_csv(os.path.join(A.data, "test.csv"))
y = train["is_fraud"].astype(int).to_numpy()
n, nt = len(train), len(test)


def folds5():
    from sklearn.model_selection import StratifiedKFold
    return list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n), y))


if A.mode == "kaggle":   # pre-cache HF weights once, then spawn one worker per available GPU
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    AutoTokenizer.from_pretrained("gpt2")
    AutoModelForCausalLM.from_pretrained("gpt2")
    plan = [(0, "0,2,4"), (1, "1,3")] if torch.cuda.device_count() >= 2 else [(0, A.folds)]
    procs = []
    for gpu, fl in plan:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        cmd = [sys.executable, __file__, "folds", "--folds", fl, "--epochs", str(A.epochs), "--orders", str(A.orders),
               "--batch", str(A.batch), "--lr", str(A.lr), "--data", A.data, "--out", A.out]
        procs.append(subprocess.Popen(cmd, env=env))
    sys.exit(max(p.wait() for p in procs))

if A.mode == "collect":
    from sklearn.metrics import roc_auc_score
    oof = np.zeros(n); tz = np.zeros(nt)
    for k, (_, va) in enumerate(folds5()):
        v = np.load(os.path.join(A.out, f"v25_f{k}_val.npy")); t = np.load(os.path.join(A.out, f"v25_f{k}_test.npy"))
        mu, sd = t.mean(), t.std()                      # label-free alignment: standardise with the fold's TEST stats
        oof[va] = (v - mu) / sd; tz += (t - mu) / sd / 5
    sig = lambda z: 1 / (1 + np.exp(-z))
    print("v25_gpt2 CV AUC", round(roc_auc_score(y, oof), 5), [round(roc_auc_score(y[va], oof[va]), 5) for _, va in folds5()])
    np.save(os.path.join(A.out, "v25_gpt2_oof.npy"), sig(oof)); np.save(os.path.join(A.out, "v25_gpt2_test.npy"), sig(tz))
    sys.exit(0)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.special import logsumexp  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
DEV = "cuda"
# Use native BF16 only on Ampere+ (sm_80+); Kaggle T4 (sm_75) emulates BF16 slowly and needs FP16 Tensor Cores
AMP = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# ---------------- pre-tokenise every field once ----------------
both = pd.concat([train[FEATS], test[FEATS]], ignore_index=True)
tok = AutoTokenizer.from_pretrained("gpt2")
tok.add_special_tokens({"additional_special_tokens": ["<Y0>", "<Y1>"] + [f"<C{j}>" for j in range(NC)]})
frag = []
for j, c in enumerate(FEATS):
    s = pd.Series([repr(float(v)) for v in both[c].to_numpy()]) if c in DEC else both[c].astype(str)
    codes, uniq = pd.factorize(s, sort=False)
    seqs = [[tok.convert_tokens_to_ids(f"<C{j}>")] + tok.encode(" " + str(u), add_special_tokens=False) for u in uniq]
    frag.append((codes, seqs))
Y0, Y1, EOS = tok.convert_tokens_to_ids("<Y0>"), tok.convert_tokens_to_ids("<Y1>"), tok.eos_token_id
used = sorted({t for _, seqs in frag for q in seqs for t in q} | {Y0, Y1, EOS})
remap = {t: i for i, t in enumerate(used)}
V = len(used)
MAX_F = max(len(q) for _, seqs in frag for q in seqs)
FT = np.full((len(both), NC, MAX_F), -1, dtype=np.int16)
for j, (codes, seqs) in enumerate(frag):
    arr = np.full((len(seqs), MAX_F), -1, dtype=np.int16)
    for i, q in enumerate(seqs):
        arr[i, :len(q)] = [remap[t] for t in q]
    FT[:, j, :] = arr[codes]
ROW_LEN = (FT >= 0).sum(axis=(1, 2))                    # body tokens per row (independent of column order)
y0, y1, eos = remap[Y0], remap[Y1], remap[EOS]
FT_G = torch.as_tensor(FT, device=DEV)
LEN_G = torch.as_tensor(ROW_LEN, device=DEV)
print(f"compact vocab V={V} (from {len(tok)}), MAX_F={MAX_F}, body tokens/row mean {ROW_LEN.mean():.1f} max {ROW_LEN.max()}", flush=True)


def pack(rows_g, labels_g, perms_g):
    """rows [B], labels [B] (0/1), perms [B,24] on GPU -> ids [B,L], att [B,L] (valid tokens first)."""
    B = rows_g.shape[0]
    flat = FT_G[rows_g[:, None], perms_g].reshape(B, -1).long()              # [B, 24*MAX_F]
    order = (flat < 0).to(torch.int8).argsort(dim=1, stable=True)
    body_len = int(LEN_G[rows_g].max())
    body = flat.gather(1, order)[:, :body_len]
    lab = torch.where(labels_g == 1, y1, y0)[:, None]
    ids = torch.cat([lab, body, torch.full((B, 1), eos, device=DEV)], 1)
    L = ids.shape[1]
    lens = LEN_G[rows_g] + 2                                                   # label + body + eos
    att = (torch.arange(L, device=DEV)[None] < lens[:, None]).long()
    ids = torch.where(ids < 0, torch.full_like(ids, eos), ids)
    ids = torch.where(att.bool(), ids, torch.full_like(ids, eos))
    eos_pos = (lens - 1)[:, None]                                              # put EOS right after each row's body
    ids.scatter_(1, eos_pos, eos)
    return ids, att


def new_model():
    m = AutoModelForCausalLM.from_pretrained("gpt2")
    m.config.use_cache = False
    for blk in m.transformer.h:
        blk.mlp.act = torch.nn.GELU(approximate="tanh")  # fused CUDA kernel (5x less VRAM than Python NewGELU)
    m.resize_token_embeddings(len(tok))
    with torch.no_grad():
        E = m.transformer.wte.weight
        E[Y0] = E[tok.encode(" is fraud False", add_special_tokens=False)].mean(0)
        E[Y1] = E[tok.encode(" is fraud True", add_special_tokens=False)].mean(0)
        for j, c in enumerate(FEATS):
            E[tok.convert_tokens_to_ids(f"<C{j}>")] = E[tok.encode(" " + c.replace("_", " "), add_special_tokens=False)].mean(0)
        W = E[torch.as_tensor(used)].clone()
    emb = torch.nn.Embedding(V, W.shape[1]); emb.weight.data.copy_(W)
    m.transformer.wte = emb
    m.lm_head = torch.nn.Linear(W.shape[1], V, bias=False); m.lm_head.weight = m.transformer.wte.weight
    m.config.vocab_size = V
    return m.to(DEV)


def train_model(rows, epochs, seed, max_steps=None):
    g = torch.Generator(device=DEV); g.manual_seed(seed); torch.manual_seed(seed)
    m = new_model(); m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=A.lr, weight_decay=0.01)
    steps = max_steps or epochs * int(np.ceil(len(rows) / A.batch))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=A.lr, total_steps=steps, pct_start=0.05, anneal_strategy="cos")
    rows_g = torch.as_tensor(rows, device=DEV); yg = torch.as_tensor(y, device=DEV)
    use_scaler = AMP == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    t0, step, ntok = time.time(), 0, 0
    while step < steps:
        perm = rows_g[torch.randperm(len(rows_g), device=DEV, generator=g)]
        for i in range(0, len(perm), A.batch):
            r = perm[i:i + A.batch]
            perms = torch.rand((len(r), NC), device=DEV, generator=g).argsort(1)
            ids, att = pack(r, yg[r], perms)
            labels = ids.masked_fill(att == 0, -100); labels[:, 0] = -100
            with torch.autocast("cuda", dtype=AMP):
                loss = m(input_ids=ids, attention_mask=att, labels=labels).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            scale_before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if not use_scaler or scaler.get_scale() >= scale_before:
                sch.step()
            step += 1; ntok += int(att.sum())
            if step % 500 == 0:
                print(f"    step {step}/{steps} loss {loss.item():.4f} {ntok / (time.time() - t0):.0f} tok/s", flush=True)
            if step >= steps:
                break
    del opt, sch, scaler
    torch.cuda.empty_cache()
    return m, ntok / (time.time() - t0)


@torch.inference_mode()
def score(m, rows, n_orders, bs=256, seed=2026):
    """Returns LLR [len(rows)] pooled over n_orders, plus per-order cumulative LLRs (for order-pooling diagnostics)."""
    m.eval(); rng = np.random.default_rng(seed)
    ll = np.zeros((len(rows), n_orders, 2))
    for oi in range(n_orders):
        order = torch.as_tensor(rng.permutation(NC), device=DEV)
        for s0 in range(0, len(rows), bs):
            r = torch.as_tensor(rows[s0:s0 + bs], device=DEV); b = len(r)
            rr = torch.cat([r, r]); labs = torch.cat([torch.zeros(b, device=DEV, dtype=torch.long), torch.ones(b, device=DEV, dtype=torch.long)])
            ids, att = pack(rr, labs, order[None].expand(2 * b, NC))
            with torch.autocast("cuda", dtype=AMP):
                logits = m(input_ids=ids, attention_mask=att, use_cache=False).logits[:, :-1]
            nll = F.cross_entropy(logits.float().transpose(1, 2), ids[:, 1:], reduction="none")
            tot = (-(nll * att[:, 1:])).double().sum(1).cpu().numpy()
            ll[s0:s0 + b, oi, 0], ll[s0:s0 + b, oi, 1] = tot[:b], tot[b:]
    return logsumexp(ll[:, :, 1], 1) - logsumexp(ll[:, :, 0], 1), ll


if A.mode == "bench":
    tr, va = folds5()[0]
    torch.cuda.reset_peak_memory_stats()
    t = time.time(); m, tps = train_model(tr, 1, 0, max_steps=A.steps); tt = time.time() - t
    sec_epoch = tt / A.steps * np.ceil(len(tr) / A.batch)
    t = time.time(); _ = score(m, va[:2000], 1); ts = (time.time() - t) / 2000
    print(f"V={V}  peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"train {tps:.0f} tok/s -> {sec_epoch / 60:.1f} min/epoch ({len(tr):,} rows, batch {A.batch})")
    print(f"score {1 / ts:.0f} rows/s per order (both labels)")
    for ep, K in [(1, 4), (2, 8), (4, 8)]:
        per_fold = ep * sec_epoch + K * ts * (len(va) + nt)
        print(f"  epochs {ep}, orders {K}: {per_fold / 60:.0f} min/fold, 5 folds = {5 * per_fold / 3600:.1f} h on one GPU of this kind")
    sys.exit(0)

if A.mode == "folds":
    from sklearn.metrics import roc_auc_score
    F5 = folds5(); TEST = np.arange(n, n + nt)
    for k in [int(x) for x in A.folds.split(",")]:
        if os.path.exists(os.path.join(A.out, f"v25_f{k}_test.npy")):
            print(f"fold {k} already done, skipping", flush=True); continue
        tr, va = F5[k]
        t0 = time.time(); m, tps = train_model(tr, A.epochs, seed=k)
        sv, llv = score(m, va, A.orders)
        np.save(os.path.join(A.out, f"v25_f{k}_val.npy"), sv)
        if k == 0:                                         # order-pooling diagnostics
            for K in (1, 2, 4, 8):
                if K <= A.orders:
                    z = logsumexp(llv[:, :K, 1], 1) - logsumexp(llv[:, :K, 0], 1)
                    print(f"  fold 0 AUC with {K} orders: {roc_auc_score(y[va], z):.5f}", flush=True)
        st, _ = score(m, TEST, A.orders)
        np.save(os.path.join(A.out, f"v25_f{k}_test.npy"), st)
        print(f"fold {k}: AUC {roc_auc_score(y[va], sv):.5f}  ({tps:.0f} tok/s, {(time.time() - t0) / 60:.0f} min)", flush=True)
        del m
        torch.cuda.empty_cache()
