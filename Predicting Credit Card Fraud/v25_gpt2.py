"""v25 - Label-first GPT-2 generative classifier on the GPU (run with E:\\envs\\kaggle\\Scripts\\python.exe).

Each row becomes a compact token sequence: [<Y0|Y1>, <C_j>, value tokens of column j, ...] with the 24
columns in a random order (fresh order per row per step). GPT-2 (124M) is fine-tuned to model
P(fields | label). Score = log P(row | Y1) - log P(row | Y0), averaged (log-sum-exp) over K column orders.
No target encoding. Weights are downloaded to HF_HOME on E:.

Usage:
  python v25_gpt2.py bench               measure train/score throughput and print time estimates
  python v25_gpt2.py fold K EPOCHS ORDERS run outer fold K (1..5): train, score its validation rows -> parts/
  python v25_gpt2.py full EPOCHS ORDERS   train on all training rows and score the test set -> parts/
  python v25_gpt2.py collect              assemble OOF + test from parts/, log the CV score
"""
import os

os.environ.setdefault("HF_HOME", r"E:\envs\hf-cache")
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import DATA, HERE, OOF_DIR, SUB_DIR, TARGET, load, log_result

warnings.filterwarnings("ignore")
torch.backends.cuda.matmul.allow_tf32 = True
PARTS = os.path.join(HERE, "parts"); os.makedirs(PARTS, exist_ok=True)
FEATS = ["amount_usd", "hours_since_last_txn", "txn_count_last_24h", "distance_from_home_km", "card_age_months",
         "customer_age", "account_balance_usd", "cvv_retry_count", "velocity_score", "time_of_day_hour", "day_of_week",
         "merchant_risk_score", "prior_disputes", "merchant_category", "card_type", "auth_method", "channel",
         "device_type", "is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch",
         "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]
DEC = {"amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd", "velocity_score", "merchant_risk_score"}
BATCH, ACCUM, LR = 32, 4, 1.5e-4   # micro-batch 32 x 4 accumulation = effective batch 128 (fits 6 GB)
AMP = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

train, test, y = load()
n = len(train)
both = pd.concat([train[FEATS], test[FEATS]], ignore_index=True)
tok = AutoTokenizer.from_pretrained("gpt2")
tok.add_special_tokens({"additional_special_tokens": ["<Y0>", "<Y1>"] + [f"<C{j}>" for j in range(len(FEATS))]})
Y_IDS = (tok.convert_tokens_to_ids("<Y0>"), tok.convert_tokens_to_ids("<Y1>"))
EOS = tok.eos_token_id
# per column: code of every row + padded token fragment per unique value ([<Cj>, value tokens...])
CODES, FRAG, FLEN = [], [], []
for j, c in enumerate(FEATS):
    s = pd.Series([repr(float(v)) for v in both[c].to_numpy()]) if c in DEC else both[c].astype(str)
    codes, uniq = pd.factorize(s, sort=False)
    seqs = [[tok.convert_tokens_to_ids(f"<C{j}>")] + tok.encode(" " + str(u), add_special_tokens=False) for u in uniq]
    m = max(len(q) for q in seqs)
    arr = np.full((len(uniq), m), -1, dtype=np.int64); ln = np.array([len(q) for q in seqs])
    for i, q in enumerate(seqs):
        arr[i, :len(q)] = q
    CODES.append(codes.astype(np.int64)); FRAG.append(arr); FLEN.append(ln)
NC = len(FEATS)


def batch(rows, labels, orders):
    """Vectorised assembly: label token + fields in the given per-row order, right-padded with EOS."""
    B = len(rows)
    pieces = [np.full((B, 1), 0, np.int64)]
    pieces[0][:, 0] = np.where(labels == 1, Y_IDS[1], Y_IDS[0])
    for pos in range(NC):
        j_per_row = orders[:, pos]
        width = max(FRAG[j].shape[1] for j in np.unique(j_per_row))
        blk = np.full((B, width), -1, np.int64)
        for j in np.unique(j_per_row):
            sel = np.where(j_per_row == j)[0]
            f = FRAG[j][CODES[j][rows[sel]]]
            blk[sel, :f.shape[1]] = f
        pieces.append(blk)
    pieces.append(np.full((B, 1), EOS, np.int64))
    M = np.concatenate(pieces, 1)
    valid = M >= 0
    L = valid.sum(1).max()
    ids = np.full((B, L), EOS, np.int64); att = np.zeros((B, L), np.int64)
    for b in range(B):                                   # compact each row (remove -1 gaps)
        v = M[b][valid[b]]; ids[b, :len(v)] = v; att[b, :len(v)] = 1
    return torch.as_tensor(ids), torch.as_tensor(att)


def new_model():
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tok))
    with torch.no_grad():
        E = model.transformer.wte.weight
        E[Y_IDS[0]] = E[tok.encode(" is fraud False", add_special_tokens=False)].mean(0)
        E[Y_IDS[1]] = E[tok.encode(" is fraud True", add_special_tokens=False)].mean(0)
        for j, c in enumerate(FEATS):
            E[tok.convert_tokens_to_ids(f"<C{j}>")] = E[tok.encode(" " + c.replace("_", " "), add_special_tokens=False)].mean(0)
    return model.cuda()


def train_model(rows, epochs, seed=0, max_steps=None):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    model = new_model(); model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = max_steps or epochs * int(np.ceil(len(rows) / (BATCH * ACCUM)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.05, anneal_strategy="cos")
    t0, step, ntok, micro = time.time(), 0, 0, 0
    opt.zero_grad(set_to_none=True)
    while step < steps:
        perm = rng.permutation(rows)
        for i in range(0, len(perm), BATCH):
            r = perm[i:i + BATCH]
            orders = np.argsort(rng.random((len(r), NC)), axis=1)
            ids, att = batch(r, y[r], orders)
            ids, att = ids.cuda(non_blocking=True), att.cuda(non_blocking=True)
            labels = ids.clone(); labels[att == 0] = -100; labels[:, 0] = -100   # loss only on fields | label
            with torch.autocast("cuda", dtype=AMP):
                loss = model(input_ids=ids, attention_mask=att, labels=labels).loss / ACCUM
            loss.backward(); micro += 1; ntok += int(att.sum())
            if micro % ACCUM:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            opt.zero_grad(set_to_none=True); step += 1
            if step % 500 == 0:
                print(f"    step {step}/{steps} loss {loss.item() * ACCUM:.4f} {ntok / (time.time() - t0):.0f} tok/s", flush=True)
            if step >= steps:
                break
    return model, ntok / (time.time() - t0)


@torch.inference_mode()
def score(model, rows, n_orders, bs=64, seed=2026):
    model.eval(); rng = np.random.default_rng(seed)
    orders_list = [rng.permutation(NC) for _ in range(n_orders)]
    ll = np.zeros((len(rows), n_orders, 2))
    for oi, order in enumerate(orders_list):
        for s0 in range(0, len(rows), bs):
            r = rows[s0:s0 + bs]; m = len(r)
            rr = np.concatenate([r, r]); labs = np.r_[np.zeros(m, int), np.ones(m, int)]
            ids, att = batch(rr, labs, np.broadcast_to(order, (2 * m, NC)))
            ids, att = ids.cuda(), att.cuda()
            with torch.autocast("cuda", dtype=AMP):
                h = model.transformer(input_ids=ids, attention_mask=att).last_hidden_state
            logits = model.lm_head(h[:, :-1])                                 # (2m, L-1, V) in half precision
            nll = F.cross_entropy(logits.float().transpose(1, 2), ids[:, 1:], reduction="none")
            tot = (-(nll * att[:, 1:])).double().sum(1).cpu().numpy()
            ll[s0:s0 + m, oi, 0], ll[s0:s0 + m, oi, 1] = tot[:m], tot[m:]
    return logsumexp(ll[:, :, 1], 1) - logsumexp(ll[:, :, 0], 1)


folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n), y))
mode = sys.argv[1]
if mode == "bench":
    tr, va = folds[0]
    t = time.time(); model, tps = train_model(tr, 1, max_steps=300); tt = time.time() - t
    sec_per_epoch = tt / 300 * np.ceil(len(tr) / (BATCH * ACCUM))
    t = time.time(); _ = score(model, va[:4096], 1); st = time.time() - t
    sec_score_row_order = st / 4096
    print(f"train: {tps:.0f} tokens/s -> {sec_per_epoch / 60:.1f} min per epoch on {len(tr):,} rows")
    print(f"score: {sec_score_row_order * 1e3:.2f} ms per row per order (both labels)")
    for ep, K in [(1, 4), (2, 4), (3, 8)]:
        per_fold = ep * sec_per_epoch + K * sec_score_row_order * len(va)
        full = ep * sec_per_epoch * n / len(tr) + K * sec_score_row_order * len(test)
        print(f"  epochs {ep}, orders {K}: {per_fold / 60:.0f} min per fold, 5 folds + full-data test model = {(5 * per_fold + full) / 3600:.1f} h")
elif mode == "fold":
    k, ep, K = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    tr, va = folds[k - 1]
    model, tps = train_model(tr, ep, seed=k)
    s = score(model, va, K)
    np.save(os.path.join(PARTS, f"v25_fold{k}.npy"), s)
    print(f"  fold {k}: AUC {roc_auc_score(y[va], s):.5f} ({tps:.0f} tok/s)", flush=True)
elif mode == "full":
    ep, K = int(sys.argv[2]), int(sys.argv[3])
    model, _ = train_model(np.arange(n), ep, seed=99)
    np.save(os.path.join(PARTS, "v25_test.npy"), score(model, np.arange(n, n + len(test)), K))
    print("  test scored", flush=True)
elif mode == "collect":
    oof = np.zeros(n)
    for k, (_, va) in enumerate(folds, 1):
        s = np.load(os.path.join(PARTS, f"v25_fold{k}.npy")); oof[va] = (s - s.mean()) / s.std()
    fa = [roc_auc_score(y[va], oof[va]) for _, va in folds]; auc = roc_auc_score(y, oof)
    sig = lambda z: 1 / (1 + np.exp(-z))
    t = np.load(os.path.join(PARTS, "v25_test.npy")); t = (t - t.mean()) / t.std()
    np.save(os.path.join(OOF_DIR, "v25_gpt2_oof.npy"), sig(oof)); np.save(os.path.join(OOF_DIR, "v25_gpt2_full.npy"), sig(t))
    ids = pd.read_csv(os.path.join(DATA, "test.csv"), usecols=["id"])["id"]
    pd.DataFrame({"id": ids, TARGET: sig(t)}).to_csv(os.path.join(SUB_DIR, "v25_gpt2.csv"), index=False)
    log_result("v25_gpt2", auc, np.std(fa), 0, "Label-first GPT-2 generative classifier (log-likelihood ratio), GPU, no TE")
    print(f"v25_gpt2: CV AUC {auc:.5f} | folds {np.round(fa, 5)}")
