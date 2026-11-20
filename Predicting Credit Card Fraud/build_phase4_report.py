"""Builds PHASE4_REPORT.md from artifacts on disk (no training)."""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import HERE, OOF_DIR, SEED, load

train, test, y = load()
f5 = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
f10 = list(StratifiedKFold(10, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
L = lambda m: np.load(os.path.join(OOF_DIR, f"{m}_oof.npy"))
fold5 = lambda o: np.array([roc_auc_score(y[va], o[va]) for _, va in f5])
log = open(os.path.join(HERE, "run_v31.log"), encoding="utf-8").read()

rows = []
for key, lab in [("v28_glm_only", "**P4.0 v28_glm_only** (10-fold dense GLM, 4 encoding seeds m20/m100)"),
                 ("v17b_glm_lgb10_s100", "v17b (reference: best single)"), ("v30_stack", "v30 (Phase 3 final)"),
                 ("v31_stack", "**v31 stack**")]:
    o = L(key); f = fold5(o)
    rows.append({"model": lab, "concat_auc": roc_auc_score(y, o), "mean_fold_auc": f.mean(),
                 **{f"f{i}": f[i] for i in range(5)}, "pr_auc": average_precision_score(y, o)})
T = pd.DataFrame(rows)
g10 = np.round([roc_auc_score(y[va], L("v28_glm_only")[va]) for _, va in f10], 5).tolist()
v26 = L("v26_realmlp"); v26f = fold5(v26)

md = []
w = md.append
w("# Phase 4 implementation report: Kaggle Playground S6E11\n")
w("Reply to the \"Phase 4 Runbook\". Local hardware: 16-core CPU + RTX 3050 Laptop GPU (6 GB), PyTorch 2.11 + CUDA 12.8 (venv on E:). "
  "Every heavy job ran **one at a time**. Base folds: `StratifiedKFold(5, shuffle=True, random_state=42)`, and 10-fold models use `StratifiedKFold(10, …, 42)`. "
  "Stacker evaluation: nested CV on fresh splits 43/44/45. All numbers come from OOF arrays and run logs on disk.\n")
w("""## Headline

- **v31 stack: nested CV AUC 0.95171** (v30 0.95166; unrounded gain **+0.00004**). The only new member that passed the gate is **P4.0 v28_glm_only** (gain exactly +0.00004, 4/5 folds).
- **Caveat:** the chosen unconstrained LogReg (C=0.05) gives v28_glm_only a **negative coefficient (−0.474)**. The best **non-negative** L2 stacker scores **0.95158**, below v30. So v31's edge depends on that negative weight. The user will submit **both v30_stack.csv and v31_stack.csv**.
- **P4.1 compact GPT-2:** the vocabulary slice (50,283 → **988**) and GPU batch packing work. Scoring is **~20× faster** (215 rows/s/order), but training is only **1.4× faster** (7,640 tok/s, **90 min/epoch** on the RTX 3050). Projected locally: 19 h (1 epoch, 4 orders) to 53 h (4 epochs, 8 orders). **Not run locally; the other agent is running `kaggle_v25_gpt2.py` on Kaggle 2× T4.**
- **P4.3 v26b RealMLP + key categoricals:** fold-1 pilot **0.94806** < gate 0.9493 → not run in full.
- **P4.4 v27b position-preserving digit CNN:** fold-1 pilot **0.94585** vs v27 0.94573 (+0.00012 < gate +0.0008) → not run in full.
- **No public LB scores yet.**
""")
w("\n## 1. Score table\n")
w(T.round(5).to_markdown(index=False) + "\n")
w(f"\nv28_glm_only on its native 10 folds: {g10} (mean {np.mean(g10):.5f}). The v30/v31 rows are nested stacker OOF on split 43.\n")

w("""
## 2. Checking the runbook's diagnoses (§1 "Direct fixes")

| Runbook claim | What we measured | Verdict |
|---|---|---|
| §1.1 GPT-2 was slow because of the 50k-vocab logits (Windows WDDM paging) and Python batch loops | Slicing the vocab to 988 + GPU packing: **scoring 11 → 215 rows/s/order (~20×)**, **training 5,398 → 7,640 tok/s (1.4×)**. A first bench at batch 128 sat at 100% GPU with 5.7 / 6.0 GB used and had not finished 200 steps after 30+ min (≥ 9 s/step, consistent with paging). We stopped it. Batch 32 runs at 3.86 GB peak. | ✅ for scoring. ❌ for training: the transformer body dominates training cost, not the head |
| §1.2 The 10-fold dense GLM is a free missing member (0.95094) | 0.95094. Stack gain +0.00004 (4/5 folds), but with a **negative** coefficient | ✅ (with caveat) |
| §1.3 v26 RealMLP had (a) a fold-calibration bug and (b) missing key categoricals | (a) concat 0.94747 vs mean-fold 0.94792 confirmed. v30 already standardises each member within its native folds, so this didn't affect the stack. (b) adding all 60 keys as categoricals raised the fold-1 pilot from 0.94659 to **0.94806** (+0.0015), still below the 0.9493 gate | partly ✅ |
| §1.4 v27's global pooling discarded column position | Segment pooling (`AdaptiveAvg/MaxPool1d(24)` → LayerNorm → Dropout 0.25 → Linear(6144, 256)), 3 seeds, test-aligned logits: fold-1 **0.94585** vs v27's **0.94573** | ❌ the pooling was not the bottleneck |
""")

w("""
## 3. P4.1 compact GPT-2: what changed and the benchmark

`kaggle_v25_gpt2.py` (standalone; same file runs locally and on Kaggle):
- Pre-tokenises every unique value once (`repr(float)` for floats, `" " + str(v)` otherwise, each field prefixed by its `<Cj>` token) into a padded int16 tensor `FT [N_all, 24, MAX_F=6]` kept on the GPU. Body length per row is fixed regardless of column order (mean 64.5, max 71 tokens).
- Remaps the **988 used token ids** to 0..987. After `resize_token_embeddings` and initialising `<Y0>`, `<Y1>`, `<Cj>` from mean word embeddings, `wte` is replaced by a 988-row `nn.Embedding`, and `lm_head` is a tied 988-output `Linear`.
- Batch packing on the GPU: `FT[rows[:,None], perms]` → flatten → stable argsort of `(tok < 0)` → gather → `[label] + body + EOS`, with an attention mask from the row lengths.
- Training: AdamW lr 2e-4, wd 0.01, OneCycle (5% warmup, cosine), bf16 autocast, grad-clip 1.0, loss on field tokens only, fresh random column order per row per step. Scoring: LLR = logsumexp over K orders of `log P(row|Y1)` − same for Y0, in chunks of 128 rows × 2 labels.
- Modes: `bench`, `folds --folds 0,2,4`, `kaggle` (spawns 2 processes: GPU0 → folds 0,2,4, GPU1 → folds 1,3), `collect` (aligns each fold's val/test LLR with that fold's **test** mean/std, averages test across folds). Skips folds whose files already exist, so it can resume.

**Local benchmark (RTX 3050, batch 32, 100 steps):**

| Metric | Value |
|---|---|
| Compact vocab V | 988 (from 50,283) |
| Peak VRAM | 3.86 GB |
| Training | 7,640 tok/s → **90.1 min / epoch** (559,968 rows) |
| Scoring | **215 rows/s per order** (both labels) |
| Projection, 1 GPU of this kind | 1 epoch + 4 orders: 227 min/fold (**18.9 h** for 5 folds) · 2 epochs + 8 orders: 37.8 h · 4 epochs + 8 orders: 52.8 h |

Two local problems, both fixed: batch-128 training thrashed GPU memory (stopped; batch 32 used instead), and scoring 1,024 rows × 2 labels per chunk ran out of GPU memory (chunk reduced to 128).
""")

w(f"""
## 4. v31 stack details

- **Base pool** = the 10 v30 members (v17, v17b, v28, v29 at 10 folds; v16, v26, v27, v11, v22, v10 at 5 folds). Each OOF is standardised within its native folds, then `rank_logit`.
- **Gate:** add each candidate alone, require mean nested gain ≥ +0.00004 (LogReg C=0.05) and ≥ 4/5 split-43 folds up.
  - v28_glm_only: nested **0.95171**, gain **+0.00004** (unrounded; exactly at the threshold), folds up **4/5** → kept.
  - v25 (GPT-2), v26b and v27b had no OOF on disk (not run in full).
- **Combiners on the final 11-member pool (splits 43 / 44 / 45):**

| Combiner | 43 / 44 / 45 | mean |
|---|---|---|
| LogReg C=0.005 | 0.95161 / 0.95161 / 0.95160 | 0.95161 |
| LogReg C=0.01 | 0.95165 / 0.95165 / 0.95163 | 0.95164 |
| LogReg C=0.03 | 0.95170 / 0.95170 / 0.95168 | 0.95169 |
| **LogReg C=0.05** (chosen) | 0.95172 / 0.95171 / 0.95169 | **0.95171** |
| Non-negative L2 LR, λ=1e-4 (L-BFGS-B, w ≥ 0) | 0.95162 / 0.95158 / 0.95155 | 0.95158 |
| Non-negative L2 LR, λ=1e-3 | 0.95149 / 0.95148 / 0.95148 | 0.95148 |

- **LogReg C=0.05 coefficients:** v17b 0.726, v17 0.516, v28 0.258, v11 0.194, v27 0.170, v22 0.073, v16 0.047, v26 −0.011, v29 −0.016, v10 −0.115, **v28_glm_only −0.474**.
- **Non-negative weights (λ=1e-4):** v17b 0.317, v17 0.290, v28 0.267, v16 0.177, v27 0.147, v11 0.145, v22 0.012, v29 0.010, v28_glm_only 0.010, v26 0, v10 0.
- **Interpretation:** v17/v17b/v28 = GLM + tree corrections. A negative weight on the plain GLM isolates the tree residual. The non-negative stacker can't do that and loses 0.00013.
- **Zero-fraud zones:** per-fold gains +0.000034, +0.000031, +0.000037, +0.000027, +0.000051 (5/5) → applied.
- **Outputs:** `submissions/v31_stack.csv` (LogReg C=0.05, full-refit test inputs) and `submissions/v31_conservative.csv` (non-negative weights, 5-fold-mean test inputs). Spearman between them on test: **0.99852**.
""")

w("""
## 5. Requests for the agent running `kaggle_v25_gpt2.py` on Kaggle

Please report back with the following, so we can integrate v25 and keep the local repo and beginner's guide accurate:

1. **Code changes:** a diff (or the full file) of every change you made to `kaggle_v25_gpt2.py`, and why (e.g. batch size, scoring chunk, precision, DataParallel/DDP, gradient checkpointing, number of orders, test-scoring strategy). We'll commit the exact version that produced the results.
2. **Run settings:** epochs, orders, batch size, lr, seeds, which folds ran on which GPU, whether test was scored per fold (5-fold mean) or by a full-data model, and the Kaggle accelerator actually used (2× T4 / P100 / TPU).
3. **Speed:** tokens/s, minutes per epoch, scoring rows/s/order on T4, total session hours used, and any OOMs or restarts.
4. **Results:**
   - per-fold validation AUC of the LLR and the `collect` CV AUC (OOF concatenated after test-stat alignment)
   - the fold-0 **order-pooling diagnostics** (AUC with 1, 2, 4, 8 orders)
   - training loss curve or final loss per fold
5. **Files to send back** (we'll place them in `oof/`): `v25_gpt2_oof.npy`, `v25_gpt2_test.npy`, and the per-fold parts `v25_f{k}_val.npy`, `v25_f{k}_test.npy`. Please confirm row order: OOF in `train.csv` row order (0..699,959), test in `test.csv` row order.
6. **Your stack check, if you ran one:** v25's Spearman with v17/v28 and its marginal nested gain on splits 43/44/45 (our gate: ≥ +0.00004, ≥ 4/5 folds up).
7. **Suggestions:**
   - whether more epochs/orders or a second LM (e.g. GPT-Neo-125M) is worth the remaining Kaggle quota
   - whether to keep the unconstrained stacker (v31, negative GLM weight) or prefer the non-negative one for the final submission
   - anything else you'd change before the competition closes
""")

w("""
## 6. Saved artifacts and code

- New code: `p4_glm_only.py`, `kaggle_v25_gpt2.py`, `KAGGLE_RUN_INSTRUCTIONS.md`, `v26b_realmlp_keys.py`, `v27b_digit_cnn_seg.py`, `v31_stack.py`.
- `oof/v28_glm_only_{oof,test,full}.npy`, `oof/v31_stack_{oof,test}.npy`. Submissions: `v28_glm_only.csv`, `v31_stack.csv`, `v31_conservative.csv` (plus `v30_stack.csv` from Phase 3).
- The beginner's guide has a new Part B4 (gates, the negative-weight caveat) and C11 entries (gates, negative vs non-negative stack weights, vocabulary slicing).
""")
open(os.path.join(HERE, "PHASE4_REPORT.md"), "w", encoding="utf-8").write("\n".join(md))
print("written")
