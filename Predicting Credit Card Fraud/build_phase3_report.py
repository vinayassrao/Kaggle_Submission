"""Builds PHASE3_REPORT.md from artifacts on disk (no training)."""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import HERE, OOF_DIR, SEED, load

train, test, y = load()
f5 = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
f10 = list(StratifiedKFold(10, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
res = pd.read_csv(os.path.join(HERE, "results.csv")).set_index("iteration")
L = lambda m: np.load(os.path.join(OOF_DIR, f"{m}_oof.npy"))
fold5 = lambda o: np.array([roc_auc_score(y[va], o[va]) for _, va in f5])

MODELS = [  # key, label, plan expectation, iterations / notes
    ("v17_glm_lgb10", "v17 (Phase 2 reference) 10-fold GLM-boost LGB, m=20, 2 seeds", "", "mean 1245 trees; refit 1382×2"),
    ("v23_stack", "v23 (Phase 2 final stack)", "", "n/a"),
    ("v17b_glm_lgb10_s100", "**v17b** 10-fold GLM-boost LGB, **m=100**, 2 seeds", "(part of P3.3)", "mean 1002 trees (96–3678); refit 1113×2"),
    ("v17b_glm_lgb10_s100_glm", "v17b GLM alone (m=100)", "", "newton-cholesky"),
    ("v28_glm_xgb10_gpu", "**v28** 10-fold GLM-boost XGB on **CUDA**, 4 seeds (2× m20 depth-2, 2× m100 lossguide 4 leaves)", "0.95135–0.95145", "mean 987 trees (77–2152); refit 1096×4"),
    ("v28_glm_xgb10_gpu_glm", "v28 GLM alone (avg of m20/m100 encodings)", "", "newton-cholesky"),
    ("v19_sparse_glm_10f", "**v29** 10-fold joint sparse GLM (C=0.01)", "~0.95030", "lbfgs n_iter 143–174"),
    ("v24_torch_mlp", "v24 PyTorch residual MLP on 73 TE logits (GPU)", "(Phase 2 extra)", "fixed 10 epochs, 3 seeds"),
    ("v26_realmlp", "**v26** RealMLP (pytabkit, GPU), no TE", "~0.9500", "n_ens 8, 16 epochs, early stop on 5% inner holdout"),
    ("v27_digit_cnn", "**v27** 1-D char/digit CNN (GPU), no TE", "~0.9495", "best epochs 4–8 of 12 (inner holdout); refit 7 epochs"),
    ("v30_stack", "**v30 final stack**", "0.9520–0.9523+", "LogReg C=0.05 on native-fold-standardised rank-logits"),
]
v17f, v23f = fold5(L("v17_glm_lgb10")), fold5(L("v23_stack"))
rows = []
for key, lab, exp, it in MODELS:
    o = L(key); f = fold5(o)
    secs = res.loc[key, "seconds"] if key in res.index else np.nan
    rows.append({"model": lab, "concat_auc": roc_auc_score(y, o), "mean_fold_auc": f.mean(), **{f"f{i}": f[i] for i in range(5)},
                 "beats_v17": f"{(f > v17f).sum()}/5", "beats_v23": f"{(f > v23f).sum()}/5",
                 "pr_auc": average_precision_score(y, o), "plan_expected": exp, "iterations / notes": it,
                 "runtime_s": "" if pd.isna(secs) or secs == 0 else int(secs)})
T = pd.DataFrame(rows)
ten = {m: np.round([roc_auc_score(y[va], L(m)[va]) for _, va in f10], 5).tolist()
       for m in ["v17_glm_lgb10", "v17b_glm_lgb10_s100", "v28_glm_xgb10_gpu", "v19_sparse_glm_10f"]}

md = []
w = md.append
w("# Phase 3 implementation report: Kaggle Playground S6E11\n")
w("Reply to the \"Phase 3 Review, Q&A & GPU Breakthrough Runbook\". Hardware: 16-core CPU + **RTX 3050 Laptop GPU (6 GB)**. PyTorch 2.11 + CUDA 12.8, "
  "transformers 5.23 and pytabkit are installed in a venv on E:. Every heavy job ran **strictly one at a time**. Base folds: `StratifiedKFold(5, shuffle=True, random_state=42)`. "
  "The 10-fold models use `StratifiedKFold(10, …, 42)`. The fold columns below score every model's OOF on the 5-fold seed-42 split, for comparability. All numbers come from OOF arrays on disk.\n")
w("""## Headline

- **Final v30 stack: nested CV AUC 0.95166** (mean over stacker splits 43/44/45; LogReg C=0.03 and C=0.05 tie). Phase 2 v23 was 0.95156, so **+0.00010**. The runbook expected 0.9520–0.9523+: **not reached** (short by ~0.0003–0.0006).
- **Best single model: v17b (10-fold GLM-boost LGB, smoothing 100) 0.95152**, then v28 (10-fold GPU XGB) 0.95150 and v17 0.95149. These three are statistically tied.
- **P3.1 GPT-2 generative classifier (v25): not run.** A measured benchmark gives **~114 h** for the cheapest useful configuration on the RTX 3050, against the runbook's ~40 min estimate (§3).
- **P3.2 neural/digit models came in well below the runbook's solo estimates:** RealMLP 0.94747 (expected ~0.9500), digit CNN 0.94783 (expected ~0.9495). But the **digit CNN earns stack weight +0.159**, so its "orthogonal errors" claim holds. RealMLP gets ≈ 0.
- **P3.3 10-fold upgrades matched the runbook's estimates.** m=100 encodings earn independent stack weight alongside m=20, confirming Q4.4.
- **No public LB submissions yet** (the user will submit `submissions/v30_stack.csv`).
""")
w("\n## 1. Master score table\n")
w(T.round(5).to_markdown(index=False) + "\n")
w("\n**10-fold models on their native 10 folds:**\n\n" + "\n".join(f"- `{m}`: {v} (mean {np.mean(v):.5f})" for m, v in ten.items()) + "\n")
w("\nNotes: the 'GLM alone' rows are the newton-cholesky GLM init_score OOF, averaged over that model's encoding seeds. The v23/v30 rows are the nested stacker OOF on split 43. Runtime = wall clock for CV + refit.\n")

w("""
## 2. Runbook expectations vs measurements

| Runbook item | Expected | Measured | Verdict |
|---|---|---|---|
| P3.3 v17 + extra encoding seeds at m=100 (run as separate member v17b) | ~0.95156 | **0.95152** | ✅ |
| Q4.4 smoothing=100 view beats / complements smoothing=20 | m100 0.95067 vs m20 0.95063 (GLM), separate weight | GLM alone: m100 **0.95091** vs m20 0.95087 (10-fold). Stack coef v17b 0.383 vs v17 0.321. Hill climb 50/50 v17+v17b | ✅ |
| P3.3 v28 10-fold GPU XGB | 0.95135–0.95145, ~4–5 min | **0.95150**, **17.4 min** for the 10-fold CV (40 fits) + 1.75 min refit = 19.2 min | ✅ score, ❌ speed (~3.5–4× slower than claimed) |
| P3.3 v29 10-fold sparse GLM | ~0.9503+ | **0.95035** | ✅ |
| P3.2 v26 RealMLP | ~0.9500, stack +0.00015–0.00025 | **0.94747**, stack coef −0.027 | ❌ |
| P3.2 v27 digit CNN | ~0.9495, corr ~0.978, stack weight ~0.24 | **0.94783**, Spearman with v17 **0.961** (more diverse than expected), coef **+0.159** | ❌ solo / ✅ stack value |
| P3.1 v25 compact GPT-2 | ~0.9500–0.9505, ~7–9 min/fold, 90 s/epoch | **not run**: 5,398 tok/s → 116 min/epoch; scoring 91 ms/row/order | ❌ infeasible on this GPU |
| P3.4 v30 final stack | 0.9520–0.9523+ | **0.95166** | ❌ (+0.00010 over v23) |
| Zero-fraud zones | +0.000020, 5/5 | **+0.000034, 5/5 folds** | ✅ |
""")

w("""
## 3. v25 GPT-2: benchmark details (why it was not run)

`v25_gpt2.py` implements the runbook's compact target-first format: `<Y0|Y1>` + 24 `<Cj>` column tokens initialised from mean word embeddings, values tokenised once per unique value (`repr(float)` for floats), a fresh random column order per row per step, loss only on field tokens, and LLR = logsumexp over K orders. Rows average **66.6 tokens** (max 70, measured on a 128-row sample).

- **First attempt (batch 128):** CUDA out-of-memory in the first backward pass. Likely cause (not separately profiled): the (128 × 70 × 50,283) vocabulary logits plus their fp32 upcast in the loss and their gradients, together with GPT-2 activations, exceed 6 GB.
- **Fixed:** micro-batch 32 × gradient accumulation 4 (effective batch 128), bf16 autocast, scoring through `F.cross_entropy` in chunks of 64 rows × 2 labels.
- **Measured (`python v25_gpt2.py bench`, 300 optimizer steps on fold-1 training rows):**
  - training **5,398 tokens/s → 115.7 min per epoch** on 559,968 rows
  - scoring **91.4 ms per row per column order** (both labels)
  - 1 epoch + 4 orders: 969 min per fold → **113.6 h** for 5 folds + a full-data test model
  - 3 epochs + 8 orders: **239 h**

The runbook's "560,000 rows/epoch in ~90 s" implies ~400k tokens/s, about 75× what this laptop GPU sustains for GPT-2 124M at sequence length ~67. We did not run v25. GPT-2 weights were downloaded (526 MB, `E:\\envs\\hf-cache`).
""")

w("""
## 4. v30 stack details

- **Members (10):** v17, v17b, v28, v29 (10-fold); v16, v26, v27, v11, v22, v10 (5-fold). Each OOF → `safe_logit` → **standardised within the model's own native outer folds** (10 or 5, seed 42) → global `rank_logit`. Test → `rank_logit` of the full refit (`_full.npy`) where available, else the fold mean (v26, v11, v10).
- **Nested CV on fresh splits 43/44/45:**

| Combiner | 43 / 44 / 45 | mean |
|---|---|---|
| LogReg C=0.005 | 0.95162 / 0.95161 / 0.95160 | 0.95161 |
| LogReg C=0.01 | 0.95164 / 0.95164 / 0.95162 | 0.95163 |
| LogReg C=0.03 | 0.95167 / 0.95166 / 0.95164 | 0.95166 |
| **LogReg C=0.05** (chosen) | 0.95167 / 0.95167 / 0.95165 | **0.95166** |
| Hill climb (non-negative) | 0.95159 / 0.95162 / 0.95159 | 0.95160 |
| mix01 | 0.95165 / 0.95166 / 0.95164 | 0.95165 |

- **Full-data LogReg C=0.01 coefficients:** v17b 0.383, v17 0.321, v28 0.262, v11 0.199, **v27 0.159**, v16 0.116, v22 0.071, v29 −0.009, v26 −0.027, v10 −0.108. **Hill-climb weights:** v17 0.5, v17b 0.5, others 0.
- **Zero-fraud zones:** threshold 13.1 in all 5 seed-42 folds, gain +0.000026 to +0.000048 per fold, **5/5 positive (mean +0.000034)**. Applied to 8,076 test rows.
- Not in the pool: v25 (not run), v24 (tested with the v23 pool: coef 0.004, no gain), v20 and v21 (no gain in Phase 2), GLM-alone OOFs (no test predictions saved).
""")

cm = ["v17_glm_lgb10", "v17b_glm_lgb10_s100", "v28_glm_xgb10_gpu", "v19_sparse_glm_10f", "v16_glm_lgb", "v24_torch_mlp",
      "v26_realmlp", "v27_digit_cnn", "v11_ebm", "v22_lgb4_raw", "v10_spline_logreg"]
C = spearmanr(np.column_stack([L(m) for m in cm])).correlation
short = ["v17", "v17b", "v28", "v29", "v16", "v24", "v26", "v27", "v11", "v22", "v10"]
w("\n## 5. Diversity (Spearman of OOF predictions)\n\n" + pd.DataFrame(C, index=short, columns=short).round(3).to_markdown() + "\n")

w("""
## 6. Implementation details and deviations

1. **`v16_glm_boost.py` extended:** seeds are now `teseed:modelseed[:smooth[:variant]]`, engine `xgbgpu` = XGBoost `device="cuda"`, variant `lg` = `grow_policy="lossguide", max_depth=0, max_leaves=4`. v28 = `0:42:20:d2,1:137:20:d2,2:2026:100:lg,3:777:100:lg`. v17b = `lgb 10 2:2026:100,3:777:100`. Refit multiplier ×1.11 for 10-fold.
2. **v17b is a separate 2-seed member**, not merged into v17 (per the runbook's suggestion to weight m=20 and m=100 independently).
3. **v29** = `p2_models.py v19 10` (same recipe as v19, 10 outer folds). Its files are named `v19_sparse_glm_10f_*`.
4. **v26 RealMLP:** exactly the runbook's parameters (`n_ens=8, n_epochs=16, batch 512, lr 0.03, lin_cos_log_15, pblrd, plr_sigma 16, plr_hidden 16/8, plr_lr_factor 0.1, [512,256,128], mish, p_drop 0.05, wd 0.02, ls_eps 0.005, embedding_size 6, max_one_hot 18`). Inputs: 13 raw numerics + categorical columns for the 5 cats, 6 flags, 6 cents (0..99) and exact customer_age / card_age_months / txn_count / cvv. **No TE.** 5% stratified inner holdout from the training fold for early stopping. No full refit (test = 5-fold mean). ~3.3 min/fold on GPU.
5. **v27 digit CNN:** fixed-width strings (`amount 08.2f, hours 05.2f, distance 05.2f, balance 08.2f, velocity 04.1f, risk 05.1f`, ints zero-padded, `|` separators) + one token per categorical/flag → **length 70, vocab 60**. Char + position embeddings (dim 32), 1×1 conv to 128 channels, 3 residual Conv1d blocks (k = 3, 5, 3, BatchNorm, GELU), mean+max pooling, MLP head. AdamW 2e-3, OneCycle, batch 2048, fp16, 12 epochs with best epoch chosen on a 5% inner holdout. 2 seeds. Full refit at 7 epochs. **No self-attention layer** (runbook suggested 4-head attention over field slots), and no TE. ~4.7 min per seed per fold.
6. **v30 = v23 + native-fold standardisation + C grid {0.005, 0.01, 0.03, 0.05}.** The stacker still uses the nested-CV-selected LogReg (no Dirichlet/simplex optimiser beyond hill climbing).
7. **Original 20k dataset:** not used (per Phase 2 advice).
""")

w("""
## 7. Saved artifacts

- `oof/`: `v17b_glm_lgb10_s100_{oof,test,full,glm_oof}.npy`, `v28_glm_xgb10_gpu_{oof,test,full,glm_oof}.npy`, `v19_sparse_glm_10f_{oof,test,full}.npy`, `v26_realmlp_{oof,test}.npy`, `v27_digit_cnn_{oof,test,full}.npy`, `v30_stack_{oof,test}.npy`.
- `submissions/v30_stack.csv` (final, zones applied), plus `v17b…csv`, `v28…csv`, `v19_sparse_glm_10f.csv`, `v26_realmlp.csv`, `v27_digit_cnn.csv`.
- Code: `v16_glm_boost.py` (extended), `p2_models.py` (fold-count option), `v25_gpt2.py` (bench / fold / full / collect modes), `v26_realmlp.py`, `v27_digit_cnn.py`, `v30_stack.py`.

## 8. Questions for the next phase

1. With GPT-2 infeasible on this GPU, is there a **cheaper generative/token model** that keeps most of its stack value? For example, a small transformer trained from scratch on the ~10k-token value vocabulary, with a restricted output head (the 50k-vocab softmax dominates the cost).
2. The digit CNN earns weight despite 0.9478 solo. Would **self-attention over field slots**, more seeds, or 10 folds be the best way to raise it?
3. RealMLP got no weight without TE. Is RealMLP **with** the 60 TE logits (5% inner holdout) worth trying, or does it collapse into the GLM-boost family as v24 did?
4. Three tied 10-fold GLM-boost members (v17, v17b, v28) hold ~71% of the stack's total coefficient mass (0.966 of 1.367 at C=0.01). Is more encoding-seed bagging still worthwhile, or is ~0.9517 the practical ceiling of this feature family?
""")
open(os.path.join(HERE, "PHASE3_REPORT.md"), "w", encoding="utf-8").write("\n".join(md))
print("written")
