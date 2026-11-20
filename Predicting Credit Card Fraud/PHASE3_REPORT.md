# Phase 3 implementation report: Kaggle Playground S6E11

Reply to the "Phase 3 Review, Q&A & GPU Breakthrough Runbook". Hardware: 16-core CPU + **RTX 3050 Laptop GPU (6 GB)**. PyTorch 2.11 + CUDA 12.8, transformers 5.23 and pytabkit are installed in a venv on E:. Every heavy job ran **strictly one at a time**. Base folds: `StratifiedKFold(5, shuffle=True, random_state=42)`. The 10-fold models use `StratifiedKFold(10, …, 42)`. The fold columns below score every model's OOF on the 5-fold seed-42 split, for comparability. All numbers come from OOF arrays on disk.

## Headline

- **Final v30 stack: nested CV AUC 0.95166** (mean over stacker splits 43/44/45; LogReg C=0.03 and C=0.05 tie). Phase 2 v23 was 0.95156, so **+0.00010**. The runbook expected 0.9520–0.9523+: **not reached** (short by ~0.0004–0.0007).
- **Best single model: v17b (10-fold GLM-boost LGB, smoothing 100) 0.95152**, then v28 (10-fold GPU XGB) 0.95150 and v17 0.95149. These three are statistically tied.
- **P3.1 GPT-2 generative classifier (v25): not run.** A measured benchmark gives **~114 h** for the cheapest useful configuration on the RTX 3050, against the runbook's ~40 min estimate (§3).
- **P3.2 neural/digit models came in well below the runbook's solo estimates:** RealMLP 0.94747 (expected ~0.9500), digit CNN 0.94783 (expected ~0.9495). But the **digit CNN earns stack weight +0.159**, so its "orthogonal errors" claim holds. RealMLP gets ≈ 0.
- **P3.3 10-fold upgrades matched the runbook's estimates.** m=100 encodings earn independent stack weight alongside m=20, confirming Q4.4.
- **No public LB submissions yet** (the user will submit `submissions/v30_stack.csv`).


## 1. Master score table

| model                                                                                           |   concat_auc |   mean_fold_auc |      f0 |      f1 |      f2 |      f3 |      f4 | beats_v17   | beats_v23   |   pr_auc | plan_expected   | iterations / notes                                    |   runtime_s |
|:------------------------------------------------------------------------------------------------|-------------:|----------------:|--------:|--------:|--------:|--------:|--------:|:------------|:------------|---------:|:----------------|:------------------------------------------------------|------------:|
| v17 (Phase 2 reference) 10-fold GLM-boost LGB, m=20, 2 seeds                                    |      0.95149 |         0.9515  | 0.95068 | 0.95212 | 0.95634 | 0.9519  | 0.94647 | 0/5         | 1/5         |  0.53455 |                 | mean 1245 trees; refit 1382×2                         |        1497 |
| v23 (Phase 2 final stack)                                                                       |      0.95157 |         0.95158 | 0.95071 | 0.9523  | 0.95643 | 0.95188 | 0.94661 | 4/5         | 0/5         |  0.53516 |                 | n/a                                                   |             |
| **v17b** 10-fold GLM-boost LGB, **m=100**, 2 seeds                                              |      0.95152 |         0.95153 | 0.9509  | 0.95217 | 0.95627 | 0.95199 | 0.94634 | 3/5         | 2/5         |  0.53479 | (part of P3.3)  | mean 1002 trees (96–3678); refit 1113×2               |         814 |
| v17b GLM alone (m=100)                                                                          |      0.95091 |         0.95092 | 0.95043 | 0.95151 | 0.95592 | 0.95091 | 0.94581 | 0/5         | 0/5         |  0.5318  |                 | newton-cholesky                                       |             |
| **v28** 10-fold GLM-boost XGB on **CUDA**, 4 seeds (2× m20 depth-2, 2× m100 lossguide 4 leaves) |      0.9515  |         0.95151 | 0.95077 | 0.95219 | 0.95627 | 0.95191 | 0.94642 | 3/5         | 2/5         |  0.53469 | 0.95135–0.95145 | mean 987 trees (77–2152); refit 1096×4                |        1149 |
| v28 GLM alone (avg of m20/m100 encodings)                                                       |      0.95094 |         0.95095 | 0.95035 | 0.95151 | 0.95598 | 0.95101 | 0.94591 | 0/5         | 0/5         |  0.5318  |                 | newton-cholesky                                       |             |
| **v29** 10-fold joint sparse GLM (C=0.01)                                                       |      0.95035 |         0.95036 | 0.94966 | 0.95081 | 0.95551 | 0.9504  | 0.94542 | 0/5         | 0/5         |  0.53018 | ~0.95030        | lbfgs n_iter 143–174                                  |         351 |
| v24 PyTorch residual MLP on 73 TE logits (GPU)                                                  |      0.94863 |         0.94867 | 0.94819 | 0.94934 | 0.95372 | 0.94941 | 0.94269 | 0/5         | 0/5         |  0.51929 | (Phase 2 extra) | fixed 10 epochs, 3 seeds                              |         183 |
| **v26** RealMLP (pytabkit, GPU), no TE                                                          |      0.94747 |         0.94792 | 0.94659 | 0.94912 | 0.95313 | 0.94825 | 0.94253 | 0/5         | 0/5         |  0.515   | ~0.9500         | n_ens 8, 16 epochs, early stop on 5% inner holdout    |         981 |
| **v27** 1-D char/digit CNN (GPU), no TE                                                         |      0.94783 |         0.94783 | 0.94573 | 0.94973 | 0.95262 | 0.94839 | 0.94268 | 0/5         | 0/5         |  0.51675 | ~0.9495         | best epochs 4–8 of 12 (inner holdout); refit 7 epochs |        3277 |
| **v30 final stack**                                                                             |      0.95167 |         0.95167 | 0.95079 | 0.95245 | 0.95646 | 0.95199 | 0.94662 | 5/5         | 5/5         |  0.53572 | 0.9520–0.9523+  | LogReg C=0.05 on native-fold-standardised rank-logits |             |


**10-fold models on their native 10 folds:**

- `v17_glm_lgb10`: [0.95, 0.95137, 0.95278, 0.95146, 0.95173, 0.96103, 0.94784, 0.95596, 0.94247, 0.95049] (mean 0.95151)
- `v17b_glm_lgb10_s100`: [0.95031, 0.9515, 0.95296, 0.95138, 0.95177, 0.96081, 0.94822, 0.95577, 0.94243, 0.95028] (mean 0.95154)
- `v28_glm_xgb10_gpu`: [0.95014, 0.95141, 0.95298, 0.95139, 0.95164, 0.96093, 0.94802, 0.95581, 0.94248, 0.95043] (mean 0.95152)
- `v19_sparse_glm_10f`: [0.94897, 0.95036, 0.95161, 0.95001, 0.95019, 0.96089, 0.94624, 0.95458, 0.9419, 0.94899] (mean 0.95037)


Notes: the 'GLM alone' rows are the newton-cholesky GLM init_score OOF, averaged over that model's encoding seeds. The v23/v30 rows are the nested stacker OOF on split 43. Runtime = wall clock for CV + refit.


## 2. Runbook expectations vs measurements

| Runbook item | Expected | Measured | Verdict |
|---|---|---|---|
| P3.3 v17 + extra encoding seeds at m=100 (run as separate member v17b) | ~0.95156 | **0.95152** | ✅ |
| Q4.4 smoothing=100 view beats / complements smoothing=20 | m100 0.95067 vs m20 0.95063 (GLM), separate weight | GLM alone: m100 **0.95091** vs m20 0.95087 (10-fold). Stack coef v17b 0.383 vs v17 0.321. Hill climb 50/50 v17+v17b | ✅ |
| P3.3 v28 10-fold GPU XGB | 0.95135–0.95145, ~4–5 min | **0.95150**, **17.4 min** for 10 folds × 4 seeds + refit | ✅ score, ❌ speed (×3.5 slower than claimed) |
| P3.3 v29 10-fold sparse GLM | ~0.9503+ | **0.95035** | ✅ |
| P3.2 v26 RealMLP | ~0.9500, stack +0.00015–0.00025 | **0.94747**, stack coef −0.027 | ❌ |
| P3.2 v27 digit CNN | ~0.9495, corr ~0.978, stack weight ~0.24 | **0.94783**, coef **+0.159** | ❌ solo / ✅ stack value |
| P3.1 v25 compact GPT-2 | ~0.9500–0.9505, ~7–9 min/fold, 90 s/epoch | **not run**: 5,398 tok/s → 116 min/epoch; scoring 91 ms/row/order | ❌ infeasible on this GPU |
| P3.4 v30 final stack | 0.9520–0.9523+ | **0.95166** | ❌ (+0.00010 over v23) |
| Zero-fraud zones | +0.000020, 5/5 | **+0.000034, 5/5 folds** | ✅ |


## 3. v25 GPT-2: benchmark details (why it was not run)

`v25_gpt2.py` implements the runbook's compact target-first format: `<Y0|Y1>` + 24 `<Cj>` column tokens initialised from mean word embeddings, values tokenised once per unique value (`repr(float)` for floats), a fresh random column order per row per step, loss only on field tokens, and LLR = logsumexp over K orders. Rows average **66.6 tokens** (max ~70).

- **First attempt (batch 128):** CUDA out-of-memory in the first backward pass. The fp32 upcast of the (128 × 70 × 50,285) logits for the loss already exceeds 6 GB with activations.
- **Fixed:** micro-batch 32 × gradient accumulation 4 (effective batch 128), bf16 autocast, scoring through `F.cross_entropy` in chunks of 64 rows × 2 labels.
- **Measured (`python v25_gpt2.py bench`, 300 optimizer steps on fold-1 training rows):**
  - training **5,398 tokens/s → 115.7 min per epoch** on 559,968 rows
  - scoring **91.4 ms per row per column order** (both labels)
  - 1 epoch + 4 orders: 969 min per fold → **113.6 h** for 5 folds + a full-data test model
  - 3 epochs + 8 orders: **239 h**

The runbook's "560,000 rows/epoch in ~90 s" implies ~400k tokens/s, about 75× what this laptop GPU sustains for GPT-2 124M at sequence length ~67. We did not run v25. GPT-2 weights were downloaded (526 MB, `E:\envs\hf-cache`).


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
- Not in the pool: v25 (not run), v24 (Phase 2 test: weight 0.004), v20 and v21 (no gain in Phase 2), GLM-alone OOFs (no test predictions saved).


## 5. Diversity (Spearman of OOF predictions)

|      |   v17 |   v17b |   v28 |   v29 |   v16 |   v24 |   v26 |   v27 |   v11 |   v22 |   v10 |
|:-----|------:|-------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|
| v17  | 1     |  0.997 | 0.999 | 0.992 | 0.997 | 0.96  | 0.946 | 0.961 | 0.973 | 0.966 | 0.963 |
| v17b | 0.997 |  1     | 0.999 | 0.992 | 0.995 | 0.96  | 0.948 | 0.964 | 0.971 | 0.966 | 0.965 |
| v28  | 0.999 |  0.999 | 1     | 0.993 | 0.997 | 0.961 | 0.947 | 0.964 | 0.973 | 0.967 | 0.965 |
| v29  | 0.992 |  0.992 | 0.993 | 1     | 0.991 | 0.955 | 0.948 | 0.966 | 0.97  | 0.968 | 0.963 |
| v16  | 0.997 |  0.995 | 0.997 | 0.991 | 1     | 0.962 | 0.946 | 0.963 | 0.975 | 0.967 | 0.965 |
| v24  | 0.96  |  0.96  | 0.961 | 0.955 | 0.962 | 1     | 0.933 | 0.937 | 0.942 | 0.939 | 0.939 |
| v26  | 0.946 |  0.948 | 0.947 | 0.948 | 0.946 | 0.933 | 1     | 0.94  | 0.936 | 0.945 | 0.945 |
| v27  | 0.961 |  0.964 | 0.964 | 0.966 | 0.963 | 0.937 | 0.94  | 1     | 0.946 | 0.962 | 0.949 |
| v11  | 0.973 |  0.971 | 0.973 | 0.97  | 0.975 | 0.942 | 0.936 | 0.946 | 1     | 0.965 | 0.967 |
| v22  | 0.966 |  0.966 | 0.967 | 0.968 | 0.967 | 0.939 | 0.945 | 0.962 | 0.965 | 1     | 0.959 |
| v10  | 0.963 |  0.965 | 0.965 | 0.963 | 0.965 | 0.939 | 0.945 | 0.949 | 0.967 | 0.959 | 1     |


## 6. Implementation details and deviations

1. **`v16_glm_boost.py` extended:** seeds are now `teseed:modelseed[:smooth[:variant]]`, engine `xgbgpu` = XGBoost `device="cuda"`, variant `lg` = `grow_policy="lossguide", max_depth=0, max_leaves=4`. v28 = `0:42:20:d2,1:137:20:d2,2:2026:100:lg,3:777:100:lg`. v17b = `lgb 10 2:2026:100,3:777:100`. Refit multiplier ×1.11 for 10-fold.
2. **v17b is a separate 2-seed member**, not merged into v17 (per the runbook's suggestion to weight m=20 and m=100 independently).
3. **v29** = `p2_models.py v19 10` (same recipe as v19, 10 outer folds). Its files are named `v19_sparse_glm_10f_*`.
4. **v26 RealMLP:** exactly the runbook's parameters (`n_ens=8, n_epochs=16, batch 512, lr 0.03, lin_cos_log_15, pblrd, plr_sigma 16, plr_hidden 16/8, plr_lr_factor 0.1, [512,256,128], mish, p_drop 0.05, wd 0.02, ls_eps 0.005, embedding_size 6, max_one_hot 18`). Inputs: 13 raw numerics + categorical columns for the 5 cats, 6 flags, 6 cents (0..99) and exact customer_age / card_age_months / txn_count / cvv. **No TE.** 5% stratified inner holdout from the training fold for early stopping. No full refit (test = 5-fold mean). ~3.3 min/fold on GPU.
5. **v27 digit CNN:** fixed-width strings (`amount 08.2f, hours 05.2f, distance 05.2f, balance 08.2f, velocity 04.1f, risk 05.1f`, ints zero-padded, `|` separators) + one token per categorical/flag → **length 70, vocab 60**. Char + position embeddings (dim 32), 1×1 conv to 128 channels, 3 residual Conv1d blocks (k = 3, 5, 3, BatchNorm, GELU), mean+max pooling, MLP head. AdamW 2e-3, OneCycle, batch 2048, fp16, 12 epochs with best epoch chosen on a 5% inner holdout. 2 seeds. Full refit at 7 epochs. **No self-attention layer** (runbook suggested 4-head attention over field slots), and no TE. ~4.7 min per seed per fold.
6. **v30 = v23 + native-fold standardisation + C grid {0.005, 0.01, 0.03, 0.05}.** The stacker still uses the nested-CV-selected LogReg (no Dirichlet/simplex optimiser beyond hill climbing).
7. **Original 20k dataset:** not used (per Phase 2 advice).


## 7. Saved artifacts

- `oof/`: `v17b_glm_lgb10_s100_{oof,test,full,glm_oof}.npy`, `v28_glm_xgb10_gpu_{oof,test,full,glm_oof}.npy`, `v19_sparse_glm_10f_{oof,test,full}.npy`, `v26_realmlp_{oof,test}.npy`, `v27_digit_cnn_{oof,test,full}.npy`, `v30_stack_{oof,test}.npy`.
- `submissions/v30_stack.csv` (final, zones applied), plus `v17b…csv`, `v28…csv`, `v19_sparse_glm_10f.csv`, `v26_realmlp.csv`, `v27_digit_cnn.csv`.
- Code: `v16_glm_boost.py` (extended), `p2_models.py` (fold-count option), `v25_gpt2.py` (bench / fold / full / collect modes), `v26_realmlp.py`, `v27_digit_cnn.py`, `v30_stack.py`.

## 8. Questions for the next phase

1. With GPT-2 infeasible on this GPU, is there a **cheaper generative/token model** that keeps most of its stack value? For example, a small transformer trained from scratch on the ~10k-token value vocabulary, with a restricted output head (the 50k-vocab softmax dominates the cost).
2. The digit CNN earns weight despite 0.9478 solo. Would **self-attention over field slots**, more seeds, or 10 folds be the best way to raise it?
3. RealMLP got no weight without TE. Is RealMLP **with** the 60 TE logits (5% inner holdout) worth trying, or does it collapse into the GLM-boost family as v24 did?
4. Three tied 10-fold GLM-boost members (v17, v17b, v28) now carry ~0.97 of the stack weight. Is more encoding-seed bagging still worthwhile, or is ~0.9517 the practical ceiling of this feature family?
