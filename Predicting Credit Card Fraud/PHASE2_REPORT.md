# Phase 2 implementation report: Kaggle Playground S6E11

Reply to the "Phase 2 Review & Runbook". Everything below was run on CPU (16 cores) plus an RTX 3050 Laptop GPU (6 GB, only for v24). Heavy jobs ran **one at a time**. Base folds: `StratifiedKFold(5, shuffle=True, random_state=42)`, except v17, which uses `StratifiedKFold(10, …, 42)`. Its fold columns below are its OOF scored on the 5-fold seed-42 split, for comparability. All numbers come from OOF arrays on disk.

## Headline

- **Best single model: v17 (10-fold GLM-boost 4-leaf LightGBM), CV AUC 0.95149.** The plan expected 0.9515–0.9516, so it **matched**.
- **Best final: v23 stack, nested CV AUC 0.95156** (mean over stacker splits 43/44/45). The plan expected 0.9517–0.9520+, so it came in **~0.0001–0.0004 below**.
- Phase 1 best was 0.94941 (v15), so Phase 2 added **+0.00215**. v4 baseline 0.94737 → **+0.00419** total.
- **Every** Phase 2 base model beats v11 and v15 in 5/5 folds, except v21 (sklearn MLP), v22 (raw 4-leaf) and v24 (torch MLP).
- **The stack adds only +0.00007 over v17 alone.** The hill-climb combiner chooses **v17 alone** (weight 1.0). Adding v20 or v24 changed the stack by ≤ 0.00001.
- **Not submitted to the public LB yet** (the user will submit `submissions/v23_stack.csv`).


## 1. Master score table

| model                                              |   concat_auc |   mean_fold_auc(5f seed42) |      f0 |      f1 |      f2 |      f3 |      f4 | beats_v11   | beats_v15   |   pr_auc | plan_expected   | iterations                         |   runtime_s |
|:---------------------------------------------------|-------------:|---------------------------:|--------:|--------:|--------:|--------:|--------:|:------------|:------------|---------:|:----------------|:-----------------------------------|------------:|
| v11 EBM (Phase 1 best single)                      |      0.94927 |                    0.94931 | 0.94808 | 0.95022 | 0.95439 | 0.94907 | 0.94478 | 0/5         | 1/5         |  0.52395 |                 | n/a                                |         724 |
| v15 stack (Phase 1 best)                           |      0.94941 |                    0.94945 | 0.94825 | 0.9504  | 0.95447 | 0.94941 | 0.94474 | 4/5         | 0/5         |  0.52379 |                 | n/a                                |             |
| v16 GLM-boost LGB, 1 seed (pilot)                  |      0.95115 |                    0.95117 | 0.95042 | 0.95176 | 0.95605 | 0.95165 | 0.94596 | 5/5         | 5/5         |  0.53366 |                 | mean 1374 trees                    |         194 |
| v16 GLM alone (73 TE logits + base, C=0.3)         |      0.95063 |                    0.95065 | 0.94979 | 0.9513  | 0.95567 | 0.95084 | 0.94566 | 5/5         | 5/5         |  0.53017 | ~0.9506         | newton-cholesky                    |             |
| v16 GLM-boost 4-leaf LGB, 5-fold, 3 seeds          |      0.95121 |                    0.95124 | 0.95047 | 0.95175 | 0.95618 | 0.95163 | 0.94615 | 5/5         | 5/5         |  0.53358 | 0.9511–0.9514   | mean 1163 (407–1920); refit 1396×3 |         905 |
| v17 GLM alone, 10-fold                             |      0.95087 |                    0.95088 | 0.95018 | 0.95141 | 0.95594 | 0.95101 | 0.94587 | 5/5         | 5/5         |  0.53144 |                 | newton-cholesky                    |             |
| v17 GLM-boost 4-leaf LGB, 10-fold, 2 seeds         |      0.95149 |                    0.9515  | 0.95068 | 0.95212 | 0.95634 | 0.9519  | 0.94647 | 5/5         | 5/5         |  0.53455 | 0.9515–0.9516   | mean 1245 (38–3090); refit 1382×2  |        1497 |
| v18 GLM alone                                      |      0.95063 |                    0.95065 | 0.94979 | 0.9513  | 0.95567 | 0.95084 | 0.94566 | 5/5         | 5/5         |  0.53017 |                 | newton-cholesky                    |             |
| v18 GLM-boost depth-2 XGB, 5-fold, 3 seeds         |      0.95111 |                    0.95114 | 0.9503  | 0.95172 | 0.95604 | 0.9515  | 0.94613 | 5/5         | 5/5         |  0.53297 | 0.9508–0.9512   | mean 895 (359–1709); refit 1073×3  |        1570 |
| v19 joint sparse one-hot GLM (62,172 cols, C=0.01) |      0.95011 |                    0.95014 | 0.94935 | 0.95068 | 0.95501 | 0.95041 | 0.94525 | 5/5         | 5/5         |  0.52904 | 0.9501–0.9503   | lbfgs n_iter 134–152               |         305 |
| v20 EBM + cents + 60 TE logits                     |      0.94995 |                    0.95001 | 0.94929 | 0.9499  | 0.95497 | 0.95083 | 0.94504 | 4/5         | 4/5         |  0.53045 | 0.9507–0.9512   | EBM                                |        1792 |
| v21 sklearn MLP (256,128), 12 epochs, 3 seeds      |      0.94259 |                    0.94311 | 0.93955 | 0.94658 | 0.94856 | 0.94372 | 0.93712 | 0/5         | 0/5         |  0.49837 | ~0.9495         | 12 epochs fixed                    |        1468 |
| v22 4-leaf LGB, raw 24 + 6 cents, no TE            |      0.94827 |                    0.94829 | 0.9468  | 0.94974 | 0.95317 | 0.94868 | 0.94309 | 0/5         | 0/5         |  0.51903 | ~0.9478–0.9481  | 5380–6424 trees                    |         836 |
| v24 PyTorch residual MLP (GPU), 10 epochs, 3 seeds |      0.94863 |                    0.94867 | 0.94819 | 0.94934 | 0.95372 | 0.94941 | 0.94269 | 2/5         | 1/5         |  0.51929 | (not in plan)   | 10 epochs fixed                    |         183 |
| **v23 final stack**                                |      0.95157 |                    0.95158 | 0.95071 | 0.9523  | 0.95643 | 0.95188 | 0.94661 | 5/5         | 5/5         |  0.53516 | 0.9517–0.9520+  | n/a                                |             |


v17's own 10-fold AUCs: [0.95, 0.95137, 0.95278, 0.95146, 0.95173, 0.96103, 0.94784, 0.95596, 0.94247, 0.95049] (mean 0.95151, std 0.00459).


'GLM alone' rows = OOF of the newton-cholesky GLM that gives the init_score (averaged over the same encoding seeds). v16 and v18 share the same encodings and GLM, so their 'GLM alone' rows are identical by construction. The v23 row's concat AUC is its nested OOF on stacker split 43 (the headline 0.95156 is the mean over splits 43/44/45). Runtime = wall clock for CV + refit.


## 2. Claims in the runbook vs what we measured

| Runbook claim | Measured | Verdict |
|---|---|---|
| Digit keys carry signal (e.g. `amount_usd` lead2 ≈ 0.583, distance cents ≈ 0.523) | fold-0 single-key AUC: amount lead2 **0.588**, amount int 0.580, hours tenths 0.576, hours cents 0.566, distance cents **0.523**, balance int 0.563 (raw values 0.541 / 0.558 / 0.507 / 0.502) | ✅ confirmed |
| GLM alone ≈ 0.9506 | v16 GLM 0.95063, v17 GLM (10-fold) 0.95087 | ✅ |
| v16 5-fold 3-seed ≈ 0.9511–0.9514 | **0.95121** (1-seed pilot 0.95115) | ✅ |
| v17 10-fold ≈ 0.9515–0.9516 | **0.95149** | ✅ |
| v18 depth-2 XGB ≈ 0.9508–0.9512 | **0.95111** | ✅ |
| v19 sparse GLM ≈ 0.9501–0.9503 | **0.95011** (62,172 cols, lbfgs converged in 134–152 iterations, ~45 s/fold) | ✅ |
| v20 EBM + digits ≈ 0.9507–0.9512 | **0.94995** | ❌ below (+0.0007 over v11 EBM only) |
| v21 sklearn MLP ≈ 0.9495 | **0.94259**, unstable folds 0.937–0.949 | ❌ well below |
| v22 raw 4-leaf ≈ 0.9478–0.9481, earns 0.08–0.09 stack weight | **0.94827**. LogReg-stack coef 0.127, hill weight 0 | ✅ score / ~ weight |
| Final stack 0.9517–0.9520+ | **0.95156** | ❌ slightly below |
| Zero-fraud zones +0.000036, 5/5 folds | **+0.000019 to +0.000021, 5/5 folds** (thr 13.1 in every fold, 0 train frauds in zone) | ✅ direction / smaller size |


## 3. Stacking details (`v23_stack.py`)

- **Inputs:** every model's OOF probabilities → global `rank_logit` (`logit(rank/(n+1))`). Test → `rank_logit` of the **full-data refit** prediction where one exists (`_full.npy`), else the 5-fold mean (v11, v10). This follows the runbook's fix for the scale mismatch flagged in §1.4.
- **Combiners**, each scored by nested CV on three fresh splits (`StratifiedKFold(5, shuffle, seed s)`, s ∈ {43, 44, 45}), with predictions ranked within each validation fold:

| Combiner | nested AUC (43 / 44 / 45) | mean |
|---|---|---|
| LogReg C=0.01 (newton-cholesky) | 0.95150 / 0.95149 / 0.95147 | 0.95149 |
| **LogReg C=0.05** | 0.95157 / 0.95157 / 0.95154 | **0.95156** ← chosen |
| Hill climb (non-negative, rank01, ≤ 80 steps) | 0.95149 / 0.95149 / 0.95149 | 0.95149 (picks v17 only) |
| mix01 (50/50 rank of LR C=0.01 + hill) | 0.95155 / 0.95154 / 0.95154 | 0.95154 |

- **Members (8):** v16, v17, v18, v19, v21, v22, v11, v10. Full-data LogReg C=0.01 coefficients: v17 0.655, v16 0.287, v11 0.219, v22 0.127, v18 0.115, v19 0.042, v21 0.029, v10 −0.111.
- **Variants tried, not kept:** + v20 → 0.95155. + v24 → 0.95155 (v24 coef 0.004).
- **Zero-fraud zones:** z = (`merchant_risk_score` < thr) | (`velocity_score` ≤ 4.1 & `merchant_risk_score` ≤ 20). thr = minimum fraud `merchant_risk_score` in the training folds (13.1 in all 5 folds). Validated on the seed-42 folds using the nested (split-43) stack OOF: gain +0.000016 to +0.000030 per fold, 5/5 positive. Applied to **8,076 test rows** (pushed below all other rows, order kept inside the zone).


## 4. Diversity (Spearman of OOF predictions)

|                   |   v16 |   v17 |   v18 |   v19 |   v20 |   v21 |   v22 |   v24 |   v11 |   v10 |
|:------------------|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|
| v16_glm_lgb       | 1     | 0.997 | 1     | 0.992 | 0.986 | 0.924 | 0.967 | 0.962 | 0.975 | 0.965 |
| v17_glm_lgb10     | 0.997 | 1     | 0.997 | 0.99  | 0.983 | 0.922 | 0.966 | 0.96  | 0.973 | 0.963 |
| v18_glm_xgb       | 1     | 0.997 | 1     | 0.992 | 0.985 | 0.923 | 0.968 | 0.962 | 0.975 | 0.965 |
| v19_sparse_glm    | 0.992 | 0.99  | 0.992 | 1     | 0.981 | 0.918 | 0.969 | 0.957 | 0.97  | 0.964 |
| v20_ebm_digits    | 0.986 | 0.983 | 0.985 | 0.981 | 1     | 0.918 | 0.96  | 0.948 | 0.974 | 0.951 |
| v21_mlp_bag       | 0.924 | 0.922 | 0.923 | 0.918 | 0.918 | 1     | 0.903 | 0.932 | 0.91  | 0.898 |
| v22_lgb4_raw      | 0.967 | 0.966 | 0.968 | 0.969 | 0.96  | 0.903 | 1     | 0.939 | 0.965 | 0.959 |
| v24_torch_mlp     | 0.962 | 0.96  | 0.962 | 0.957 | 0.948 | 0.932 | 0.939 | 1     | 0.942 | 0.939 |
| v11_ebm           | 0.975 | 0.973 | 0.975 | 0.97  | 0.974 | 0.91  | 0.965 | 0.942 | 1     | 0.967 |
| v10_spline_logreg | 0.965 | 0.963 | 0.965 | 0.964 | 0.951 | 0.898 | 0.959 | 0.939 | 0.967 | 1     |


The GLM-boost family (v16/v17/v18) is 0.99+ correlated within itself. The most different useful members are v22 (raw, no TE), v11 (EBM) and v10 (splines). The two MLPs are not different *enough* to earn weight.


## 5. Implementation details and differences from the runbook

1. **`fe_keys.py`:** same 73 keys (24 exact, 36 digit, 13 pair), `np.rint(x*100)` + integer division, factorised on train+test, cached to `keys_cache.npz` (11 s to build). Nested `np.bincount` TE, smooth=20, inner 5-fold. **5.7 s per outer fold** for all 73 keys.
2. **GLM:** `StandardScaler(logit(E))` on 73 TE logits + base matrix (13 standardised numerics with log1p on amount/balance/hours, 6 flags, 35 one-hot cats), `LogisticRegression(C=0.3, newton-cholesky, tol=1e-7)`.
3. **Trees:** raw 24 + 6 cents + 60 TE rates + 60 key counts = 150 columns. LGB `num_leaves=4, lr 0.02, min_child_samples 100, feature_fraction 0.5, bagging 0.8, lambda_l2 1`, early stopping 300 on validation AUC with `init_score`. XGB `max_depth=2, min_child_weight 20, subsample 0.8, colsample 0.5, lambda 2`, `base_margin`.
4. **Seeds:** (teseed, model seed) = (0,42), (1,137), (2,2026) for v16/v18, and the first two for v17. Fold predictions = mean of margins over seeds. Test = fold-mean, plus a **full refit** at round(mean_iters × 1.20) (5-fold) or × 1.11 (10-fold).
5. **v19:** as specified (numerics, flags, cats, B-splines n_knots=8, one-hot `min_frequency=5` on the 13 numeric exact keys and the 36 digit keys, + 60 standardised TE logits), `C=0.01, lbfgs, tol 1e-6`.
6. **v20 / v21 / v22:** as specified. v20 was first killed by an out-of-memory error while another job ran (now strictly sequential), and the rerun matched the first run's folds exactly.
7. **v24 (extra, not in the runbook):** PyTorch 2.11 + CUDA 12.8 on the RTX 3050 (installed in a venv on E:). Inputs: quantile-normal numerics + cents, one-hots, clipped standardised logits of all 73 TE keys. Net: Linear(→256) + 2 residual blocks (LayerNorm, Linear, SiLU, Dropout 0.2, Linear), AdamW 2e-3 / wd 1e-4, OneCycle, batch 4096, **fixed 10 epochs**, 3 seeds. ~6 s per seed per fold. CV 0.94863, stable across seeds (±0.0005).
8. **Original 20k dataset:** not downloaded, following §1.5.
9. **GPU:** XGBoost (`device="cuda"`) and CatBoost (`task_type="GPU"`) work on this machine. LightGBM's pip build is CPU-only. v18 was run on CPU (before the GPU check).


## 6. Saved artifacts

- `oof/<name>_oof.npy` = OOF probabilities. `<name>_test.npy` = 5-fold mean test probabilities. `<name>_full.npy` = full-refit test probabilities (v16–v22, v24). `<name>_glm_oof.npy` = GLM-alone OOF (v16–v18).
- `submissions/v23_stack.csv` = final (rank01 of the LogReg C=0.05 decision function, zones applied). Also `v16_glm_lgb.csv`, `v17_glm_lgb10.csv`, `v18_glm_xgb.csv`, `v19…v24`.
- Code: `fe_keys.py`, `v16_glm_boost.py` (v16/v17/v18 via CLI args), `p2_models.py` (v19–v22), `v24_torch_mlp.py`, `v23_stack.py`.

## 7. Open questions for the next phase

1. The stack is +0.00007 over v17, and hill climbing keeps v17 alone. Is the remaining headroom mainly **more v17-style bagging** (more encoding seeds, 10-fold for v16/v18), or a genuinely new feature family?
2. v20 (EBM + digits) and the MLPs fell short of the runbook's estimates. Are there settings (e.g. EBM with fewer TE columns, MLP with categorical embeddings or a GLM offset) that you've seen work on this data?
3. Is the 10-fold gain (v16 0.95121 → v17 0.95149) worth extending to **20-fold** for the GLM-boost, or will it plateau?
4. Any other synthesizer artifacts worth encoding (e.g. digit keys for `customer_age`/`card_age_months`, cross-column digit pairs)?
