# Predicting Credit Card Fraud

Competition: https://www.kaggle.com/competitions/playground-series-s6e11 (Kaggle Playground Series S6E11)

Predict the probability that a card transaction is fraud (`is_fraud`). Only 1.67% of the 699,960 training rows are fraud.
**Metric:** ROC AUC. **Submission:** `id,is_fraud` with probabilities for 299,984 test rows.

## Setup

```bash
python -m pip install pandas scikit-learn lightgbm xgboost catboost imbalanced-learn matplotlib
```

Download the competition data from https://www.kaggle.com/competitions/playground-series-s6e11/data into `data/` (train.csv, test.csv, sample_submission.csv).
Optional original dataset (CC0, 20k rows): https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026 → `data/credit_card_fraud_2026.csv`. Run scripts from this folder.

## Iterations (5-fold stratified CV, seed 42, same folds for every version)

| Version | Script | Change | CV AUC | Public LB |
|---|---|---|---|---|
| v1 | `v1_logistic_regression.py` | Logistic regression baseline | 0.94187 | - |
| v2 | `v2_lightgbm.py` | LightGBM, raw features | 0.94519 | - |
| v3 | `v3_lightgbm_features.py` | + 13 engineered features (no gain, dropped) | 0.94522 | - |
| v4 | `v4_lightgbm_tuned.py` | Regularised LightGBM (depth 4, lr 0.02, sampling) | 0.94737 | - |
| v5 | `v5_xgboost.py` | XGBoost | 0.94712 | - |
| v6 | `v6_catboost.py` | CatBoost | 0.94619 | - |
| v7 | `v7_ensemble.py` | Weighted rank blend of v4/v5/v6/v1 | **0.94744** | **0.94786** |
| v8 | `v8_imbalance.py` | Class weights / undersampling / SMOTE-NC on v4 (none helped; SMOTE 0.916) | 0.94641-0.94728 | - |
| v9 | `v9_lightgbm_woe.py` (+ `fe_v9.py`) | In-fold exact-value WoE, residual WoE, max_bin 1023 | 0.94855 | - |
| v10 | `v10_spline_logreg.py` | Spline logistic regression + WoE | 0.94721 | - |
| v11 | `v11_ebm.py` | Explainable Boosting Machine | 0.94927 | - |
| v12 / v13 / v14 | `v12_xgboost_woe.py`, `v13_catboost_woe.py`, `v14_lightgbm_extratrees.py` | Old boosters + WoE features | 0.94847 / 0.94780 / 0.94758 | - |
| v15 | `v15_stack.py` | Leak-free LogReg stack of v9-v14 | 0.94941 | **0.95010** |
| v16 | `v16_glm_boost.py` (+ `fe_keys.py`) | 73 TE keys (exact + digit + pair) → GLM → 4-leaf LightGBM, 3 seeds, full refit | 0.95121 | - |
| v17 | `v16_glm_boost.py v17_glm_lgb10 lgb 10 ...` | Same, 10-fold, 2 seeds | **0.95149** (best single) | - |
| v18 | `v16_glm_boost.py v18_glm_xgb xgb 5 ...` | GLM → depth-2 XGBoost | 0.95111 | - |
| v19 / v20 / v21 / v22 | `p2_models.py v19 / v20 / v21 / v22` | Sparse one-hot GLM / EBM + digits / sklearn MLP / raw 4-leaf LightGBM | 0.95011 / 0.94995 / 0.94259 / 0.94827 | - |
| v24 | `v24_torch_mlp.py` (run with `E:\envs\kaggle\Scripts\python.exe`) | PyTorch residual MLP on the GPU, 3 seeds, full refit (stack weight ≈ 0) | 0.94863 | - |
| v17b | `v16_glm_boost.py v17b_glm_lgb10_s100 lgb 10 2:2026:100,3:777:100 refit` | 10-fold GLM-boost LightGBM, smoothing 100 | 0.95152 | - |
| v28 | `v16_glm_boost.py v28_glm_xgb10_gpu xgbgpu 10 ...` | 10-fold GLM-boost XGBoost on GPU, 4 seeds (smoothing 20 depth-2 + smoothing 100 4-leaf lossguide) | 0.95150 | - |
| v29 | `p2_models.py v19 10` | 10-fold joint sparse GLM | 0.95035 | - |
| v25 | `v25_gpt2.py` | Label-first GPT-2 generative classifier: **not run** (benchmark: ~114 h on the RTX 3050) | - | - |
| v26 | `v26_realmlp.py` | RealMLP (pytabkit, GPU), no target encoding | 0.94747 | - |
| v27 | `v27_digit_cnn.py` | 1-D character/digit CNN (GPU), no target encoding | 0.94783 | - |
| v30 | `v30_stack.py` | Rank-logit LogReg stack (native-fold standardisation, C grid) of 10 models + zero-fraud zones | **0.95166** | **0.95128** |
| P4.0 | `p4_glm_only.py` | v28's pure 10-fold dense GLM as its own member | 0.95094 | - |
| v25 | `kaggle_v25_gpt2.py` (run on **Kaggle 2× T4**, 11.6 h, 4 epochs, 8 orders) | Compact-vocab label-first GPT-2 generative classifier, no TE | 0.94836 | - |
| v26b / v27b | `v26b_realmlp_keys.py`, `v27b_digit_cnn_seg.py` | Fold-1 pilots: 0.94806 (gate 0.9493) / 0.94585 (gate +0.0008 over v27) — both failed the gate | - | - |
| v31 | `v31_stack.py` | v30 pool + GLM-only (gate +0.00004, 4/5 folds), LogReg C=0.05 (GLM-only gets a **negative** weight); non-negative stacker 0.95158 | 0.95171 | - |
| **v32** | `v32_stack.py` | v31 pool + v25 GPT-2 (gate: +0.00009, 12/15 folds; GPT-2 coef +0.312, top non-negative weight 0.218) | **0.95180** | **0.95154** |
| v23 | `v23_stack.py` | Rank-logit LogReg stack (nested on 3 fresh splits) + zero-fraud zones | **0.95156** | **0.95166** |

`common.py` holds data loading, features and the shared CV runner. Every run appends to `results.csv` and writes `oof/` predictions plus `submissions/<version>.csv`.
`analysis.py` answers the original dataset's questions: feature importance, PR-AUC (0.514) and threshold tuning, AI-scam detection, authentication (`analysis_results.txt`).
`tune_lightgbm.py` is the settings search behind v4 (`tune_lightgbm_results.txt`).

Only the v4, v7, v15, v23, v30, v31 and v32 submissions are committed. The others are regenerated by running their scripts (v7 needs v1, v4, v5 and v6 run first).

## Guide

[guide/Credit_Card_Fraud_Beginner_Guide.md](guide/Credit_Card_Fraud_Beginner_Guide.md): a beginner's walkthrough of every iteration and why each one did or didn't beat the previous one, the dataset questions, and a detailed "Terms and techniques" section (every method explained in depth and like you're 5), with 24 diagrams **embedded inside the file** (one self-contained Markdown file). Sources and diagram scripts are in `guide/build/`. Rebuild with `python guide/build/build_guide.py`.
