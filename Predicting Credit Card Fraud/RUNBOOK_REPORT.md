# Runbook implementation report: Kaggle Playground S6E11

Everything below was run on the same folds, `StratifiedKFold(5, shuffle=True, random_state=42)`, and summarised from artifacts on disk. Fold numbering is 0 to 4. "Concat" = AUC of the concatenated raw OOF probabilities. "Std-logit" = AUC of the OOF after within-fold standardised log-odds (= within-fold rank AUC to 5 decimals here). PR-AUC = average precision on the concatenated OOF.

**Status:** Experiments 1, 2, 3, 4, 5 and 8 are done. **Experiment 6 (MLP) not run** (PyTorch isn't installed). **Experiment 7 / Section 2 not run** (the original `credit_card_fraud_2026.csv` isn't on disk yet). **No public LB submissions yet.**

**Headline:** best single model **v11 EBM 0.94927**. Best blend **v15 LogReg stack 0.94941** (concat; std-logit 0.94945). Starting point v4 0.94737 / v7 0.94744.


## 1. Master score table

| model                                                         |   concat_auc |   std_logit_auc |   fold0 |   fold1 |   fold2 |   fold3 |   fold4 | beats_v4   | beats_te_nums   |    pr_auc | best_iters               |   runtime_s |
|:--------------------------------------------------------------|-------------:|----------------:|--------:|--------:|--------:|--------:|--------:|:-----------|:----------------|----------:|:-------------------------|------------:|
| v1 LogReg (baseline)                                          |      0.94187 |         0.94191 | 0.9413  | 0.9436  | 0.94664 | 0.94168 | 0.9363  | 0/5        | 0/5             |   0.48931 | see 3                    |          11 |
| v4 LightGBM (reference)                                       |      0.94737 |         0.9474  | 0.94552 | 0.9493  | 0.95189 | 0.94749 | 0.94282 | 0/5        | 1/5             |   0.5137  | 2280 2212 2547 1674 2355 |         188 |
| v5 XGBoost                                                    |      0.94712 |         0.94723 | 0.94551 | 0.94914 | 0.95154 | 0.94729 | 0.94258 | 0/5        | 1/5             |   0.51156 | 2254 2185 1592 1987 2746 |         338 |
| v6 CatBoost                                                   |      0.94619 |         0.94624 | 0.9444  | 0.94771 | 0.9507  | 0.94632 | 0.94211 | 0/5        | 0/5             |   0.50621 | 2102 1551 1935 1254 2070 |        1731 |
| v7 rank blend (old best)                                      |      0.94744 |         0.94748 | 0.94567 | 0.94935 | 0.95195 | 0.94754 | 0.94295 | 5/5        | 1/5             |   0.51404 | n/a                      |             |
| screen base (fast LGB lr 0.05)                                |      0.94716 |         0.94722 | 0.94548 | 0.94922 | 0.95174 | 0.94736 | 0.94232 | 0/5        | 0/5             |   0.51123 | 968 955 847 698 1069     |             |
| screen_te_nums (sklearn TE on 13 numerics, fast LGB)          |      0.94791 |         0.94801 | 0.9466  | 0.94965 | 0.95248 | 0.94872 | 0.94257 | 4/5        | 0/5             |   0.51575 | 680 698 515 499 693      |             |
| **v9 = v4_plus** (LGB lr 0.02 + WoE m40 + det + max_bin 1023) |      0.94855 |         0.94858 | 0.94751 | 0.94987 | 0.95343 | 0.94904 | 0.94309 | 5/5        | 5/5             |   0.51844 | 1908 1834 1838 1277 1696 |         184 |
| **v10 spline LogReg** (Exp 3)                                 |      0.94721 |         0.94725 | 0.94595 | 0.94816 | 0.95289 | 0.94694 | 0.94234 | 2/5        | 1/5             |   0.51014 | see 3                    |         100 |
| **v11 EBM** (Exp 4, 4 bags, 10 interactions)                  |      0.94927 |         0.94929 | 0.94808 | 0.95022 | 0.95439 | 0.94907 | 0.94478 | 5/5        | 5/5             |   0.52395 | see 3                    |         724 |
| v11b EBM (8 bags, 20 interactions)                            |      0.94919 |         0.94921 | 0.94809 | 0.95025 | 0.95426 | 0.94893 | 0.9446  | 5/5        | 5/5             |   0.52384 | see 3                    |        1297 |
| **v12 = xgb_v5_plus** (Exp 5)                                 |      0.94847 |         0.94851 | 0.94757 | 0.9496  | 0.95329 | 0.94868 | 0.94344 | 5/5        | 3/5             |   0.51912 | 2001 1799 1938 1647 2090 |         445 |
| **v13 = cat_v6_plus** (Exp 5)                                 |      0.9478  |         0.94783 | 0.94613 | 0.94872 | 0.95326 | 0.94823 | 0.94284 | 4/5        | 2/5             |   0.51654 | 4308 3377 4151 3705 3461 |        1977 |
| **v14 = lgb_extratrees** (Exp 5)                              |      0.94758 |         0.94759 | 0.94622 | 0.94895 | 0.9528  | 0.94787 | 0.94221 | 3/5        | 1/5             |   0.5143  | 2349 2096 2071 1456 1577 |         238 |
| **v15 OOF LogReg stack** (Exp 8.2)                            |      0.94941 |         0.94945 | 0.94825 | 0.9504  | 0.95447 | 0.94941 | 0.94474 | 5/5        | 5/5             |   0.52379 | n/a                      |             |
| s9_maxbin (max_bin 1023 only)                                 |      0.94751 |       nan       | 0.94598 | 0.94934 | 0.95193 | 0.94759 | 0.9428  | 4/5        | 1/5             | nan       | 1045 1089 905 1022 796   |         108 |
| s9_det (+logfreq, is_velocity_zero, max_bin)                  |      0.94727 |       nan       | 0.94596 | 0.9487  | 0.95179 | 0.94784 | 0.9424  | 2/5        | 0/5             | nan       | 1103 821 831 863 998     |          94 |
| s9_woe (m=15, resid, memo, det, max_bin)                      |      0.94839 |       nan       | 0.94718 | 0.94981 | 0.95322 | 0.94887 | 0.94308 | 5/5        | 5/5             | nan       | 576 519 689 513 685      |          98 |
| s9_woe_noresid (m=15, plain WoE for risk/velocity)            |      0.9483  |       nan       | 0.94741 | 0.94932 | 0.95315 | 0.94898 | 0.94282 | 4/5        | 4/5             | nan       | 615 646 585 545 552      |          80 |
| s9_woe_m40 (m=40)                                             |      0.94842 |       nan       | 0.94722 | 0.94966 | 0.95317 | 0.94903 | 0.94324 | 5/5        | 5/5             | nan       | 591 620 620 483 682      |          89 |
| s9_woe_m40_nodet (m=40, no logfreq/velocity-zero)             |      0.94832 |       nan       | 0.94699 | 0.94972 | 0.95341 | 0.94895 | 0.94302 | 5/5        | 5/5             | nan       | not logged               |         112 |
| s9_woe_dropnoise (m=15, drop 6 noise cols)                    |      0.9482  |       nan       | 0.94737 | 0.94957 | 0.95286 | 0.94862 | 0.94294 | 5/5        | 3/5             | nan       | 838 695 555 471 643      |          91 |
| s9_woe_pairs (m=15, + 3 pair WoEs)                            |      0.94829 |       nan       | 0.94711 | 0.94942 | 0.95333 | 0.94867 | 0.94323 | 5/5        | 3/5             | nan       | 896 759 622 557 698      |         106 |


Notes: the screen variants used the fast LightGBM (lr 0.05, patience 200) and didn't save OOF arrays, so std-logit and PR-AUC are blank for them. Their fold AUCs/iterations come from the run log. Runtime of v11/v11b = EBM wall clock for all 5 folds. v7/v15 are blends (no iterations). The v15 concat AUC is computed on the meta-model's nested OOF decision function.


## 2. Feature engineering & ablation

**Exact WoE specification (`fe_v9.fold_woe_features`)**
- Exact-value smoothed WoE, `log(p_smooth/(1-p_smooth)) - logit(prior)` with `p_smooth = (sum + m*prior)/(count + m)`, keyed on `col.astype(str)`:
  `hours_since_last_txn, customer_age, account_balance_usd, amount_usd, distance_from_home_km`.
- Residual WoE (exact WoE minus coarse WoE on `floor(x/2)`, coarse smoothing `m=50`): `merchant_risk_score, velocity_score`.
- `woe_memo_sum` = sum of the WoEs of `account_balance_usd, amount_usd, distance_from_home_km, hours_since_last_txn` + the two residual WoEs.
- Training rows: inner `StratifiedKFold(5, shuffle, seed 42)` out-of-fold encoding. Validation and test rows: encoding fitted on the full training fold. Unseen values → 0.
- Smoothing tested: **m = 15 → 0.94839, m = 40 → 0.94842** (m=40 chosen). 2-way interaction WoEs (`auth×customer_age`, `merchant_category×auth`, `auth×cvv`) tested → 0.94829 (not kept).
- Label-free: `*_logfreq` (log1p count in train+test) for `account_balance_usd, amount_usd, hours_since_last_txn, merchant_risk_score, velocity_score`; `is_velocity_zero`.

**Individual vs combined effect (fast-LGB screen, base 0.94716):**

| Change | CV AUC | Δ vs base |
|---|---|---|
| max_bin 1023 alone | 0.94751 | +0.00035 |
| + logfreq + is_velocity_zero (with max_bin) | 0.94727 | +0.00011 (−0.00024 vs max_bin alone) |
| + WoE set, m=40, no det features | 0.94832 | +0.00116 |
| + WoE set, m=40, with det features (**chosen**) | 0.94842 | +0.00126 |
| WoE without residualisation (plain WoE for risk/velocity) | 0.94830 | +0.00114 |

So the WoE block carries almost all of the gain. max_bin helps on raw features. The det features hurt alone but add +0.0001 on top of WoE (within noise). `woe_memo_sum` was not ablated separately.


**Top 20 features of v9 (fold-0 model, 1908 trees; fold-0 validation AUC 0.94751)**


| feature                       |   gain_pct |   perm_auc_drop |
|:------------------------------|-----------:|----------------:|
| cvv_retry_count               |   21.471   |         0.03125 |
| merchant_risk_score           |   13.5796  |         0.04441 |
| auth_method                   |    9.59544 |         0.02186 |
| velocity_score                |    8.87161 |         0.03707 |
| is_new_merchant               |    7.33412 |         0.01458 |
| ip_country_mismatch           |    6.11262 |         0.00568 |
| billing_shipping_mismatch     |    3.98638 |         0.00565 |
| customer_age_woe              |    3.95843 |         0.01053 |
| is_foreign_transaction        |    2.9872  |         0.00233 |
| used_vpn                      |    2.75249 |         0.00384 |
| merchant_risk_score_logfreq   |    2.25014 |         0.00102 |
| merchant_category             |    1.9069  |         0.00185 |
| velocity_score_logfreq        |    1.88949 |         0.00157 |
| customer_age                  |    1.38176 |         0.00067 |
| is_ai_generated_scam_attempt  |    1.19732 |         0.00157 |
| hours_since_last_txn_woe      |    0.84949 |         0.00113 |
| merchant_risk_score_resid_woe |    0.82272 |         0.0003  |
| hours_since_last_txn          |    0.81196 |         0.0025  |
| woe_memo_sum                  |    0.73955 |         0.00065 |
| amount_usd                    |    0.73285 |         0.00057 |


New-feature permutation drops (all): customer_age_woe +0.01053, merchant_risk_score_logfreq +0.00102, velocity_score_logfreq +0.00157, hours_since_last_txn_woe +0.00113, merchant_risk_score_resid_woe +0.00030, woe_memo_sum +0.00065, hours_since_last_txn_logfreq +0.00022, account_balance_usd_woe +0.00007, distance_from_home_km_woe +0.00007, amount_usd_logfreq +0.00040, velocity_score_resid_woe -0.00009, amount_usd_woe +0.00001, account_balance_usd_logfreq +0.00004, is_velocity_zero -0.00000


**Noise-column drop** (`channel, device_type, card_type, time_of_day_hour, day_of_week, card_age_months`; on the m=15 WoE screen): CV AUC **0.94839 → 0.94820 (−0.00019)**, folds beat 1/5. Mean best trees 596 → 640. Not adopted.


## 3. Non-tree & diverse models

**v10 spline LogReg (Exp 3).** `ColumnTransformer`:
- `SplineTransformer(n_knots=12, degree=3, knots="quantile", include_bias=False)` on `merchant_risk_score, velocity_score, log1p(hours), log1p(amount), log1p(balance), distance_from_home_km`;
- `OneHotEncoder(handle_unknown="ignore", min_frequency=10)` on exact `customer_age, cvv_retry_count, txn_count_last_24h, prior_disputes` + the 5 categoricals;
- `StandardScaler` on the 6 flags, `flag_count`, `is_velocity_zero`, 5 `*_logfreq`, the 7 WoE columns + `woe_memo_sum`;
- plus 20 standardised interactions: `auth_method` dummies × `[is_velocity_zero, merchant_risk_score/100, cvv_retry_count, flag_count]`.

`LogisticRegression(C=0.2, solver="lbfgs", max_iter=3000)` on the sparse matrix (239 columns). Fold-0 refit converged in **n_iter_ = 55** (< 3000, no warning).
CV AUC **0.94721** vs v1 0.94187 (+0.00534), beats v4 in 2/5 folds. 99 s for all folds.

**Hyperparameters of the other Exp 4/5 models** (all with raw features + det features + in-fold WoE m=40):

| Model | Settings | CV AUC | Notes |
|---|---|---|---|
| v11 EBM | `max_bins=1024, max_interaction_bins=64, interactions=10, outer_bags=4, inner_bags=0, learning_rate=0.015, min_samples_leaf=100, early_stopping_rounds=100, n_jobs=-1` | 0.94927 | 724 s. Categoricals passed as strings. Harmless joblib/loky resource-tracker `KeyError` (temp memmap cleanup) printed on Windows; run unaffected |
| v11b EBM | same, `interactions=20, outer_bags=8` | 0.94919 | 1,297 s. No gain |
| v12 XGBoost | `hist, max_bin=1024, lr 0.02, max_depth 4, min_child_weight 10, subsample 0.7, colsample_bytree 0.4, reg_lambda 10, enable_categorical, max_cat_to_onehot=16, early_stopping 400` | 0.94847 | 444 s |
| v13 CatBoost | `depth 4, lr 0.03, l2_leaf_reg 15, rsm 0.4, Bernoulli subsample 0.75, border_count 254, max_ctr_complexity 1, od_wait 300`; cat_features = 5 cats + `hours_since_last_txn_str`, `customer_age_str` | 0.94780 | 1,977 s (slowest) |
| v14 LGB extra_trees | `extra_trees=True, max_depth 5, num_leaves 24, colsample 0.5, max_bin 1023, lr 0.02` (other v4 params) | 0.94758 | 238 s |
| MLP (Exp 6) | not run | | PyTorch not installed on Python 3.14 env |


**Diversity matrix (Spearman of OOF predictions)**


|                         |    v1 |    v4 |    v9 |   v10 |   v11 |   v11b |   v12 |   v13 |   v14 |
|:------------------------|------:|------:|------:|------:|------:|-------:|------:|------:|------:|
| v1_logistic_regression  | 1     | 0.938 | 0.923 | 0.92  | 0.901 |  0.9   | 0.923 | 0.938 | 0.925 |
| v4_lightgbm_tuned       | 0.938 | 1     | 0.978 | 0.962 | 0.962 |  0.962 | 0.977 | 0.968 | 0.975 |
| v9_lightgbm_woe         | 0.923 | 0.978 | 1     | 0.976 | 0.98  |  0.98  | 0.997 | 0.983 | 0.989 |
| v10_spline_logreg       | 0.92  | 0.962 | 0.976 | 1     | 0.967 |  0.966 | 0.976 | 0.976 | 0.98  |
| v11_ebm                 | 0.901 | 0.962 | 0.98  | 0.967 | 1     |  0.998 | 0.98  | 0.97  | 0.973 |
| v11b_ebm_big            | 0.9   | 0.962 | 0.98  | 0.966 | 0.998 |  1     | 0.98  | 0.969 | 0.973 |
| v12_xgboost_woe         | 0.923 | 0.977 | 0.997 | 0.976 | 0.98  |  0.98  | 1     | 0.983 | 0.988 |
| v13_catboost_woe        | 0.938 | 0.968 | 0.983 | 0.976 | 0.97  |  0.969 | 0.983 | 1     | 0.98  |
| v14_lightgbm_extratrees | 0.925 | 0.975 | 0.989 | 0.98  | 0.973 |  0.973 | 0.988 | 0.98  | 1     |


## 4. Stacking, seeds, original data, leaderboard

**v15 stack (`v15_stack.py`)**
- Inputs: OOF of `v9, v10, v11, v12, v13, v14`, each converted to **within-fold standardised log-odds**. Test inputs = the stored 5-fold mean of std-logit test predictions.
- Meta-model: `LogisticRegression(C=0.05, max_iter=1000)` (unconstrained sign), scored **nested on the same 5 folds** (fit on 4 folds' OOF, predict the 5th).
- Alternative: greedy non-negative hill climbing (with replacement, ≤ 60 steps), also nested. The script keeps whichever nested AUC is higher.
- Results: **LogReg stack 0.94941** vs hill-climb 0.94936. Full-data meta coefficients: v9 0.281, v10 0.195, **v11 1.385**, v12 0.376, v13 0.293, v14 −0.223. Hill-climb weights: v11 0.667, v9 0.167, v10 0.167.
- Gain vs best single (v11 0.94927): **+0.00014**. Adding v11b → 0.94942 (noise, not kept). An earlier 5-model stack without the EBM (v9, v10, v4, v5, v6) gave 0.94892.

**Seed averaging:** not run yet.

**Original 20k dataset:** not on disk, so Section 2 and Experiment 7 haven't been run. Source: https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026

**Public LB:** nothing submitted yet. Candidate: `submissions/v15_stack.csv` (CV 0.94941).


## 5. Saved artifacts & implementation differences

**OOF / test arrays in `oof/`:** `screen_base_oof.npy`, `screen_te_nums_oof.npy`, `v10_spline_logreg_oof.npy`, `v10_spline_logreg_test.npy`, `v11_ebm_oof.npy`, `v11_ebm_test.npy`, `v11b_ebm_big_oof.npy`, `v11b_ebm_big_test.npy`, `v12_xgboost_woe_oof.npy`, `v12_xgboost_woe_test.npy`, `v13_catboost_woe_oof.npy`, `v13_catboost_woe_test.npy`, `v14_lightgbm_extratrees_oof.npy`, `v14_lightgbm_extratrees_test.npy`, `v15_stack_oof.npy`, `v15_stack_test.npy`, `v1_logistic_regression_oof.npy`, `v1_logistic_regression_test.npy`, `v2_lightgbm_oof.npy`, `v2_lightgbm_test.npy`, `v3_lightgbm_features_oof.npy`, `v3_lightgbm_features_test.npy`, `v4_lightgbm_tuned_oof.npy`, `v4_lightgbm_tuned_test.npy`, `v5_xgboost_oof.npy`, `v5_xgboost_test.npy`, `v6_catboost_oof.npy`, `v6_catboost_test.npy`, `v7_ensemble_oof.npy`, `v7_ensemble_test.npy`, `v8_class_weights_oof.npy`, `v8_class_weights_sqrt_oof.npy`, `v8_class_weights_sqrt_test.npy`, `v8_class_weights_test.npy`, `v8_none_oof.npy`, `v8_none_test.npy`, `v8_smote_oof.npy`, `v8_smote_test.npy`, `v8_undersample_oof.npy`, `v8_undersample_test.npy`, `v9_lightgbm_woe_oof.npy`, `v9_lightgbm_woe_test.npy`


- v1 to v8 (`common.run_cv`): test = **5-fold mean of raw probabilities**. OOF = raw probabilities.
- v9 to v14 (`fe_v9.run_cv_ff`): test = **5-fold mean of within-fold standardised log-odds** (`_test.npy` holds the z-scores). The submission CSV = `sigmoid(z)`, a monotone transform, so it's AUC-equivalent but the probabilities aren't calibrated (mean ≈ 0.5). OOF = raw probabilities.
- v15: OOF = nested meta decision function. Test = meta decision function on the test z-scores. CSV = sigmoid of it.
- No full-data refits. All test predictions are fold-model averages.
- Submissions on disk: `submissions/v*.csv` for every version (only v4, v7, v15 are committed to git).

**Differences from MODEL_IMPROVEMENT_PLAN.md**
1. **Exp 1 adopted for test aggregation, not for model selection.** Within-fold standardisation lifts *OOF* AUC by ~0.0001 because it equals the mean of fold AUCs (≥ concatenated AUC). The LB scores one averaged test vector, so that lift isn't a real gain. I compare models on concatenated raw OOF AUC and report std-logit alongside.
2. **WoE coarse bins** for the residual WoE use `floor(x/2)` as in the plan, with coarse smoothing m=50. The memo sum uses the residual (not plain) WoE for risk/velocity, as specified.
3. **WoE implemented directly** (inner 5-fold) rather than via sklearn TargetEncoder. Unseen values → 0 (prior).
4. **Spline LogReg**: interactions built as auth dummies × 4 drivers (20 columns) and standardised. No polynomial expansion.
5. **EBM**: `outer_bags=4` (plan suggested 8, tried as v11b with no gain) and `interactions=10` (v11b: 20).
6. **XGBoost**: `max_cat_to_onehot=16` as suggested. **CatBoost**: `depth=4` (plan allowed 4 or 5).
7. **Stacker**: plain `LogisticRegression(C=0.05)` without a non-negativity constraint (v14 got a negative coefficient). The hill-climb alternative is non-negative and scored 0.00005 lower.

```python
# fe_v9.run_cv_ff: test aggregation
p_va, p_te = fit_predict(X_tr, y[tr], X_va, y[va], X_te)
oof[va] = p_va                                   # raw probabilities kept for OOF
test_z += to_std_logit(np.asarray(p_te)) / N_FOLDS
```
