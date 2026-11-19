# Model review brief: Kaggle Playground S6E11 "Predicting Credit Card Fraud"

**Purpose of this document:** a complete technical summary of my work so far, so that you can suggest concrete ways to raise the score. Please read it all, then answer the questions in the final section.

**Goal:** move the cross-validated ROC AUC from **0.9474** towards **0.96**. (The competition metric is ROC AUC, not accuracy.) If you think 0.96 is unrealistic for this data, please say so and explain why, and tell me what the realistic ceiling is and how to get closest to it.

---

## 1. Competition

| Item | Detail |
|---|---|
| Competition | Kaggle Playground Series Season 6 Episode 11, "Predicting Credit Card Fraud" (`playground-series-s6e11`) |
| Task | Binary classification: predict P(`is_fraud` = 1) per transaction |
| Metric | **ROC AUC** between predicted probability and target |
| Submission | `id,is_fraud` (probability), 299,984 rows |
| Data origin | **Synthetic.** Generated (presumably by a deep generative model, as usual in Playground series) from the original CC0 dataset "Credit Card Fraud 2026": 20,000 rows, 339 frauds (1.70%), itself fully synthetic and generated with NumPy (`gen.py`, which I have not seen). |

### Original dataset's description (verbatim key points)
> 20,000 transactions in 2026 with readable columns... 1.70% fraud (339 of 20,000). 26 columns, zero missing values. Fully synthetic, generated with NumPy (see gen.py). Ideas: report PR-AUC, feature importance, threshold tuning, compare SMOTE / class weights / undersampling, are AI-generated scam attempts harder to catch, does strong authentication suppress fraud, engineer velocity features from `hours_since_last_txn` and `txn_count_last_24h`.

---

## 2. Data

- `train.csv`: **699,960 rows**, 26 columns (`id`, 24 features, `is_fraud`). Fraud rate **1.670%** (11,689 frauds).
- `test.csv`: **299,984 rows**, ids 699960 to 999943 (they follow train's ids directly).
- **No missing values.** No exact duplicate feature rows in train. No test row exactly matches a train row.
- No noticeable train/test distribution shift (standardised mean difference < 0.05 on every numeric column).

### Features

| Feature | Type | Unique values | Range / values | Single-feature AUC* |
|---|---|---|---|---|
| amount_usd | float | 13,387 | 1.00 to 6,872.69 (mean 79.9) | 0.545 (+) |
| hours_since_last_txn | float | 3,633 | 0.01 to 51.98 | 0.568 (−) |
| txn_count_last_24h | int | 12 | 0 to 11 | 0.620 (+) |
| distance_from_home_km | float | 4,365 | 0 to 57.93 | 0.504 |
| card_age_months | int | 49 | 25 to 73 | 0.502 |
| customer_age | int | 64 | 18 to 81 | 0.625 (+) |
| account_balance_usd | float | 43,317 | 58.25 to 99,701 | 0.506 |
| cvv_retry_count | int | 4 | 0 to 3 | 0.708 (+) |
| velocity_score | float | 666 | 0 to 72.9 | 0.711 (+) |
| time_of_day_hour | int | 24 | 0 to 23 | 0.500 |
| day_of_week | int | 7 | 0 to 6 | 0.508 |
| merchant_risk_score | float | 950 | 0 to 100 | 0.749 (+) |
| prior_disputes | int | 5 | 0 to 4 | 0.532 (+) |
| merchant_category | cat | 12 | Groceries, Online Retail, Restaurants, Fuel, Electronics, Streaming, Healthcare, Utilities, Travel, Gaming, Gift Cards, Crypto Exchange | |
| card_type | cat | 5 | Visa, Mastercard, Amex, RuPay, Discover | |
| auth_method | cat | 5 | 3D Secure, OTP, PIN, Biometric, No Authentication | |
| channel | cat | 5 | Online, POS, Contactless, In-App, ATM | |
| device_type | cat | 8 | Android Phone, iPhone, POS Terminal, Windows PC, Mac, Tablet, Smart Watch, ATM Machine | |
| is_foreign_transaction | bool | 2 | 1.7% True | |
| is_new_merchant | bool | 2 | 15.2% True | |
| used_vpn | bool | 2 | 3.4% True | |
| ip_country_mismatch | bool | 2 | 1.6% True | |
| billing_shipping_mismatch | bool | 2 | 1.3% True | |
| is_ai_generated_scam_attempt | bool | 2 | 0.3% True | |

\* Single-feature AUC = max(AUC, 1 − AUC) of the raw column vs target; (+) = higher value → more fraud.

### Fraud rates by category / flag (train)

| Variable | Fraud rate by value |
|---|---|
| auth_method | Biometric 0.50%, PIN 0.87%, OTP 1.41%, 3D Secure 1.47%, **No Authentication 6.44%** |
| merchant_category | Gift Cards 7.3%, Crypto Exchange 6.5%, Gaming 5.5%, Electronics 3.1%, Streaming 2.3%, Travel 1.9%, Utilities 1.4%, Healthcare 1.3%, Fuel 1.2%, Restaurants/Groceries/Online Retail 1.1% |
| card_type | 1.5% to 1.9% (weak) |
| channel | 1.55% to 1.74% (almost no signal) |
| device_type | 1.61% to 1.73% (almost no signal) |
| ip_country_mismatch | True 16.5% vs False 1.43% |
| is_ai_generated_scam_attempt | True 15.2% vs False 1.63% |
| billing_shipping_mismatch | True 13.0% vs False 1.52% |
| is_foreign_transaction | True 12.6% vs False 1.48% |
| used_vpn | True 6.6% vs False 1.50% |
| is_new_merchant | True 4.7% vs False 1.12% |

Authentication ordering holds even among rows with a location/address red flag (Biometric 4.0% … No Auth 35.4%).

### Feature importance (tuned LightGBM, 80/20 split, validation AUC 0.946)

| Feature | Gain % | Permutation AUC drop |
|---|---|---|
| merchant_risk_score | 16.4 | 0.0524 |
| velocity_score | 11.2 | 0.0522 |
| cvv_retry_count | 22.1 | 0.0313 |
| auth_method | 10.2 | 0.0207 |
| is_new_merchant | 7.0 | 0.0160 |
| customer_age | 5.0 | 0.0147 |
| billing_shipping_mismatch | 3.8 | 0.0088 |
| used_vpn | 2.6 | 0.0064 |
| ip_country_mismatch | 6.7 | 0.0050 |
| merchant_category | 2.6 | 0.0025 |
| hours_since_last_txn | 1.3 | 0.0024 |
| is_ai_generated_scam_attempt | 1.3 | 0.0022 |
| is_foreign_transaction | 2.8 | 0.0019 |
| amount_usd | 1.4 | 0.0011 |
| prior_disputes, txn_count_last_24h | ~0.5 to 1.0 | ~0.0008 |
| device_type, account_balance_usd, card_type, card_age_months, channel, time_of_day_hour, distance_from_home_km, day_of_week | < 1 | ≈ 0 |

---

## 3. Validation setup

- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, the **same folds for every experiment**.
- Score = ROC AUC of the concatenated out-of-fold predictions. Test predictions = mean of the 5 fold models.
- Fold-to-fold std of AUC ≈ **0.003**. Paired fold-by-fold comparison is used to judge small gains (an idea counts if it wins 5/5 folds).
- Early stopping on the validation fold (patience 200 to 300). Known small optimistic bias, equal across experiments.
- Hardware: Windows, 16 CPU cores, no GPU used, Python 3.14. Libraries: scikit-learn, LightGBM 4.7, XGBoost 3.4, CatBoost 1.2, imbalanced-learn 0.14.
- **No public leaderboard score yet** for this competition (submissions not made yet), so all numbers are CV.

---

## 4. Everything tried so far (5-fold CV ROC AUC)

| # | Model / idea | CV AUC | PR-AUC | Notes |
|---|---|---|---|---|
| v1 | Logistic regression (one-hot cats, standardised nums, C=1) | 0.94187 | n/a | 11 s. Very strong for a linear model |
| v2 | LightGBM default-ish (lr 0.05, 31 leaves, raw features, native cats) | 0.94519 | n/a | +0.0034 vs v1, 5/5 folds. Early-stopped at ~190 to 390 trees |
| v3 | v2 + 13 engineered features | 0.94522 | n/a | no gain (3/5 folds). Features: flag_count, log_amount, amount/balance, amount / merchant-category mean amount, is_night, merchant_risk×velocity, no_auth, no_auth×flag_count, txn_count_24h/(hours_since_last+0.1), frequency encoding of amount, velocity, merchant_risk, balance |
| — | LightGBM regularisation search (3 configs × raw/engineered) | 0.94721 to 0.94737 | | All configs ≈ equal. Engineered features again no help |
| v4 | **LightGBM regularised**: lr 0.02, max_depth 4, num_leaves 31, min_child_samples 300, colsample_bytree 0.4, subsample 0.7, reg_lambda 10, cat_smooth 50, raw features | **0.94737** | 0.5137 | +0.0022 vs v2, 5/5 folds. ~1,500 to 2,400 trees |
| v5 | XGBoost hist, depth 4, lr 0.02, min_child_weight 10, subsample 0.7, colsample 0.5, lambda 10, native categorical | 0.94712 | | lost 5/5 to v4 by ~0.0002 |
| v6 | CatBoost depth 6, lr 0.05, l2 10, native cats | 0.94619 | | 29 min. Lost 5/5 to v4 |
| v7 | **Rank-average blend**, weights optimised on OOF (Nelder-Mead): v4 0.63, v5 0.24, v6 0.12, v1 0.01 | **0.94744** | 0.5140 | best. +0.00008 vs v4, 5/5 folds. Equal-weight blend = 0.94702 |
| v8a | v4 + scale_pos_weight 58.9 | 0.94641 | 0.5067 | (first run gave 0.840 due to early stopping on logloss; fixed with metric="auc") |
| v8b | v4 + scale_pos_weight 7.7 (sqrt) | 0.94728 | 0.5127 | |
| v8c | v4 on random undersample 1:10 (in-fold) | 0.94678 | 0.5103 | 5× fewer rows, nearly equal |
| v8d | v4 on SMOTE-NC to 10% minority (in-fold) | 0.91607 | 0.3319 | strongly harmful |
| screen | target encoding of every numeric column's exact value; low-cardinality numerics as categorical (fast LightGBM, lr 0.05) | *running at time of writing* | | see Appendix A if filled in |

### Spearman correlation of OOF predictions

| | LogReg | LightGBM v4 | XGBoost | CatBoost |
|---|---|---|---|---|
| LogReg | 1 | 0.938 | 0.939 | 0.947 |
| LightGBM v4 | | 1 | 0.995 | 0.985 |
| XGBoost | | | 1 | 0.986 |
| CatBoost | | | | 1 |

### Operating points of the v7 blend (OOF)

| Flag top | Recall | Precision |
|---|---|---|
| 0.5% | 24.9% | 83.2% |
| 1% | 38.9% | 65.0% |
| 2% | 54.0% | 45.1% |
| 5% | 71.5% | 23.9% |
| 10% | 83.1% | 13.9% |
| 20% | 91.7% | 7.7% |

---

## 5. Observations and my current hypotheses

1. **Plateau:** a linear model gets 0.942, and every tree model and tuning variant lands in 0.9462 to 0.9474. The spread across very different model families is only ~0.005. That suggests we're near the information ceiling of these features (label noise from the generator), not limited by model capacity.
2. **Mostly additive signal:** logistic regression is within 0.005 of boosting, so the log-odds are probably close to an additive function of the features (the original `gen.py` is likely a logistic model on hand-picked features plus noise). The boosting gain may come mainly from non-linear shapes of single features (thresholds in merchant_risk_score, velocity_score, customer_age) rather than interactions.
3. **Engineered features and imbalance handling don't help**, which is consistent with (1) and (2).
4. **Fast overfitting:** at lr 0.05 the model peaks around 300 trees. Strong regularisation helps (+0.002).
5. **Uninformative features** (channel, device_type, card_type, time_of_day_hour, day_of_week, distance_from_home_km, card_age_months, account_balance_usd) are probably pure noise added by the original generator. I haven't tried dropping them.
6. The **original 20k dataset** hasn't been added to training yet (the file is being re-downloaded). In other Playground episodes this typically gives +0.0005 to +0.002.
7. **Not yet tried:** Optuna tuning, seed averaging, neural networks (MLP / TabM / FT-Transformer), GAM / spline-logistic models (e.g. an EBM), stacking with a meta-model, dropping noise features, monotone constraints, binning numerics, pseudo-labelling, k-NN features, LightGBM `extra_trees` / DART, larger CatBoost with tuned settings.

---

## 6. Constraints

- Must stay leak-free: anything using labels has to be computed in-fold.
- No external data except the original CC0 dataset (allowed in Playground competitions).
- CPU only (16 cores). Runs of up to ~1 hour per experiment are fine.
- I need to explain every step in a beginner's guide, so I prefer ideas with a clear rationale over opaque leaderboard hacks.

---

## 7. What I'd like from you

1. **Is 0.96 CV AUC realistic** for this data, given the plateau above? How would you estimate the ceiling (e.g. the Bayes-optimal AUC under the suspected generator)? What evidence would convince you either way?
2. **The top 5 to 10 concrete experiments, ranked** by expected gain per hour of compute. For each: what to do (exact settings where possible), why it should help here, the expected gain, and how to validate it without leakage.
3. **The original dataset:** best way to use it (append with or without a flag column, weighting, using it only for target encoding, using its fitted logistic structure as a feature...).
4. **Reverse-engineering the generator:** can the shape of the signal (logistic regression being so close to boosting) be exploited, e.g. with a GAM / spline logistic model, or by fitting the presumed `gen.py` formula as a feature?
5. **Model diversity for the ensemble:** which genuinely different model families should I add (NN, EBM, k-NN, SVM...), and how should I blend or stack them (rank average, logistic stacking, hill climbing)?
6. **Anything in my setup that looks wrong** or could be hiding performance (CV design, early stopping, categorical handling, hyperparameters).
7. **Playground-specific tricks** that have worked in similar binary-classification episodes (e.g. numeric-as-categorical, rounding artifacts, original-data merge, multi-seed / multi-fold averaging).

Please give answers I can implement directly in Python (scikit-learn / LightGBM / XGBoost / CatBoost / PyTorch), with code snippets where useful.

---

## Appendix A: latest screening results

Fast LightGBM (lr 0.05, otherwise v4 settings, `metric="auc"`, patience 200), same 5 folds. Target encoding = `sklearn.preprocessing.TargetEncoder(target_type="binary", cv=5)` on each numeric column cast to string (exact value as category), fitted inside each training fold (cross-fitted), applied to the validation fold. Added as 13 extra columns next to the raw features.

| Idea | CV AUC | Fold AUCs | vs base |
|---|---|---|---|
| base (raw features) | 0.94716 | 0.94548, 0.94922, 0.95174, 0.94736, 0.94232 | n/a |
| + target encoding of all 13 numeric columns' exact values | **0.94791** | 0.94660, 0.94965, 0.95248, 0.94872, 0.94257 | **+0.00075, 5/5 folds** |
| + low-cardinality numerics also as categorical (txn_count_last_24h, cvv_retry_count, prior_disputes, card_age_months, customer_age, time_of_day_hour, day_of_week as extra LightGBM categorical columns) | 0.94652 | 0.94447, 0.94821, 0.95163, 0.94702, 0.94149 | −0.00064, 0/5 folds |
| both | 0.94707 | 0.94569, 0.94852, 0.95220, 0.94784, 0.94152 | −0.00009 |

So: target encoding of exact numeric values helps; letting LightGBM treat low-cardinality numerics as native categoricals hurts (probably overfits the category splits) and cancels out the target-encoding gain when combined.

This is the first feature idea that helped. It suggests the synthetic generator left value-level artifacts (specific exact values carrying fraud signal). Applied to the slower v4 settings and blended, it should lift the best CV to roughly 0.948. Ideas for pushing this direction further are welcome.
