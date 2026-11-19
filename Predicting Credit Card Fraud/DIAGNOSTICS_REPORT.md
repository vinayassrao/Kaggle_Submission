# Diagnostics report: Kaggle Playground S6E11 (reply to the follow-up request)

All numbers come from code in this folder: `diagnostics.py` (sections 1.3, 3, 4.1 coverage, 4.3, 5) and `diag_te.py` (sections 1.3 TE rows, 4.1 importance, 4.2). Same 5 folds everywhere: `StratifiedKFold(5, shuffle=True, random_state=42)`.

**Section 2 (original 20k dataset) is pending:** the file is not on disk right now. I'll run it as soon as it's re-downloaded. Everything else is below.

## Key findings at a glance

1. **The noise ceiling is about 0.945 to 0.95.** Treating the (very well calibrated) `oof_v4` as the true probabilities and simulating labels gives a mean AUC of **0.9443** (sd 0.0012), which is *below* the observed 0.9474. Calibration is near perfect (mean prediction 0.01662 vs rate 0.01670, and every bin matches). In my view 0.96 isn't reachable without genuinely new signal (see 3.3).
2. **Missed frauds look clean.** Only 1.42% of frauds fall in the bottom 50% of scores (8.4% in the bottom 80%). The lowest-scored frauds have cvv 0, no flags, low risk and low velocity: a random baseline fraud floor, not a hidden pattern (3.2).
3. **Concatenated vs per-fold scoring makes almost no difference:** within-fold rank or log-odds standardisation changes AUC by ≤ 0.0001 (1.3).
4. **The generator left artifacts:**
   - `velocity_score == 0` (48,499 rows, 6.9%) has a 0.04% fraud rate vs 0.35% for nearby values (a clipping or zero-inflation point mass). The GBDT's largest disagreement with LogReg is exactly `velocity_score <= 0.05` (5.2).
   - `merchant_risk_score` has popular exact values (31.2 to 32.2) with fraud rates of 0.4% to 0.9%, against 1.2% to 1.6% for other popular values (4.3).
   - `velocity_score` and `merchant_risk_score` always have exactly 1 decimal, and the other floats have 2.
5. **Non-linear shapes LogReg misses:** monotone but kinked risk curves. `customer_age` jumps at 34 (0.5% → 1.4%). `velocity_score` has a jump at 0 and accelerates above 33. `merchant_risk_score` is convex, and the GBDT corrects LogReg's over-prediction above 80. `hours_since_last_txn` is high only below ~0.5 h. `cvv_retry_count` 2+ ≈ 40% fraud (5.1, 5.2).
6. **Row order is i.i.d.:** AUC of `id` = 0.4986, and the fraud rate is flat across id deciles (5.3).
7. **The target-encoding gain (+0.00075) comes from two columns:** `hours_since_last_txn` (−0.00064 when dropped) and `customer_age` (−0.00055), plus a little from `account_balance_usd` and `amount_usd`. These are exact-value effects that the monotone trend misses. TE on noise columns gives ≈ 0, and `card_age_months` TE is slightly harmful. With TE the model also early-stops much sooner (500 to 700 vs 700 to 1,070 trees at lr 0.05) (4.1, 4.2).

---

## 1.1 Core pipeline code

### Shared CV loop (`common.py`)

```python
SEED, N_FOLDS = 42, 5
CATS = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
FLAGS = [... 6 boolean columns, cast to int ...]
NUMS = [... 13 numeric columns ...]

def raw_features(train, test):
    cols = NUMS + FLAGS + CATS
    X, X_test = train[cols].copy(), test[cols].copy()
    for c in CATS:   # pandas 'category' dtype, categories fixed from train
        cat_type = pd.CategoricalDtype(sorted(train[c].unique()))
        X[c], X_test[c] = X[c].astype(cat_type), X_test[c].astype(cat_type)
    return X, X_test

def run_cv(name, fit_predict, X, X_test, y, description=""):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X)); test_pred = np.zeros(len(X_test))
    for k, (tr, va) in enumerate(skf.split(X, y)):
        p_va, p_te = fit_predict(X.iloc[tr], y[tr], X.iloc[va], y[va], X_test)
        oof[va] = p_va
        test_pred += np.asarray(p_te) / N_FOLDS          # plain mean of fold probabilities
    auc = roc_auc_score(y, oof)                          # concatenated OOF AUC
```

### v4 LightGBM: categoricals passed as pandas `category` (LightGBM auto-detects them)

```python
PARAMS = dict(objective="binary", n_estimators=10000, learning_rate=0.02, num_leaves=31, max_depth=4,
              min_child_samples=300, colsample_bytree=0.4, subsample=0.7, subsample_freq=1,
              reg_lambda=10, cat_smooth=50, random_state=42, verbose=-1, n_jobs=-1)
model = lgb.LGBMClassifier(**PARAMS)
model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
          callbacks=[lgb.early_stopping(300, verbose=False)])
# NOTE: no metric="auc" in PARAMS, so early stopping watched both binary_logloss and auc
# (stops when either stalls). Re-running with metric="auc" gave identical fold scores (0.94737).
p_va, p_te = model.predict_proba(X_va)[:, 1], model.predict_proba(X_te)[:, 1]   # best_iteration used
```

### v5 XGBoost: native categorical (`enable_categorical=True`, `max_cat_to_onehot=1` = partition splits)

```python
PARAMS = dict(objective="binary:logistic", eval_metric="auc", learning_rate=0.02, max_depth=4,
              min_child_weight=10, subsample=0.7, colsample_bytree=0.5, reg_lambda=10,
              n_estimators=10000, early_stopping_rounds=300, tree_method="hist",
              enable_categorical=True, max_cat_to_onehot=1, random_state=42, n_jobs=-1)
model = xgb.XGBClassifier(**PARAMS); model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
```

### v6 CatBoost: categoricals cast to `str`, passed via `cat_features`

```python
PARAMS = dict(iterations=10000, learning_rate=0.05, depth=6, l2_leaf_reg=10, eval_metric="AUC",
              od_type="Iter", od_wait=300, random_seed=42, thread_count=-1, verbose=False)
model = CatBoostClassifier(**PARAMS, cat_features=CATS)
model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
```

### v7 blend

Each model's OOF and test predictions are converted to global ranks scaled to [0, 1] (`(rankdata(a) - 1) / (n - 1)`). The weights `w` (abs-normalised) maximise `roc_auc_score(y, oof_ranks @ w)` with Nelder-Mead from equal weights.

## 1.2 Appendix A TargetEncoder (exact)

```python
from sklearn.preprocessing import TargetEncoder
# fast LightGBM for the screen: lr=0.05, metric="auc", other params as v4, early_stopping(200)
te = TargetEncoder(target_type="binary", cv=5, shuffle=True, random_state=42)   # smooth="auto" (default)
names = [f"{c}_te" for c in NUMS]                            # all 13 numeric columns
X_tr[names] = te.fit_transform(X_tr[NUMS].astype(str), y_tr)  # cross-fitted inside the training fold
X_va[names] = te.transform(X_va[NUMS].astype(str))            # encoder fit on the full training fold
# raw numeric columns are kept; the TE columns are added alongside them
```

Values are cast with `.astype(str)`, so each exact float value is one category (pandas prints floats with their shortest representation, e.g. `31.2`). `fit_transform` uses internal 5-fold cross-fitting, so training rows never see their own label. Validation rows are encoded by the encoder fitted on the whole training fold.

---

## 1.3 Concatenated vs per-fold OOF AUC

| model                  |   mean_fold_auc |   concat_raw |   concat_within_fold_rank |   concat_logodds_std |
|:-----------------------|----------------:|-------------:|--------------------------:|---------------------:|
| v1_logistic_regression |         0.9419  |      0.94187 |                   0.9419  |              0.94191 |
| v4_lightgbm_tuned      |         0.94741 |      0.94737 |                   0.94741 |              0.9474  |
| v5_xgboost             |         0.94721 |      0.94712 |                   0.94721 |              0.94723 |
| v6_catboost            |         0.94625 |      0.94619 |                   0.94625 |              0.94624 |
| v7_ensemble            |         0.94749 |      0.94744 |                   0.94749 |              0.94748 |
| screen_base            |         0.94722 |      0.94716 |                   0.94722 |              0.94722 |
| screen_te_nums         |         0.948   |      0.94791 |                   0.94801 |              0.94801 |



## 2. Original 20k dataset

*Pending: `credit_card_fraud_2026.csv` is not on disk at the moment. Sections 2.1 to 2.5 will be run as soon as it's re-downloaded. Source: https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026*

## 3.1 Calibration / reliability of oof_v4 (20 equal-frequency bins)

|   bin |   count |   fraud_count |   mean_pred |   actual_rate |   p_min |   p_max |   share_of_frauds |
|------:|--------:|--------------:|------------:|--------------:|--------:|--------:|------------------:|
|     0 |   34998 |             0 |     5e-05   |       0       | 0       | 8e-05   |           0       |
|     1 |   34998 |             1 |     0.00012 |       3e-05   | 8e-05   | 0.00017 |           9e-05   |
|     2 |   34998 |             9 |     0.00021 |       0.00026 | 0.00017 | 0.00026 |           0.00077 |
|     3 |   34998 |            13 |     0.00032 |       0.00037 | 0.00026 | 0.00038 |           0.00111 |
|     4 |   34998 |             8 |     0.00044 |       0.00023 | 0.00038 | 0.00051 |           0.00068 |
|     5 |   34998 |            12 |     0.00058 |       0.00034 | 0.00051 | 0.00066 |           0.00103 |
|     6 |   34998 |            13 |     0.00075 |       0.00037 | 0.00066 | 0.00085 |           0.00111 |
|     7 |   34998 |            31 |     0.00096 |       0.00089 | 0.00085 | 0.00107 |           0.00265 |
|     8 |   34998 |            31 |     0.0012  |       0.00089 | 0.00107 | 0.00134 |           0.00265 |
|     9 |   34998 |            48 |     0.0015  |       0.00137 | 0.00134 | 0.00168 |           0.00411 |
|    10 |   34998 |            64 |     0.00188 |       0.00183 | 0.00168 | 0.00211 |           0.00548 |
|    11 |   34998 |            67 |     0.00237 |       0.00191 | 0.00211 | 0.00266 |           0.00573 |
|    12 |   34998 |           106 |     0.00302 |       0.00303 | 0.00266 | 0.00341 |           0.00907 |
|    13 |   34998 |           118 |     0.00391 |       0.00337 | 0.00341 | 0.00447 |           0.01009 |
|    14 |   34998 |           201 |     0.0052  |       0.00574 | 0.00447 | 0.00604 |           0.0172  |
|    15 |   34998 |           262 |     0.0072  |       0.00749 | 0.00604 | 0.0086  |           0.02241 |
|    16 |   34998 |           375 |     0.01064 |       0.01071 | 0.0086  | 0.01318 |           0.03208 |
|    17 |   34998 |           616 |     0.01745 |       0.0176  | 0.01318 | 0.02323 |           0.0527  |
|    18 |   34998 |          1361 |     0.03607 |       0.03889 | 0.02323 | 0.05754 |           0.11643 |
|    19 |   34998 |          8353 |     0.23849 |       0.23867 | 0.05754 | 0.99924 |           0.7146  |


Top tails:

| bucket   |   count |   fraud_count |   mean_pred |   actual_rate |   share_of_frauds |
|:---------|--------:|--------------:|------------:|--------------:|------------------:|
| top 5.0% |   34998 |          8353 |     0.23849 |       0.23867 |           0.7146  |
| top 2.0% |   14000 |          6308 |     0.44956 |       0.45057 |           0.53965 |
| top 1.0% |    7000 |          4558 |     0.64989 |       0.65114 |           0.38994 |
| top 0.5% |    3500 |          2911 |     0.82283 |       0.83171 |           0.24904 |
| top 0.1% |     700 |           682 |     0.96999 |       0.97429 |           0.05835 |


Overall mean prediction 0.01662 vs fraud rate 0.01670. Brier 0.01097.


## 3.2 Low-score frauds

- Frauds in bottom 50% of oof_v4: **166 = 1.42%** of all frauds (fraud rate inside that bottom 50%: 0.047%)

- Frauds in bottom 80% of oof_v4: **984 = 8.42%** of all frauds (fraud rate inside that bottom 80%: 0.176%)


False negatives in bottom 50%: n=166; true negatives in bottom 50%: n=349,814.


| feature                           |   FN_bottom50_mean |   TN_bottom50_mean |   all_train_mean |   FN_minus_TN |
|:----------------------------------|-------------------:|-------------------:|-----------------:|--------------:|
| amount_usd                        |            90.2927 |            63.7009 |          79.8931 |       26.5918 |
| hours_since_last_txn              |             7.0501 |             7.1298 |           6.6337 |       -0.0797 |
| txn_count_last_24h                |             2.9699 |             2.6424 |           2.9841 |        0.3274 |
| distance_from_home_km             |            14.0277 |            13.5101 |          13.5876 |        0.5176 |
| card_age_months                   |            46.753  |            46.8283 |          46.863  |       -0.0753 |
| customer_age                      |            53.4699 |            47.3085 |          51.4303 |        6.1614 |
| account_balance_usd               |          2196.6    |          2024.81   |        2065      |      171.788  |
| cvv_retry_count                   |             0.012  |             0.019  |           0.0855 |       -0.007  |
| velocity_score                    |            17.2645 |            14.8446 |          18.9294 |        2.4198 |
| time_of_day_hour                  |            11.9337 |            11.8064 |          11.7619 |        0.1273 |
| day_of_week                       |             2.7831 |             2.9568 |           2.9817 |       -0.1737 |
| merchant_risk_score               |            32.8434 |            29.8156 |          35.2176 |        3.0278 |
| prior_disputes                    |             0.1506 |             0.1269 |           0.1528 |        0.0237 |
| is_foreign_transaction            |             0.006  |             0.005  |           0.0173 |        0.001  |
| is_new_merchant                   |             0.0602 |             0.0747 |           0.1525 |       -0.0145 |
| used_vpn                          |             0.0301 |             0.013  |           0.0338 |        0.0172 |
| ip_country_mismatch               |             0      |             0.0029 |           0.0162 |       -0.0029 |
| billing_shipping_mismatch         |             0      |             0.002  |           0.0129 |       -0.002  |
| is_ai_generated_scam_attempt      |             0      |             0.0005 |           0.0033 |       -0.0005 |
| merchant_category=Crypto Exchange |             0      |             0.0015 |           0.0076 |       -0.0015 |
| merchant_category=Electronics     |             0.0482 |             0.0355 |           0.0696 |        0.0127 |
| merchant_category=Fuel            |             0.0542 |             0.0946 |           0.0855 |       -0.0403 |
| merchant_category=Gaming          |             0.0181 |             0.0108 |           0.034  |        0.0073 |
| merchant_category=Gift Cards      |             0.006  |             0.0029 |           0.0135 |        0.0031 |
| merchant_category=Groceries       |             0.1928 |             0.2359 |           0.2037 |       -0.0431 |
| merchant_category=Healthcare      |             0.0422 |             0.0559 |           0.0513 |       -0.0138 |
| merchant_category=Online Retail   |             0.2952 |             0.2447 |           0.2216 |        0.0505 |
| merchant_category=Restaurants     |             0.1807 |             0.1626 |           0.1448 |        0.0181 |
| merchant_category=Streaming       |             0.0542 |             0.0554 |           0.0674 |       -0.0012 |
| merchant_category=Travel          |             0.0361 |             0.0381 |           0.041  |       -0.002  |
| merchant_category=Utilities       |             0.0723 |             0.062  |           0.06   |        0.0103 |
| card_type=Amex                    |             0.0783 |             0.0605 |           0.0654 |        0.0178 |
| card_type=Discover                |             0.0361 |             0.026  |           0.0246 |        0.0101 |
| card_type=Mastercard              |             0.3554 |             0.3698 |           0.3641 |       -0.0144 |
| card_type=RuPay                   |             0.0602 |             0.0527 |           0.053  |        0.0075 |
| card_type=Visa                    |             0.4699 |             0.4909 |           0.4929 |       -0.0211 |
| auth_method=3D Secure             |             0.2651 |             0.2833 |           0.3001 |       -0.0182 |
| auth_method=Biometric             |             0.1627 |             0.1868 |           0.1338 |       -0.0241 |
| auth_method=No Authentication     |             0.0542 |             0.0347 |           0.0958 |        0.0195 |
| auth_method=OTP                   |             0.253  |             0.2436 |           0.2503 |        0.0094 |
| auth_method=PIN                   |             0.2651 |             0.2515 |           0.22   |        0.0135 |
| channel=ATM                       |             0.0241 |             0.0419 |           0.0432 |       -0.0178 |
| channel=Contactless               |             0.1506 |             0.1519 |           0.1509 |       -0.0013 |
| channel=In-App                    |             0.1265 |             0.121  |           0.1196 |        0.0055 |
| channel=Online                    |             0.488  |             0.4025 |           0.3914 |        0.0855 |
| channel=POS                       |             0.2108 |             0.2827 |           0.295  |       -0.0718 |
| device_type=ATM Machine           |             0.0602 |             0.0282 |           0.0298 |        0.032  |
| device_type=Android Phone         |             0.3614 |             0.3493 |           0.3392 |        0.0121 |
| device_type=Mac                   |             0.0422 |             0.048  |           0.0485 |       -0.0058 |
| device_type=POS Terminal          |             0.1205 |             0.1587 |           0.1618 |       -0.0382 |
| device_type=Smart Watch           |             0.0361 |             0.0295 |           0.0302 |        0.0067 |
| device_type=Tablet                |             0.0241 |             0.0393 |           0.0396 |       -0.0152 |
| device_type=Windows PC            |             0.1084 |             0.1182 |           0.1156 |       -0.0098 |
| device_type=iPhone                |             0.247  |             0.2287 |           0.2353 |        0.0183 |


**15 lowest-scored actual frauds**

|     id |      p |   amount_usd |   hours_since_last_txn |   txn_count_last_24h |   distance_from_home_km |   card_age_months |   customer_age |   account_balance_usd |   cvv_retry_count |   velocity_score |   time_of_day_hour |   day_of_week |   merchant_risk_score |   prior_disputes |   is_foreign_transaction |   is_new_merchant |   used_vpn |   ip_country_mismatch |   billing_shipping_mismatch |   is_ai_generated_scam_attempt | merchant_category   | card_type   | auth_method       | channel   | device_type   |
|-------:|-------:|-------------:|-----------------------:|---------------------:|------------------------:|------------------:|---------------:|----------------------:|------------------:|-----------------:|-------------------:|--------------:|----------------------:|-----------------:|-------------------------:|------------------:|-----------:|----------------------:|----------------------------:|-------------------------------:|:--------------------|:------------|:------------------|:----------|:--------------|
| 375919 | 0.0001 |        28.22 |                  17.04 |                    3 |                   24.79 |                56 |             49 |               1310.88 |                 0 |             11.6 |                 18 |             4 |                  21.4 |                1 |                        0 |                 0 |          0 |                     0 |                           0 |                              0 | Online Retail       | Visa        | Biometric         | Online    | POS Terminal  |
| 626714 | 0.0002 |        25.91 |                   3.75 |                    2 |                   16.88 |                43 |             57 |               3201.48 |                 0 |              7   |                  4 |             3 |                  17   |                0 |                        0 |                 0 |          0 |                     0 |                           0 |                              0 | Restaurants         | Visa        | 3D Secure         | In-App    | Android Phone |
| 668156 | 0.0002 |        55.96 |                   7.08 |                    2 |                   20.59 |                47 |             48 |               1542.11 |                 0 |             10.8 |                 16 |             1 |                  21.5 |                0 |                        0 |                 0 |          0 |                     0 |                           0 |                              0 | Groceries           | Amex        | 3D Secure         | ATM       | POS Terminal  |
| 226293 | 0.0002 |         8.09 |                   5.46 |                    0 |                    5.82 |                54 |             48 |                881.72 |                 0 |              0   |                 19 |             1 |                  42.3 |                0 |                        0 |                 0 |          0 |                     0 |                           0 |                              0 | Online Retail       | Mastercard  | No Authentication | Online    | POS Terminal  |
| 405262 | 0.0002 |        20.83 |                   0.68 |                    4 |                    3.44 |                48 |             74 |                661.12 |                 0 |             21.4 |                 12 |             2 |                  24   |                0 |                        0 |                 0 |          0 |                     0 |                           0 |                              0 | Online Retail       | Visa        | Biometric         | ATM       | iPhone        |
| 121166 | 0.0002 |       638.28 |                   4.74 |                    6 |                   30.13 |                46 |             23 |               1758.24 |                 0 |             25   |                 17 |             2 |                  24.1 |                0 |                        0 |                 0 |          0 |                     0 |                           0 |                              0 | Travel              | Visa        | PIN               | In-App    | iPhone        |
|   1958 | 0.0002 |        28.28 |                   8.7  |                    1 |                   23.73 |                44 |             75 |                757.14 |                 0 |              1   |                 11 |             3 |                  37.8 |                1 |                        0 |                 0 |          0 |                     0 |                           0 |                              0 | Utilities           | Visa        | 3D Secure         | Online    | POS Terminal  |
| 582033 | 0.0002 |        16.92 |                   5.53 |                    1 |                    6.89 |                50 |             26 |               6996.07 |                 0 |              8.6 |                  1 |             2 |                  37.3 |                0 |                        0 |                 0 |          0 |                     0 |                           0 |                              0 | Online Retail       | Amex        | PIN               | Online    | Android Phone |
| 271077 | 0.0002 |        33.6  |                   4.24 |                    4 |                   12.39 |                49 |             48 |               2353.43 |                 0 |             12.7 |                 22 |             3 |                  28   |                0 |                        0 |                 0 |          0 |                     0 |                           0 |                              0 | Online Retail       | Mastercard  | OTP               | Online    | Tablet        |
|  42229 | 0.0003 |        16.84 |                   9.87 |                    2 |                   17.27 |                42 |             19 |               3365.83 |                 0 |              2.8 |                  8 |             1 |                  35.7 |                0 |                        0 |                 0 |          0 |                     0 |                           0 |                              0 | Restaurants         | Amex        | No Authentication | Online    | iPhone        |
|  29768 | 0.0003 |        40.91 |                   8.88 |                    3 |                    1.95 |                51 |             41 |               1777.8  |                 0 |             18.6 |                 18 |             4 |                  22.9 |                0 |                        0 |                 0 |          0 |                     0 |                           0 |                              0 | Streaming           | Visa        | Biometric         | Online    | Android Phone |
| 326219 | 0.0003 |         7.14 |                   3.8  |                    1 |                    1.79 |                46 |             81 |               3291.73 |                 0 |              0   |                 11 |             3 |                  46.9 |                0 |                        0 |                 0 |          0 |                     0 |                           0 |                              0 | Fuel                | Visa        | 3D Secure         | Online    | Android Phone |
| 264238 | 0.0003 |        38.32 |                  16.41 |                    3 |                    5.41 |                56 |             79 |               1439.75 |                 0 |             11.7 |                 15 |             1 |                  17.7 |                0 |                        0 |                 0 |          0 |                     0 |                           0 |                              0 | Restaurants         | Mastercard  | PIN               | Online    | Smart Watch   |
| 242953 | 0.0003 |        36.3  |                   4.73 |                    6 |                    1.23 |                56 |             68 |               1345.13 |                 0 |             22.6 |                 15 |             4 |                  19.9 |                0 |                        0 |                 0 |          0 |                     0 |                           0 |                              0 | Groceries           | Mastercard  | 3D Secure         | In-App    | Android Phone |
| 653934 | 0.0003 |        18.03 |                   8.5  |                    1 |                   25.88 |                41 |             54 |               1637.26 |                 0 |              2.1 |                 12 |             5 |                  32.4 |                1 |                        0 |                 0 |          0 |                     0 |                           0 |                              0 | Groceries           | RuPay       | 3D Secure         | In-App    | Android Phone |


**15 highest-scored non-frauds**

|     id |      p |   amount_usd |   hours_since_last_txn |   txn_count_last_24h |   distance_from_home_km |   card_age_months |   customer_age |   account_balance_usd |   cvv_retry_count |   velocity_score |   time_of_day_hour |   day_of_week |   merchant_risk_score |   prior_disputes |   is_foreign_transaction |   is_new_merchant |   used_vpn |   ip_country_mismatch |   billing_shipping_mismatch |   is_ai_generated_scam_attempt | merchant_category   | card_type   | auth_method       | channel     | device_type   |
|-------:|-------:|-------------:|-----------------------:|---------------------:|------------------------:|------------------:|---------------:|----------------------:|------------------:|-----------------:|-------------------:|--------------:|----------------------:|-----------------:|-------------------------:|------------------:|-----------:|----------------------:|----------------------------:|-------------------------------:|:--------------------|:------------|:------------------|:------------|:--------------|
| 552342 | 0.9818 |        43.64 |                   0.01 |                    4 |                    4.9  |                50 |             75 |                708.81 |                 1 |             54.4 |                 19 |             2 |                  57.4 |                0 |                        0 |                 1 |          1 |                     0 |                           0 |                              0 | Groceries           | Visa        | No Authentication | Contactless | Tablet        |
| 365390 | 0.9812 |       217.48 |                  20.09 |                    4 |                    0.67 |                45 |             56 |               1568.53 |                 1 |             15.1 |                  1 |             3 |                  67.1 |                0 |                        1 |                 1 |          1 |                     0 |                           0 |                              0 | Electronics         | Mastercard  | No Authentication | POS         | POS Terminal  |
|  47568 | 0.9801 |        84.13 |                   0.38 |                    4 |                   13.49 |                50 |             69 |               4635.11 |                 1 |             36.4 |                 18 |             2 |                  59.2 |                0 |                        0 |                 1 |          0 |                     0 |                           1 |                              0 | Gaming              | Visa        | No Authentication | Online      | Android Phone |
| 422247 | 0.9796 |        17.61 |                   4.86 |                    5 |                    2.24 |                34 |             71 |               1172.86 |                 2 |             33.5 |                 17 |             1 |                  59.6 |                1 |                        0 |                 0 |          0 |                     0 |                           0 |                              0 | Streaming           | Mastercard  | No Authentication | Online      | Android Phone |
| 548000 | 0.9775 |        34.58 |                   8.07 |                    4 |                   20.07 |                47 |             70 |               3388.13 |                 0 |             31.6 |                  8 |             3 |                  58.8 |                1 |                        1 |                 1 |          1 |                     1 |                           0 |                              0 | Online Retail       | Mastercard  | 3D Secure         | In-App      | iPhone        |
| 251528 | 0.9768 |        22.3  |                   1.86 |                    3 |                   35.34 |                53 |             80 |                551.26 |                 2 |             28.4 |                  8 |             6 |                  54.9 |                0 |                        0 |                 0 |          0 |                     1 |                           0 |                              0 | Fuel                | Visa        | No Authentication | Online      | iPhone        |
|  34204 | 0.9748 |         2.68 |                   0.12 |                    3 |                   26.85 |                47 |             59 |                681.18 |                 2 |             37.6 |                  7 |             0 |                  39.7 |                0 |                        0 |                 0 |          1 |                     0 |                           0 |                              0 | Streaming           | Visa        | No Authentication | POS         | Mac           |
| 687230 | 0.9733 |        31.93 |                   5.34 |                    3 |                   24.42 |                42 |             58 |               3254.57 |                 1 |             26.3 |                 19 |             4 |                  39.1 |                0 |                        1 |                 1 |          1 |                     1 |                           0 |                              0 | Restaurants         | Mastercard  | OTP               | In-App      | Windows PC    |
| 261264 | 0.9665 |        49.38 |                   4.44 |                    6 |                    8.58 |                46 |             67 |                821.32 |                 1 |             37.2 |                  6 |             3 |                  52.3 |                0 |                        0 |                 0 |          0 |                     0 |                           0 |                              1 | Gaming              | Amex        | No Authentication | Online      | POS Terminal  |
| 494432 | 0.9609 |        41.36 |                   1.87 |                    6 |                   10.14 |                37 |             68 |                582.83 |                 1 |             42.4 |                 22 |             6 |                  63.9 |                0 |                        0 |                 0 |          1 |                     0 |                           0 |                              0 | Gift Cards          | Visa        | No Authentication | Online      | POS Terminal  |
| 682865 | 0.9555 |       563.44 |                  13.8  |                    2 |                   43.94 |                40 |             76 |                454.85 |                 1 |             17.6 |                 10 |             3 |                  41.4 |                1 |                        0 |                 1 |          0 |                     0 |                           0 |                              1 | Electronics         | Mastercard  | No Authentication | Online      | iPhone        |
| 353918 | 0.9546 |        14.18 |                  13.36 |                    5 |                   19.61 |                40 |             70 |               3170.55 |                 1 |             25.3 |                 12 |             0 |                  44.3 |                0 |                        1 |                 1 |          0 |                     1 |                           0 |                              0 | Streaming           | Visa        | PIN               | POS         | Android Phone |
| 443528 | 0.9533 |        47.89 |                  13.11 |                    3 |                    9.54 |                55 |             58 |               2081.3  |                 1 |             27.8 |                 10 |             0 |                  59.7 |                0 |                        0 |                 1 |          1 |                     0 |                           0 |                              0 | Electronics         | Visa        | No Authentication | Online      | Tablet        |
| 102704 | 0.9511 |        17.53 |                  12.22 |                    4 |                   14.68 |                37 |             59 |               5133.87 |                 1 |             21.7 |                  8 |             1 |                  35.8 |                0 |                        1 |                 1 |          0 |                     1 |                           0 |                              0 | Utilities           | Visa        | No Authentication | POS         | Android Phone |
| 359963 | 0.9502 |       166.89 |                   1.81 |                    3 |                    2.64 |                32 |             53 |                928.86 |                 1 |             21.8 |                  8 |             4 |                  62.7 |                0 |                        1 |                 1 |          0 |                     1 |                           0 |                              0 | Electronics         | Visa        | 3D Secure         | POS         | POS Terminal  |


## 3.3 Simulated AUC if oof_v4 were the true probabilities

y_sim ~ Bernoulli(oof_v4), 10 draws: mean AUC **0.94433** (sd 0.00117); observed AUC 0.94737.


## 4.1 Exact-value coverage of numeric columns (+ TE correlation)

| column                |   n_unique_train |   n_unique_test |   pct_test_value_in_train |   pct_test_value_seen_ge10 |   spearman_col_vs_te |   auc_of_te_alone |   auc_of_raw_alone |
|:----------------------|-----------------:|----------------:|--------------------------:|---------------------------:|---------------------:|------------------:|-------------------:|
| amount_usd            |            13387 |           10510 |                   99.4606 |                    96.7005 |              -0.017  |            0.5323 |             0.5448 |
| hours_since_last_txn  |             3633 |            3331 |                   99.947  |                    99.3553 |              -0.4111 |            0.5831 |             0.5682 |
| txn_count_last_24h    |               12 |              11 |                  100      |                   100      |               0.98   |            0.6185 |             0.6203 |
| distance_from_home_km |             4365 |            4108 |                   99.957  |                    99.45   |              -0.0479 |            0.511  |             0.5039 |
| card_age_months       |               49 |              49 |                   99.9997 |                    99.9973 |               0.1519 |            0.5104 |             0.5019 |
| customer_age          |               64 |              64 |                  100      |                   100      |               0.8447 |            0.6452 |             0.6247 |
| account_balance_usd   |            43317 |           31803 |                   97.7772 |                    87.8584 |              -0.0291 |            0.5276 |             0.5063 |
| cvv_retry_count       |                4 |               4 |                  100      |                   100      |               0.4846 |            0.7057 |             0.7081 |
| velocity_score        |              666 |             644 |                   99.996  |                    99.9433 |               0.9255 |            0.7053 |             0.7108 |
| time_of_day_hour      |               24 |              24 |                  100      |                   100      |              -0.095  |            0.5018 |             0.5002 |
| day_of_week           |                7 |               7 |                  100      |                   100      |               0.7496 |            0.5047 |             0.5081 |
| merchant_risk_score   |              950 |             917 |                   99.9957 |                    99.9197 |               0.9176 |            0.7485 |             0.7489 |
| prior_disputes        |                5 |               5 |                  100      |                    99.9993 |               0.6182 |            0.5295 |             0.5322 |


(TE here = cross-fitted 5-fold TargetEncoder on all of train, only for these coverage/correlation stats.)



## 4.1 (cont.) Importance of the TE columns in the Appendix A model (fold-1 model, fold-1 validation)

| column                |   raw_gain_pct |   te_gain_pct |   raw_perm_auc_drop |   te_perm_auc_drop |   cv_auc_when_this_TE_dropped |   loss_vs_full_TE |
|:----------------------|---------------:|--------------:|--------------------:|-------------------:|------------------------------:|------------------:|
| hours_since_last_txn  |          1.017 |         0.867 |             0.00312 |            0.00187 |                       0.94727 |           0.00064 |
| customer_age          |          1.786 |         3.392 |             0.00071 |            0.01073 |                       0.94736 |           0.00055 |
| account_balance_usd   |          0.375 |         0.459 |             5e-05   |            0.00014 |                       0.94775 |           0.00017 |
| amount_usd            |          0.725 |         0.556 |             0.00041 |            0.00044 |                       0.94782 |           9e-05   |
| cvv_retry_count       |         14.585 |         6.822 |             0.01255 |            0.00229 |                       0.94787 |           4e-05   |
| distance_from_home_km |          0.408 |         0.474 |             1e-05   |            0.00012 |                       0.9479  |           1e-05   |
| time_of_day_hour      |          0.206 |         0.336 |             2e-05   |           -5e-05   |                       0.94793 |          -2e-05   |
| txn_count_last_24h    |          0.21  |         0.464 |             7e-05   |            8e-05   |                       0.94793 |          -2e-05   |
| day_of_week           |          0.093 |         0.194 |             1e-05   |           -1e-05   |                       0.94794 |          -3e-05   |
| merchant_risk_score   |         10.908 |         6.315 |             0.02715 |            0.00459 |                       0.94795 |          -4e-05   |
| prior_disputes        |          0.368 |         0.293 |             0.00023 |            4e-05   |                       0.94796 |          -5e-05   |
| velocity_score        |          9.054 |         2.782 |             0.02925 |            0.00197 |                       0.94798 |          -7e-05   |
| card_age_months       |          0.245 |         0.343 |            -5e-05   |           -4e-05   |                       0.94802 |          -0.00011 |

Full TE model CV AUC 0.94791; base (no TE) 0.94716; total TE gain +0.00075. Best iterations full TE: [680, 698, 515, 499, 693], base: [968, 955, 847, 698, 1069].


## 4.2 Which TE columns carry the gain

| variant                                                                    |   cv_auc |   gain_vs_base |
|:---------------------------------------------------------------------------|---------:|---------------:|
| base (no TE)                                                               |  0.94716 |        0       |
| TE on all 13 numerics                                                      |  0.94791 |        0.00075 |
| TE on high-signal only (merchant_risk, velocity, amount, hours_since_last) |  0.94742 |        0.00026 |
| TE on noise only (account_balance, distance, card_age)                     |  0.94721 |        5e-05   |

**Reading:** the gain is concentrated in **`hours_since_last_txn`** (−0.00064 when its TE is dropped) and **`customer_age`** (−0.00055), with a little from `account_balance_usd` (−0.00017) and `amount_usd` (−0.00009). All other TE columns are ≈ 0, and `card_age_months` TE is slightly harmful (+0.00011 when dropped). The two key columns are both cases where the *exact value* carries signal beyond the monotone trend:
- `hours_since_last_txn` TE has Spearman only −0.41 with the raw value (its TE alone has AUC 0.583 vs raw 0.568): popular exact values have their own fraud rates (e.g. 0.38 h → 2.8% vs 1.4% baseline).
- `customer_age` TE alone has AUC 0.645 vs raw 0.625: non-monotone per-age effects (e.g. 70 to 73 → 3.4%, then 74 to 76 → 2.4%).
TE on only high-signal columns (+0.00026) underperforms TE on all, because `customer_age` isn't in that group. Noise-only TE gives +0.00005.


## 4.3 Decimal places and most frequent values

| column                |   0_decimals_% |   1_decimals_% |   2_decimals_% |   3_decimals_% |   max_decimals |   min |      max |
|:----------------------|---------------:|---------------:|---------------:|---------------:|---------------:|------:|---------:|
| amount_usd            |              0 |           8.12 |          91.88 |              0 |              3 |  1    |  6872.69 |
| hours_since_last_txn  |              0 |           7.94 |          92.06 |              0 |              2 |  0.01 |    51.98 |
| distance_from_home_km |              0 |           8.23 |          91.77 |              0 |              2 |  0    |    57.93 |
| account_balance_usd   |              0 |          11.3  |          88.7  |              0 |              2 | 58.25 | 99701.4  |
| velocity_score        |              0 |         100    |           0    |              0 |              2 |  0    |    72.9  |
| merchant_risk_score   |              0 |         100    |           0    |              0 |              2 |  0    |   100    |


**merchant_risk_score** (overall fraud rate 0.0167)

|   merchant_risk_score |   rows |   fraud_rate |   fraud_rate_of_5+5_neighbouring_values |
|----------------------:|-------:|-------------:|----------------------------------------:|
|                  34.3 |   3498 |       0.0146 |                                  0.0142 |
|                  31.2 |   3436 |       0.009  |                                  0.0063 |
|                  36.3 |   3222 |       0.0127 |                                  0.0137 |
|                  31.6 |   3185 |       0.0057 |                                  0.0065 |
|                  32.2 |   3152 |       0.0041 |                                  0.0066 |
|                  31.4 |   3032 |       0.0046 |                                  0.0068 |
|                  31.5 |   3018 |       0.0046 |                                  0.0067 |
|                  31.8 |   2994 |       0.008  |                                  0.0055 |
|                  38   |   2988 |       0.0164 |                                  0.0164 |
|                  35.7 |   2981 |       0.0131 |                                  0.0154 |


**velocity_score** (overall fraud rate 0.0167)

|   velocity_score |   rows |   fraud_rate |   fraud_rate_of_5+5_neighbouring_values |
|-----------------:|-------:|-------------:|----------------------------------------:|
|              0   |  48499 |       0.0004 |                                  0.0035 |
|             15.7 |   3715 |       0.01   |                                  0.0111 |
|             16.3 |   3224 |       0.0155 |                                  0.0132 |
|             14.7 |   3222 |       0.0096 |                                  0.0111 |
|             16.8 |   3221 |       0.0124 |                                  0.0143 |
|             15.3 |   3209 |       0.0078 |                                  0.0109 |
|             20.8 |   3196 |       0.0088 |                                  0.0141 |
|             13.2 |   3109 |       0.0106 |                                  0.008  |
|             24.1 |   3098 |       0.0216 |                                  0.019  |
|             17.7 |   3072 |       0.0176 |                                  0.0152 |


**amount_usd** (overall fraud rate 0.0167)

|   amount_usd |   rows |   fraud_rate |   fraud_rate_of_5+5_neighbouring_values |
|-------------:|-------:|-------------:|----------------------------------------:|
|        14.68 |   1910 |       0.0147 |                                  0.009  |
|        26.52 |   1207 |       0.0174 |                                  0.0181 |
|        27.21 |   1202 |       0.0166 |                                  0.015  |
|        10.07 |   1167 |       0.0146 |                                  0.0155 |
|        17.23 |   1154 |       0.0147 |                                  0.021  |
|        38.61 |   1139 |       0.0097 |                                  0.0154 |
|         9.74 |   1044 |       0.0211 |                                  0.0121 |
|        30.78 |   1041 |       0.0067 |                                  0.0126 |
|        12.74 |   1001 |       0.015  |                                  0.0202 |
|        44.76 |    979 |       0.0245 |                                  0.0168 |


**hours_since_last_txn** (overall fraud rate 0.0167)

|   hours_since_last_txn |   rows |   fraud_rate |   fraud_rate_of_5+5_neighbouring_values |
|-----------------------:|-------:|-------------:|----------------------------------------:|
|                   0.63 |   3252 |       0.0185 |                                  0.0247 |
|                   1.07 |   2981 |       0.0178 |                                  0.0225 |
|                   2.82 |   2814 |       0.0121 |                                  0.0159 |
|                   1.78 |   2337 |       0.0193 |                                  0.0149 |
|                   1.59 |   2288 |       0.0219 |                                  0.0198 |
|                   1.58 |   2285 |       0.0188 |                                  0.0209 |
|                   3.86 |   2270 |       0.0176 |                                  0.0152 |
|                   0.38 |   2233 |       0.0278 |                                  0.0317 |
|                   3.09 |   2174 |       0.0147 |                                  0.0151 |
|                   1.44 |   2119 |       0.0198 |                                  0.0237 |


## 5.1 Empirical log-odds by bin


**merchant_risk_score**

|   bin_min |   bin_max |   count |   fraud_rate |   empirical_log_odds |
|----------:|----------:|--------:|-------------:|---------------------:|
|       0   |      16.5 |   35105 |       0.0009 |              -7.0629 |
|      16.6 |      20   |   35651 |       0.0023 |              -6.0721 |
|      20.1 |      22.6 |   34602 |       0.0023 |              -6.0669 |
|      22.7 |      24.7 |   36515 |       0.0031 |              -5.7658 |
|      24.8 |      26.4 |   33994 |       0.006  |              -5.1047 |
|      26.5 |      28.1 |   35650 |       0.0069 |              -4.965  |
|      28.2 |      29.7 |   33848 |       0.0058 |              -5.1404 |
|      29.8 |      31.2 |   35710 |       0.0071 |              -4.9346 |
|      31.3 |      32.6 |   35441 |       0.0058 |              -5.1516 |
|      32.7 |      34.2 |   34736 |       0.0113 |              -4.4754 |
|      34.3 |      35.6 |   33884 |       0.0157 |              -4.1363 |
|      35.7 |      37.2 |   35155 |       0.0133 |              -4.3034 |
|      37.3 |      39   |   36437 |       0.0173 |              -4.0385 |
|      39.1 |      40.7 |   34433 |       0.0232 |              -3.7412 |
|      40.8 |      42.7 |   33953 |       0.0178 |              -4.0128 |
|      42.8 |      45.1 |   36180 |       0.0235 |              -3.7284 |
|      45.2 |      47.9 |   33863 |       0.0302 |              -3.4699 |
|      48   |      51.6 |   35420 |       0.0332 |              -3.3722 |
|      51.7 |      57.3 |   35090 |       0.0374 |              -3.2466 |
|      57.4 |     100   |   34293 |       0.0726 |              -2.5473 |


**velocity_score**

|   bin_min |   bin_max |   count |   fraud_rate |   empirical_log_odds |
|----------:|----------:|--------:|-------------:|---------------------:|
|       0   |       4.1 |   70275 |       0.0012 |              -6.7393 |
|       4.2 |       7.3 |   35947 |       0.0055 |              -5.2009 |
|       7.4 |       9.5 |   34279 |       0.0059 |              -5.1329 |
|       9.6 |      11.4 |   34639 |       0.0059 |              -5.1187 |
|      11.5 |      13.2 |   37659 |       0.0075 |              -4.8832 |
|      13.3 |      14.5 |   32197 |       0.0099 |              -4.6012 |
|      14.6 |      15.8 |   35365 |       0.0105 |              -4.5439 |
|      15.9 |      17.3 |   36576 |       0.0139 |              -4.2646 |
|      17.4 |      18.6 |   33199 |       0.0155 |              -4.1524 |
|      18.7 |      20.1 |   37228 |       0.0141 |              -4.2452 |
|      20.2 |      21.4 |   33706 |       0.0137 |              -4.2738 |
|      21.5 |      22.9 |   36004 |       0.0165 |              -4.0878 |
|      23   |      24.4 |   35086 |       0.017  |              -4.0581 |
|      24.5 |      26   |   33193 |       0.0218 |              -3.8046 |
|      26.1 |      28   |   35270 |       0.024  |              -3.7035 |
|      28.1 |      30.3 |   35501 |       0.0218 |              -3.8037 |
|      30.4 |      33.1 |   34459 |       0.0241 |              -3.6992 |
|      33.2 |      37.4 |   34558 |       0.0422 |              -3.1232 |
|      37.5 |      72.9 |   34819 |       0.063  |              -2.6998 |


**customer_age**

|   bin_min |   bin_max |   count |   fraud_rate |   empirical_log_odds |
|----------:|----------:|--------:|-------------:|---------------------:|
|        18 |        21 |   41238 |       0.0042 |              -5.4636 |
|        22 |        25 |   29459 |       0.0039 |              -5.5504 |
|        26 |        29 |   48039 |       0.0064 |              -5.0464 |
|        30 |        33 |   24638 |       0.0054 |              -5.2237 |
|        34 |        38 |   37246 |       0.0135 |              -4.289  |
|        39 |        41 |   31126 |       0.0165 |              -4.0869 |
|        42 |        44 |   42408 |       0.0141 |              -4.2438 |
|        45 |        47 |   34805 |       0.0155 |              -4.1521 |
|        48 |        50 |   38004 |       0.0166 |              -4.0813 |
|        51 |        52 |   23686 |       0.0156 |              -4.1433 |
|        53 |        55 |   39838 |       0.0175 |              -4.0252 |
|        56 |        58 |   40577 |       0.0174 |              -4.0309 |
|        59 |        60 |   27164 |       0.0141 |              -4.25   |
|        61 |        63 |   40357 |       0.0193 |              -3.9306 |
|        64 |        66 |   31613 |       0.0203 |              -3.8762 |
|        67 |        69 |   34304 |       0.0225 |              -3.7712 |
|        70 |        73 |   35037 |       0.034  |              -3.3479 |
|        74 |        76 |   38767 |       0.024  |              -3.7036 |
|        77 |        79 |   37155 |       0.0315 |              -3.4243 |
|        80 |        81 |   24499 |       0.0216 |              -3.8116 |


**hours_since_last_txn**

|   bin_min |   bin_max |   count |   fraud_rate |   empirical_log_odds |
|----------:|----------:|--------:|-------------:|---------------------:|
|      0.01 |      0.47 |   35247 |       0.0359 |              -3.2916 |
|      0.48 |      0.85 |   35333 |       0.0237 |              -3.72   |
|      0.86 |      1.33 |   35234 |       0.0209 |              -3.8474 |
|      1.34 |      1.71 |   35361 |       0.0194 |              -3.9214 |
|      1.72 |      2.11 |   33966 |       0.0202 |              -3.8803 |
|      2.12 |      2.54 |   35399 |       0.0153 |              -4.1637 |
|      2.55 |      3.09 |   36312 |       0.0161 |              -4.112  |
|      3.1  |      3.62 |   33701 |       0.0156 |              -4.1423 |
|      3.63 |      4.13 |   34441 |       0.0159 |              -4.1283 |
|      4.14 |      4.73 |   36205 |       0.0137 |              -4.2745 |
|      4.74 |      5.45 |   33867 |       0.0137 |              -4.2743 |
|      5.46 |      6.14 |   35082 |       0.0141 |              -4.2507 |
|      6.15 |      6.99 |   34934 |       0.0127 |              -4.3525 |
|      7    |      7.94 |   35127 |       0.0141 |              -4.2438 |
|      7.95 |      9.19 |   35062 |       0.0135 |              -4.2921 |
|      9.2  |     10.47 |   35171 |       0.0142 |              -4.239  |
|     10.48 |     12.35 |   34560 |       0.0137 |              -4.2754 |
|     12.36 |     14.99 |   35101 |       0.0142 |              -4.241  |
|     15    |     19.4  |   34902 |       0.0132 |              -4.3179 |
|     19.42 |     51.98 |   34955 |       0.0137 |              -4.2762 |


**amount_usd**

|   bin_min |   bin_max |   count |   fraud_rate |   empirical_log_odds |
|----------:|----------:|--------:|-------------:|---------------------:|
|      1    |      6.86 |   35661 |       0.0179 |              -4.007  |
|      6.87 |      9.22 |   34457 |       0.0148 |              -4.2001 |
|      9.23 |     11.2  |   34968 |       0.0146 |              -4.215  |
|     11.21 |     13.61 |   34924 |       0.0155 |              -4.1518 |
|     13.62 |     15.81 |   35016 |       0.0148 |              -4.1986 |
|     15.82 |     18.12 |   35041 |       0.0168 |              -4.0671 |
|     18.13 |     20.41 |   35073 |       0.0125 |              -4.3703 |
|     20.42 |     22.54 |   34941 |       0.0148 |              -4.1964 |
|     22.56 |     24.74 |   35088 |       0.0145 |              -4.2165 |
|     24.75 |     27.25 |   35004 |       0.0146 |              -4.2141 |
|     27.26 |     30.11 |   35251 |       0.0129 |              -4.3391 |
|     30.12 |     32.48 |   34726 |       0.011  |              -4.496  |
|     32.5  |     35.64 |   34959 |       0.0138 |              -4.2721 |
|     35.65 |     39.06 |   34908 |       0.0145 |              -4.2193 |
|     39.08 |     43.11 |   34985 |       0.016  |              -4.1167 |
|     43.12 |     46.85 |   35028 |       0.017  |              -4.0547 |
|     46.86 |     52.06 |   35046 |       0.021  |              -3.8405 |
|     52.07 |     89.8  |   34890 |       0.0185 |              -3.9704 |
|     89.84 |    542.46 |   34999 |       0.0345 |              -3.3321 |
|    542.52 |   6872.69 |   34995 |       0.0239 |              -3.7089 |


**cvv_retry_count**

|   bin_min |   bin_max |   count |   fraud_rate |   empirical_log_odds |
|----------:|----------:|--------:|-------------:|---------------------:|
|         0 |         0 |  642088 |       0.0093 |              -4.6652 |
|         1 |         1 |   55942 |       0.0875 |              -2.3445 |
|         2 |         2 |    1905 |       0.4173 |              -0.3338 |
|         3 |         3 |      25 |       0.36   |              -0.5754 |


**txn_count_last_24h**

|   bin_min |   bin_max |   count |   fraud_rate |   empirical_log_odds |
|----------:|----------:|--------:|-------------:|---------------------:|
|         0 |         0 |   18462 |       0.0023 |              -6.0595 |
|         1 |         1 |   92705 |       0.0069 |              -4.9686 |
|         2 |         2 |  170757 |       0.0128 |              -4.3461 |
|         3 |         3 |  178367 |       0.0166 |              -4.0788 |
|         4 |         4 |  131869 |       0.0198 |              -3.9028 |
|         5 |         5 |   69701 |       0.0259 |              -3.6274 |
|         6 |         6 |   26488 |       0.0351 |              -3.3135 |
|         7 |         7 |    8907 |       0.0436 |              -3.089  |
|         8 |         8 |    2252 |       0.0395 |              -3.1906 |
|         9 |         9 |     378 |       0.0582 |              -2.7839 |
|        10 |        10 |      65 |       0.1231 |              -1.9636 |
|        11 |        11 |       9 |       0.2222 |              -1.2528 |


**prior_disputes**

|   bin_min |   bin_max |   count |   fraud_rate |   empirical_log_odds |
|----------:|----------:|--------:|-------------:|---------------------:|
|         0 |         0 |  598396 |       0.0155 |              -4.1531 |
|         1 |         1 |   96365 |       0.0236 |              -3.7227 |
|         2 |         2 |    5032 |       0.03   |              -3.4758 |
|         3 |         3 |     165 |       0.0364 |              -3.2771 |
|         4 |         4 |       2 |       0      |             -13.8155 |


## 5.2 What explains the LogReg -> GBDT gap (residual analysis)

Decision tree (depth 4, min leaf 2000) fitted to residual y - oof_v1. Leaf value = mean residual (positive = LogReg under-predicts fraud there).

```
|--- merchant_risk_score <= 80.750
|   |--- merchant_risk_score <= 61.950
|   |   |--- merchant_risk_score <= 59.950
|   |   |   |--- hours_since_last_txn <= 0.635
|   |   |   |   |--- value: [-0.005]
|   |   |   |--- hours_since_last_txn >  0.635
|   |   |   |   |--- value: [0.000]
|   |   |--- merchant_risk_score >  59.950
|   |   |   |--- velocity_score <= 22.450
|   |   |   |   |--- value: [-0.018]
|   |   |   |--- velocity_score >  22.450
|   |   |   |   |--- value: [-0.050]
|   |--- merchant_risk_score >  61.950
|   |   |--- velocity_score <= 13.050
|   |   |   |--- hours_since_last_txn <= 7.585
|   |   |   |   |--- value: [-0.003]
|   |   |   |--- hours_since_last_txn >  7.585
|   |   |   |   |--- value: [0.011]
|   |   |--- velocity_score >  13.050
|   |   |   |--- auth_method_PIN <= 0.500
|   |   |   |   |--- value: [0.029]
|   |   |   |--- auth_method_PIN >  0.500
|   |   |   |   |--- value: [0.005]
|--- merchant_risk_score >  80.750
|   |--- value: [-0.072]
```


Top residual-tree features: merchant_risk_score (0.81), velocity_score (0.09), hours_since_last_txn (0.06), auth_method_PIN (0.04)


Decision tree (depth 3) fitted to logit(oof_v4) - logit(oof_v1): where the GBDT disagrees with LogReg.

```
|--- velocity_score <= 0.050
|   |--- auth_method_Biometric <= 0.500
|   |   |--- auth_method_PIN <= 0.500
|   |   |   |--- value: [-1.620]
|   |   |--- auth_method_PIN >  0.500
|   |   |   |--- value: [-1.342]
|   |--- auth_method_Biometric >  0.500
|   |   |--- value: [-1.084]
|--- velocity_score >  0.050
|   |--- customer_age <= 20.500
|   |   |--- velocity_score <= 19.350
|   |   |   |--- value: [-0.974]
|   |   |--- velocity_score >  19.350
|   |   |   |--- value: [-1.263]
|   |--- customer_age >  20.500
|   |   |--- merchant_risk_score <= 24.250
|   |   |   |--- value: [-0.443]
|   |   |--- merchant_risk_score >  24.250
|   |   |   |--- value: [0.027]
```


## 5.3 id / row-order check

Single-feature AUC of id: 0.49855


|   id_decile |   rows |   fraud_rate |
|------------:|-------:|-------------:|
|           0 |  69996 |      0.0167  |
|           1 |  69996 |      0.01587 |
|           2 |  69996 |      0.01712 |
|           3 |  69996 |      0.01734 |
|           4 |  69996 |      0.01713 |
|           5 |  69996 |      0.01686 |
|           6 |  69996 |      0.01673 |
|           7 |  69996 |      0.01626 |
|           8 |  69996 |      0.01696 |
|           9 |  69996 |      0.01603 |
