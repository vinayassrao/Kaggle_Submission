"""Builds RUNBOOK_REPORT.md from artifacts already on disk (OOF arrays, logs, results.csv).
Only two light computations: v9 fold-1 feature importance (one LightGBM fit + permutation) and
v10 fold-1 n_iter_ (one logistic-regression fit)."""
import json
import os
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import HERE, OOF_DIR, SEED, load, raw_features
from fe_v9 import deterministic_features, fold_woe_features, to_std_logit

warnings.filterwarnings("ignore")
train, test, y = load()
folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
res = pd.read_csv(os.path.join(HERE, "results.csv")).set_index("iteration")
diag = json.load(open(os.path.join(HERE, "diag_te_results.json")))

TREES = {  # parsed from run logs (best iteration per fold)
    "v4_lightgbm_tuned": [2280, 2212, 2547, 1674, 2355], "v5_xgboost": [2254, 2185, 1592, 1987, 2746],
    "v6_catboost": [2102, 1551, 1935, 1254, 2070], "screen_base": diag["base"]["iters"], "screen_te_nums": diag["full"]["iters"],
    "v9_lightgbm_woe": [1908, 1834, 1838, 1277, 1696], "v12_xgboost_woe": [2001, 1799, 1938, 1647, 2090],
    "v13_catboost_woe": [4308, 3377, 4151, 3705, 3461], "v14_lightgbm_extratrees": [2349, 2096, 2071, 1456, 1577],
}
SECS = {"screen_base": None, "screen_te_nums": None}
saved = ["v1_logistic_regression", "v4_lightgbm_tuned", "v5_xgboost", "v6_catboost", "v7_ensemble", "screen_base", "screen_te_nums",
         "v9_lightgbm_woe", "v10_spline_logreg", "v11_ebm", "v11b_ebm_big", "v12_xgboost_woe", "v13_catboost_woe",
         "v14_lightgbm_extratrees", "v15_stack"]
O = {m: np.load(os.path.join(OOF_DIR, f"{m}_oof.npy")) for m in saved}
fold_auc = lambda o: np.array([roc_auc_score(y[va], o[va]) for _, va in folds])
FA = {m: fold_auc(o) for m, o in O.items()}

# screen variants (OOF not saved): fold AUCs + trees from logs
SCREEN = {
    "s9_maxbin (max_bin 1023 only)": ([0.94598, 0.94934, 0.95193, 0.94759, 0.9428], 0.94751, [1045, 1089, 905, 1022, 796], 108),
    "s9_det (+logfreq, is_velocity_zero, max_bin)": ([0.94596, 0.9487, 0.95179, 0.94784, 0.9424], 0.94727, [1103, 821, 831, 863, 998], 94),
    "s9_woe (m=15, resid, memo, det, max_bin)": ([0.94718, 0.94981, 0.95322, 0.94887, 0.94308], 0.94839, [576, 519, 689, 513, 685], 98),
    "s9_woe_noresid (m=15, plain WoE for risk/velocity)": ([0.94741, 0.94932, 0.95315, 0.94898, 0.94282], 0.94830, [615, 646, 585, 545, 552], 80),
    "s9_woe_m40 (m=40)": ([0.94722, 0.94966, 0.95317, 0.94903, 0.94324], 0.94842, [591, 620, 620, 483, 682], 89),
    "s9_woe_m40_nodet (m=40, no logfreq/velocity-zero)": ([0.94699, 0.94972, 0.95341, 0.94895, 0.94302], 0.94832, None, 112),
    "s9_woe_dropnoise (m=15, drop 6 noise cols)": ([0.94737, 0.94957, 0.95286, 0.94862, 0.94294], 0.94820, [838, 695, 555, 471, 643], 91),
    "s9_woe_pairs (m=15, + 3 pair WoEs)": ([0.94711, 0.94942, 0.95333, 0.94867, 0.94323], 0.94829, [896, 759, 622, 557, 698], 106),
}
md = []
w = md.append
w("# Runbook implementation report: Kaggle Playground S6E11\n")
w("Everything below was run on the same folds, `StratifiedKFold(5, shuffle=True, random_state=42)`, and summarised from artifacts on disk. "
  "Fold numbering is 0 to 4. \"Concat\" = AUC of the concatenated raw OOF probabilities. \"Std-logit\" = AUC of the OOF after within-fold standardised log-odds "
  "(= within-fold rank AUC to 5 decimals here). PR-AUC = average precision on the concatenated OOF.\n")
w("**Status:** Experiments 1, 2, 3, 4, 5 and 8 are done. **Experiment 6 (MLP) not run** (PyTorch isn't installed). **Experiment 7 / Section 2 not run** (the original "
  "`credit_card_fraud_2026.csv` isn't on disk yet). **No public LB submissions yet.**\n")
w("**Headline:** best single model **v11 EBM 0.94927**. Best blend **v15 LogReg stack 0.94941** (concat; std-logit 0.94945). Starting point v4 0.94737 / v7 0.94744.\n")

# 1. master table
w("\n## 1. Master score table\n")
rows = []
v4f, tef = FA["v4_lightgbm_tuned"], FA["screen_te_nums"]
label = {"v1_logistic_regression": "v1 LogReg (baseline)", "v4_lightgbm_tuned": "v4 LightGBM (reference)", "v5_xgboost": "v5 XGBoost",
         "v6_catboost": "v6 CatBoost", "v7_ensemble": "v7 rank blend (old best)", "screen_base": "screen base (fast LGB lr 0.05)",
         "screen_te_nums": "screen_te_nums (sklearn TE on 13 numerics, fast LGB)", "v9_lightgbm_woe": "**v9 = v4_plus** (LGB lr 0.02 + WoE m40 + det + max_bin 1023)",
         "v10_spline_logreg": "**v10 spline LogReg** (Exp 3)", "v11_ebm": "**v11 EBM** (Exp 4, 4 bags, 10 interactions)",
         "v11b_ebm_big": "v11b EBM (8 bags, 20 interactions)", "v12_xgboost_woe": "**v12 = xgb_v5_plus** (Exp 5)",
         "v13_catboost_woe": "**v13 = cat_v6_plus** (Exp 5)", "v14_lightgbm_extratrees": "**v14 = lgb_extratrees** (Exp 5)",
         "v15_stack": "**v15 OOF LogReg stack** (Exp 8.2)"}
for m in saved:
    o = O[m]; z = np.zeros_like(o)
    for _, va in folds:   # v15 OOF is already a log-odds decision function: standardise it directly
        z[va] = (o[va] - o[va].mean()) / o[va].std() if m == "v15_stack" else to_std_logit(o[va])
    f = FA[m]
    secs = res.loc[m, "seconds"] if m in res.index else None
    rows.append({"model": label[m], "concat_auc": roc_auc_score(y, o), "std_logit_auc": roc_auc_score(y, z),
                 **{f"fold{i}": f[i] for i in range(5)}, "beats_v4": f"{int((f > v4f).sum())}/5",
                 "beats_te_nums": f"{int((f > tef).sum())}/5", "pr_auc": average_precision_score(y, o),
                 "best_iters": " ".join(map(str, TREES[m])) if m in TREES else ("n/a" if m in ("v7_ensemble", "v15_stack") else "see 3"),
                 "runtime_s": int(secs) if secs is not None and not pd.isna(secs) and secs > 0 else ""})
for k, (f, auc, it, secs) in SCREEN.items():
    f = np.array(f)
    rows.append({"model": k, "concat_auc": auc, "std_logit_auc": np.nan, **{f"fold{i}": f[i] for i in range(5)},
                 "beats_v4": f"{int((f > v4f).sum())}/5", "beats_te_nums": f"{int((f > tef).sum())}/5", "pr_auc": np.nan,
                 "best_iters": " ".join(map(str, it)) if it else "not logged", "runtime_s": secs})
T = pd.DataFrame(rows)
w(T.round(5).to_markdown(index=False) + "\n")
w("\nNotes: the screen variants used the fast LightGBM (lr 0.05, patience 200) and didn't save OOF arrays, so std-logit and PR-AUC are blank for them. "
  "Their fold AUCs/iterations come from the run log. Runtime of v11/v11b = EBM wall clock for all 5 folds. "
  "v7/v15 are blends (no iterations). The v15 concat AUC is computed on the meta-model's nested OOF decision function.\n")

# 2. feature details + importance
w("\n## 2. Feature engineering & ablation\n")
w("""**Exact WoE specification (`fe_v9.fold_woe_features`)**
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
""")
# importance on fold 1 for v9
from v9_lightgbm_woe import PARAMS as V9P
X0, Xt0 = raw_features(train, test)
D, Dt = deterministic_features(train, test)
X = pd.concat([X0, D], axis=1)
tr, va = folds[0]
f_tr, (f_va,) = fold_woe_features(train.iloc[tr].reset_index(drop=True), y[tr], [train.iloc[va].reset_index(drop=True)], m=40.0)
Xtr = pd.concat([X.iloc[tr].reset_index(drop=True), f_tr], axis=1); Xva = pd.concat([X.iloc[va].reset_index(drop=True), f_va], axis=1)
mdl = lgb.LGBMClassifier(**{**V9P, "n_estimators": TREES["v9_lightgbm_woe"][0]}).fit(Xtr, y[tr])
g = pd.Series(mdl.booster_.feature_importance("gain"), index=Xtr.columns); g = g / g.sum() * 100
pi = permutation_importance(mdl, Xva, y[va], scoring="roc_auc", n_repeats=2, random_state=SEED, n_jobs=1)
imp = pd.DataFrame({"feature": Xtr.columns, "gain_pct": g.values, "perm_auc_drop": pi.importances_mean}).sort_values("gain_pct", ascending=False)
w(f"\n**Top 20 features of v9 (fold-0 model, {TREES['v9_lightgbm_woe'][0]} trees; fold-0 validation AUC {roc_auc_score(y[va], mdl.predict_proba(Xva)[:, 1]):.5f})**\n\n")
w(imp.head(20).round(5).to_markdown(index=False) + "\n")
w(f"\nNew-feature permutation drops (all): " + ", ".join(f"{r.feature} {r.perm_auc_drop:+.5f}" for r in imp.itertuples()
   if ("woe" in r.feature or "logfreq" in r.feature or r.feature == "is_velocity_zero")) + "\n")
w("\n**Noise-column drop** (`channel, device_type, card_type, time_of_day_hour, day_of_week, card_age_months`; on the m=15 WoE screen): CV AUC **0.94839 → 0.94820 (−0.00019)**, "
  "folds beat 1/5. Mean best trees 596 → 640. Not adopted.\n")

# 3. diverse models
from v10_spline_logreg import X as SX, fit_predict as _unused  # noqa: F401
import v10_spline_logreg as V10
f_tr, (f_va,) = fold_woe_features(train.iloc[tr].reset_index(drop=True), y[tr], [train.iloc[va].reset_index(drop=True)], m=40.0)
import sklearn.linear_model as sklm
captured = {}
_orig = sklm.LogisticRegression.fit
def _cap(self, *a, **k):
    r = _orig(self, *a, **k); captured["n_iter"] = int(self.n_iter_[0]); captured["n_feat"] = self.coef_.shape[1]; return r
sklm.LogisticRegression.fit = _cap
V10.LogisticRegression.fit = _cap
V10.fit_predict(pd.concat([SX.iloc[tr].reset_index(drop=True), f_tr], axis=1), y[tr],
                pd.concat([SX.iloc[va].reset_index(drop=True), f_va], axis=1), y[va], pd.concat([SX.iloc[va].reset_index(drop=True), f_va], axis=1))
w("\n## 3. Non-tree & diverse models\n")
w(f"""**v10 spline LogReg (Exp 3).** `ColumnTransformer`:
- `SplineTransformer(n_knots=12, degree=3, knots="quantile", include_bias=False)` on `merchant_risk_score, velocity_score, log1p(hours), log1p(amount), log1p(balance), distance_from_home_km`;
- `OneHotEncoder(handle_unknown="ignore", min_frequency=10)` on exact `customer_age, cvv_retry_count, txn_count_last_24h, prior_disputes` + the 5 categoricals;
- `StandardScaler` on the 6 flags, `flag_count`, `is_velocity_zero`, 5 `*_logfreq`, the 7 WoE columns + `woe_memo_sum`;
- plus 20 standardised interactions: `auth_method` dummies × `[is_velocity_zero, merchant_risk_score/100, cvv_retry_count, flag_count]`.

`LogisticRegression(C=0.2, solver="lbfgs", max_iter=3000)` on the sparse matrix ({captured.get('n_feat')} columns). Fold-0 refit converged in **n_iter_ = {captured.get('n_iter')}** (< 3000, no warning).
CV AUC **0.94721** vs v1 0.94187 (+0.00534), beats v4 in {int((FA['v10_spline_logreg'] > v4f).sum())}/5 folds. 99 s for all folds.

**Hyperparameters of the other Exp 4/5 models** (all with raw features + det features + in-fold WoE m=40):

| Model | Settings | CV AUC | Notes |
|---|---|---|---|
| v11 EBM | `max_bins=1024, max_interaction_bins=64, interactions=10, outer_bags=4, inner_bags=0, learning_rate=0.015, min_samples_leaf=100, early_stopping_rounds=100, n_jobs=-1` | 0.94927 | 724 s. Categoricals passed as strings. Harmless joblib/loky resource-tracker `KeyError` (temp memmap cleanup) printed on Windows; run unaffected |
| v11b EBM | same, `interactions=20, outer_bags=8` | 0.94919 | 1,297 s. No gain |
| v12 XGBoost | `hist, max_bin=1024, lr 0.02, max_depth 4, min_child_weight 10, subsample 0.7, colsample_bytree 0.4, reg_lambda 10, enable_categorical, max_cat_to_onehot=16, early_stopping 400` | 0.94847 | 444 s |
| v13 CatBoost | `depth 4, lr 0.03, l2_leaf_reg 15, rsm 0.4, Bernoulli subsample 0.75, border_count 254, max_ctr_complexity 1, od_wait 300`; cat_features = 5 cats + `hours_since_last_txn_str`, `customer_age_str` | 0.94780 | 1,977 s (slowest) |
| v14 LGB extra_trees | `extra_trees=True, max_depth 5, num_leaves 24, colsample 0.5, max_bin 1023, lr 0.02` (other v4 params) | 0.94758 | 238 s |
| MLP (Exp 6) | not run | | PyTorch not installed on Python 3.14 env |
""")
w("\n**Diversity matrix (Spearman of OOF predictions)**\n\n")
cm = ["v1_logistic_regression", "v4_lightgbm_tuned", "v9_lightgbm_woe", "v10_spline_logreg", "v11_ebm", "v11b_ebm_big",
      "v12_xgboost_woe", "v13_catboost_woe", "v14_lightgbm_extratrees"]
C = spearmanr(np.column_stack([O[m] for m in cm])).correlation
w(pd.DataFrame(C, index=cm, columns=[c.split("_")[0] for c in cm]).round(3).to_markdown() + "\n")

# 4. stacking
w("""
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
""")

# 5. artifacts
w("\n## 5. Saved artifacts & implementation differences\n")
files = sorted(f for f in os.listdir(OOF_DIR) if f.endswith(".npy"))
w("**OOF / test arrays in `oof/`:** " + ", ".join(f"`{f}`" for f in files) + "\n")
w("""
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
""")
open(os.path.join(HERE, "RUNBOOK_REPORT.md"), "w", encoding="utf-8").write("\n".join(md))
print("written")
