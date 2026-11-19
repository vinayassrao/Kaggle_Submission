"""Fast diagnostics requested by the reviewer (sections 1.3, 3, 4.1 coverage, 4.3, 5). Writes diag_report_fast.md."""
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.tree import DecisionTreeRegressor, export_text

from common import CATS, FLAGS, HERE, NUMS, OOF_DIR, SEED, load

train, test, y = load()
folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
md = []
w = md.append


def table(df, floatfmt=5):
    return df.to_markdown(index=False, floatfmt=f".{floatfmt}g") if hasattr(df, "to_markdown") else df.to_string()


def oof(name):
    p = os.path.join(OOF_DIR, f"{name}_oof.npy")
    return np.load(p) if os.path.exists(p) else None


# ---------------- 1.3 OOF aggregation check ----------------
w("## 1.3 Concatenated vs per-fold OOF AUC\n")
rows = []
for name in ["v1_logistic_regression", "v4_lightgbm_tuned", "v5_xgboost", "v6_catboost", "v7_ensemble", "screen_base", "screen_te_nums"]:
    o = oof(name)
    if o is None:
        continue
    fa = [roc_auc_score(y[va], o[va]) for _, va in folds]
    rk = np.zeros(len(o)); lz = np.zeros(len(o))
    for _, va in folds:
        rk[va] = rankdata(o[va]) / len(va)
        z = logit(np.clip(o[va], 1e-7, 1 - 1e-7)); lz[va] = (z - z.mean()) / z.std()
    rows.append(dict(model=name, mean_fold_auc=np.mean(fa), concat_raw=roc_auc_score(y, o),
                     concat_within_fold_rank=roc_auc_score(y, rk), concat_logodds_std=roc_auc_score(y, lz)))
w(pd.DataFrame(rows).round(5).to_markdown(index=False) + "\n")

# ---------------- 3.1 calibration ----------------
p = oof("v4_lightgbm_tuned")
w("\n## 3.1 Calibration / reliability of oof_v4 (20 equal-frequency bins)\n")
q = pd.qcut(rankdata(p, method="ordinal"), 20, labels=False)
cal = pd.DataFrame({"bin": q, "p": p, "y": y}).groupby("bin").agg(count=("y", "size"), fraud_count=("y", "sum"),
                                                                     mean_pred=("p", "mean"), actual_rate=("y", "mean"),
                                                                     p_min=("p", "min"), p_max=("p", "max"))
cal["share_of_frauds"] = cal.fraud_count / y.sum()
w(cal.reset_index().round(5).to_markdown(index=False) + "\n")
w("\nTop tails:\n")
tails = []
for t in [0.05, 0.02, 0.01, 0.005, 0.001]:
    m = p >= np.quantile(p, 1 - t)
    tails.append(dict(bucket=f"top {t:.1%}", count=int(m.sum()), fraud_count=int(y[m].sum()), mean_pred=p[m].mean(),
                      actual_rate=y[m].mean(), share_of_frauds=y[m].sum() / y.sum()))
w(pd.DataFrame(tails).round(5).to_markdown(index=False) + "\n")
w(f"\nOverall mean prediction {p.mean():.5f} vs fraud rate {y.mean():.5f}. Brier {np.mean((p - y) ** 2):.5f}.\n")

# ---------------- 3.2 low-score frauds ----------------
w("\n## 3.2 Low-score frauds\n")
r = rankdata(p) / len(p)
for cut in [0.5, 0.8]:
    w(f"- Frauds in bottom {cut:.0%} of oof_v4: **{(y[r <= cut]).sum():,} = {(y[r <= cut]).sum() / y.sum():.2%}** of all frauds "
      f"(fraud rate inside that bottom {cut:.0%}: {y[r <= cut].mean():.3%})\n")
low = r <= 0.5
fn, tn = train[low & (y == 1)], train[low & (y == 0)]
comp = []
for c in NUMS + FLAGS:
    comp.append(dict(feature=c, FN_bottom50_mean=fn[c].mean(), TN_bottom50_mean=tn[c].mean(), all_train_mean=train[c].mean()))
for c in CATS:
    for v in sorted(train[c].unique()):
        comp.append(dict(feature=f"{c}={v}", FN_bottom50_mean=(fn[c] == v).mean(), TN_bottom50_mean=(tn[c] == v).mean(),
                         all_train_mean=(train[c] == v).mean()))
comp = pd.DataFrame(comp)
comp["FN_minus_TN"] = comp.FN_bottom50_mean - comp.TN_bottom50_mean
w(f"\nFalse negatives in bottom 50%: n={len(fn):,}; true negatives in bottom 50%: n={len(tn):,}.\n\n")
w(comp.round(4).to_markdown(index=False) + "\n")
show = NUMS + FLAGS + CATS
lo = train.assign(p=p)[y == 1].nsmallest(15, "p")
hi = train.assign(p=p)[y == 0].nlargest(15, "p")
w("\n**15 lowest-scored actual frauds**\n\n" + lo[["id", "p"] + show].round(4).to_markdown(index=False) + "\n")
w("\n**15 highest-scored non-frauds**\n\n" + hi[["id", "p"] + show].round(4).to_markdown(index=False) + "\n")

# ---------------- 3.3 simulated Bayes AUC ----------------
w("\n## 3.3 Simulated AUC if oof_v4 were the true probabilities\n")
rng = np.random.default_rng(0)
sims = [roc_auc_score(rng.binomial(1, p), p) for _ in range(10)]
w(f"y_sim ~ Bernoulli(oof_v4), 10 draws: mean AUC **{np.mean(sims):.5f}** (sd {np.std(sims):.5f}); observed AUC {roc_auc_score(y, p):.5f}.\n")
p7 = oof("v7_ensemble")

# ---------------- 4.1 coverage + TE correlation ----------------
w("\n## 4.1 Exact-value coverage of numeric columns (+ TE correlation)\n")
te = TargetEncoder(target_type="binary", cv=5, shuffle=True, random_state=SEED)
te_vals = te.fit_transform(train[NUMS].astype(str), y)
cov = []
for i, c in enumerate(NUMS):
    vc = train[c].value_counts()
    cov.append(dict(column=c, n_unique_train=train[c].nunique(), n_unique_test=test[c].nunique(),
                    pct_test_value_in_train=test[c].isin(vc.index).mean() * 100,
                    pct_test_value_seen_ge10=test[c].map(vc).fillna(0).ge(10).mean() * 100,
                    spearman_col_vs_te=pd.Series(train[c]).corr(pd.Series(te_vals[:, i]), method="spearman"),
                    auc_of_te_alone=max(roc_auc_score(y, te_vals[:, i]), 1 - roc_auc_score(y, te_vals[:, i])),
                    auc_of_raw_alone=max(roc_auc_score(y, train[c]), 1 - roc_auc_score(y, train[c]))))
w(pd.DataFrame(cov).round(4).to_markdown(index=False) + "\n")
w("\n(TE here = cross-fitted 5-fold TargetEncoder on all of train, only for these coverage/correlation stats.)\n")

# ---------------- 4.3 decimals / frequent values ----------------
w("\n## 4.3 Decimal places and most frequent values\n")
dec = []
for c in ["amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd", "velocity_score", "merchant_risk_score"]:
    s = pd.concat([train[c], test[c]]).astype(str)
    nd = s.str.split(".").str[1].fillna("").str.len()
    dec.append(dict(column=c, **{f"{k}_decimals_%": round((nd == k).mean() * 100, 2) for k in range(0, 4)},
                    max_decimals=int(nd.max()), min=float(train[c].min()), max=float(train[c].max())))
w(pd.DataFrame(dec).to_markdown(index=False) + "\n")
for c in ["merchant_risk_score", "velocity_score", "amount_usd", "hours_since_last_txn"]:
    vc = train.groupby(c)["is_fraud"].agg(["size", "mean"]).sort_values("size", ascending=False)
    top = vc.head(10).reset_index()
    nb = []
    sv = np.sort(train[c].unique())
    for v in top[c]:
        i = np.searchsorted(sv, v)
        neigh = sv[max(0, i - 5):i].tolist() + sv[i + 1:i + 6].tolist()
        mm = train[c].isin(neigh)
        nb.append(train.loc[mm, "is_fraud"].mean())
    top["neighbour_values_fraud_rate"] = nb
    top.columns = [c, "rows", "fraud_rate", "fraud_rate_of_5+5_neighbouring_values"]
    w(f"\n**{c}** (overall fraud rate {y.mean():.4f})\n\n" + top.round(4).to_markdown(index=False) + "\n")

# ---------------- 5.1 log-odds shapes ----------------
w("\n## 5.1 Empirical log-odds by bin\n")
for c in ["merchant_risk_score", "velocity_score", "customer_age", "hours_since_last_txn", "amount_usd",
          "cvv_retry_count", "txn_count_last_24h", "prior_disputes"]:
    if train[c].nunique() <= 15:
        g = train.groupby(c)["is_fraud"].agg(["min", "size", "mean"]).reset_index()
        g = pd.DataFrame({"bin_min": g[c], "bin_max": g[c], "count": g["size"], "fraud_rate": g["mean"]})
    else:
        b = pd.qcut(train[c], 20, duplicates="drop")
        g = train.groupby(b, observed=True).agg(bin_min=(c, "min"), bin_max=(c, "max"), count=(c, "size"), fraud_rate=("is_fraud", "mean")).reset_index(drop=True)
    g["empirical_log_odds"] = np.log((g.fraud_rate + 1e-6) / (1 - g.fraud_rate + 1e-6))
    w(f"\n**{c}**\n\n" + g.round(4).to_markdown(index=False) + "\n")

# ---------------- 5.2 residual tree on LogReg ----------------
w("\n## 5.2 What explains the LogReg -> GBDT gap (residual analysis)\n")
p1, p4 = oof("v1_logistic_regression"), p
Xd = train[NUMS + FLAGS].copy()
for c in CATS:
    Xd = pd.concat([Xd, pd.get_dummies(train[c], prefix=c, dtype=int)], axis=1)
resid = y - p1
tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=2000, random_state=0).fit(Xd, resid)
w("Decision tree (depth 4, min leaf 2000) fitted to residual y - oof_v1. Leaf value = mean residual (positive = LogReg under-predicts fraud there).\n\n```\n"
  + export_text(tree, feature_names=list(Xd.columns), decimals=3, show_weights=False) + "```\n")
imp = pd.Series(tree.feature_importances_, index=Xd.columns).sort_values(ascending=False).head(10)
w("\nTop residual-tree features: " + ", ".join(f"{k} ({v:.2f})" for k, v in imp.items() if v > 0) + "\n")
# where does v4 gain over v1? difference in logit
d = logit(np.clip(p4, 1e-7, 1 - 1e-7)) - logit(np.clip(p1, 1e-7, 1 - 1e-7))
tree2 = DecisionTreeRegressor(max_depth=3, min_samples_leaf=5000, random_state=0).fit(Xd, d)
w("\nDecision tree (depth 3) fitted to logit(oof_v4) - logit(oof_v1): where the GBDT disagrees with LogReg.\n\n```\n"
  + export_text(tree2, feature_names=list(Xd.columns), decimals=3) + "```\n")

# ---------------- 5.3 id order ----------------
w("\n## 5.3 id / row-order check\n")
w(f"Single-feature AUC of id: {roc_auc_score(y, train['id']):.5f}\n\n")
dec_ = pd.qcut(train["id"], 10, labels=False)
w(train.groupby(dec_)["is_fraud"].agg(["size", "mean"]).rename(columns={"size": "rows", "mean": "fraud_rate"}).reset_index().rename(columns={"id": "id_decile"}).round(5).to_markdown(index=False) + "\n")

with open(os.path.join(HERE, "diag_report_fast.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print("written", len(md), "blocks")
