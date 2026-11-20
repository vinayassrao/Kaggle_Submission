"""Answers the questions from the original dataset's description, using our best model's OOF predictions.

1. Which signals matter most? (LightGBM gain importance + permutation importance)
2. Precision / recall / PR-AUC and threshold tuning: how much recall per false alarm?
3. Are AI-generated scam attempts harder to catch than ordinary fraud?
4. Does strong authentication actually suppress fraud?
Writes analysis_results.txt and the charts used in the guide (guide/images/13_* to 17_*).
"""
import os

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split

from common import HERE, OOF_DIR, SEED, load, raw_features
from v4_lightgbm_tuned import PARAMS

IMG = os.path.join(HERE, "guide", "build", "images")
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
    "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlecolor": INK,
})
lines = []


def say(s=""):
    print(s)
    lines.append(s)


def save(fig, name):
    fig.savefig(os.path.join(IMG, name), dpi=160, bbox_inches="tight")
    plt.close(fig)


train, test, y = load()
X, _ = raw_features(train, test)
best = np.load(os.path.join(OOF_DIR, "v7_ensemble_oof.npy"))
v4 = np.load(os.path.join(OOF_DIR, "v4_lightgbm_tuned_oof.npy"))
pretty = lambda c: c.replace("_", " ")

# 1. Feature importance ---------------------------------------------------------------
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
model = lgb.LGBMClassifier(**{**PARAMS, "n_estimators": 2200})
model.fit(X_tr, y_tr)
gain = pd.Series(model.booster_.feature_importance("gain"), index=X.columns)
gain = gain / gain.sum() * 100
perm = permutation_importance(model, X_va, y_va, scoring="roc_auc", n_repeats=3, random_state=SEED, n_jobs=1)
perm = pd.Series(perm.importances_mean, index=X.columns)
say("== 1. Feature importance ==")
say(f"Validation AUC of the importance model: {roc_auc_score(y_va, model.predict_proba(X_va)[:, 1]):.5f}")
imp = pd.DataFrame({"gain_%": gain.round(2), "permutation_auc_drop": perm.round(5)}).sort_values("permutation_auc_drop", ascending=False)
say(imp.to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
for ax, s, ttl, xl in [(axes[0], gain, "Gain: how much each feature's splits improved the model", "% of total gain"),
                       (axes[1], perm, "Permutation: AUC lost when the column is shuffled", "drop in AUC")]:
    s = s.sort_values()
    ax.barh([pretty(c) for c in s.index], s.values, color=BLUE, height=0.65)
    ax.set_title(ttl, loc="left", fontsize=11.5); ax.set_xlabel(xl); ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=9)
fig.tight_layout()
save(fig, "13_feature_importance.png")

# 2. Precision / recall / threshold ------------------------------------------------------
say("\n== 2. Precision, recall, PR-AUC, thresholds (v7 ensemble OOF) ==")
ap = average_precision_score(y, best)
say(f"ROC AUC {roc_auc_score(y, best):.5f}   PR-AUC (average precision) {ap:.5f}   (a random model's PR-AUC = fraud rate = {y.mean():.4f})")
prec, rec, thr = precision_recall_curve(y, best)
n_fraud, n_norm = y.sum(), (y == 0).sum()
say(f"{'flag top':>9} {'recall':>7} {'precision':>9} {'frauds caught':>14} {'false alarms':>13} {'alarms per catch':>17}")
table = []
for q in [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]:
    t = np.quantile(best, 1 - q)
    flagged = best >= t
    tp = int((flagged & (y == 1)).sum()); fp = int((flagged & (y == 0)).sum())
    table.append((q, tp / n_fraud, tp / max(tp + fp, 1), tp, fp, fp / max(tp, 1)))
    say(f"{q:>8.1%} {tp / n_fraud:>7.1%} {tp / (tp + fp):>9.1%} {tp:>14,} {fp:>13,} {fp / tp:>17.1f}")
f1 = 2 * prec * rec / (prec + rec + 1e-12)
bi = int(np.argmax(f1[:-1]))
say(f"Best-F1 threshold: score >= {thr[bi]:.4f} -> precision {prec[bi]:.1%}, recall {rec[bi]:.1%}, F1 {f1[bi]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
ax = axes[0]
ax.plot(rec, prec, color=BLUE, lw=2, label=f"v7 ensemble (PR-AUC {ap:.3f})")
ax.axhline(y.mean(), color=AXIS, ls="--", lw=1.2, label=f"random guessing ({y.mean():.3f})")
ax.scatter([rec[bi]], [prec[bi]], color=ORANGE, s=70, edgecolor=SURFACE, linewidth=2, zorder=3)
ax.text(rec[bi] + 0.02, prec[bi] + 0.02, f"best F1: precision {prec[bi]:.0%}, recall {rec[bi]:.0%}", fontsize=9.5)
ax.set_xlabel("Recall: % of all frauds caught"); ax.set_ylabel("Precision: % of alarms that are real fraud")
ax.set_title("Precision-recall curve", loc="left"); ax.legend(frameon=False, loc="upper right", fontsize=9.5)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax = axes[1]
qs = [f"top {q:.1%}".replace(".0%", "%") for q, *_ in table]
ax.bar(qs, [r * 100 for _, r, *_ in table], color=BLUE, width=0.6)
for i, (_, r, p, tp, fp, apc) in enumerate(table):
    ax.text(i, r * 100 + 2, f"{r:.0%}\n{apc:.1f} false\nalarms/catch", ha="center", fontsize=8.5)
ax.set_ylim(0, 115); ax.set_ylabel("% of all fraud caught"); ax.grid(axis="x", visible=False)
ax.set_title("Flag more transactions: catch more, but more false alarms", loc="left", fontsize=11.5)
fig.tight_layout()
save(fig, "14_precision_recall.png")

# 3. AI-generated scams --------------------------------------------------------------------
say("\n== 3. Are AI-generated scam frauds harder to catch? ==")
fraud = y == 1
ai = train["is_ai_generated_scam_attempt"].values == 1
t5 = np.quantile(best, 0.95)
for lbl, m in [("AI-scam frauds", fraud & ai), ("other frauds", fraud & ~ai)]:
    say(f"{lbl:15s} n={m.sum():6,}  caught in top 5%: {(best[m] >= t5).mean():.1%}   median suspicion rank: {pd.Series(best).rank(pct=True)[m].median():.3f}")
say(f"AI-scam attempts that are fraud: {y[ai].mean():.1%}   vs other transactions: {y[~ai].mean():.2%}")

# 4. Authentication ------------------------------------------------------------------------------
say("\n== 4. Does strong authentication suppress fraud? ==")
g = train.groupby("auth_method")["is_fraud"].agg(["mean", "size"]).sort_values("mean")
risky = train[(train[["ip_country_mismatch", "is_foreign_transaction", "billing_shipping_mismatch"]].sum(axis=1) > 0)]
g2 = risky.groupby("auth_method")["is_fraud"].mean()
say("auth method        fraud rate (all)   fraud rate (when another red flag is present)")
for a, r in g.iterrows():
    say(f"{a:18s} {r['mean']:>8.2%}            {g2[a]:>8.2%}")
fig, ax = plt.subplots(figsize=(10, 3.8))
idx = np.arange(len(g))
ax.bar(idx - 0.2, g["mean"] * 100, width=0.38, color=BLUE, label="All transactions")
ax.bar(idx + 0.2, g2.reindex(g.index) * 100, width=0.38, color=ORANGE, label="With a location/address red flag")
for i, a in enumerate(g.index):
    ax.text(i - 0.2, g.loc[a, "mean"] * 100 + 0.3, f"{g.loc[a, 'mean'] * 100:.1f}", ha="center", fontsize=9)
    ax.text(i + 0.2, g2[a] * 100 + 0.3, f"{g2[a] * 100:.1f}", ha="center", fontsize=9)
ax.set_xticks(idx); ax.set_xticklabels(g.index); ax.set_ylabel("% fraud"); ax.grid(axis="x", visible=False)
ax.set_title("Stronger authentication = less fraud, even when other red flags are present", loc="left", fontsize=12)
ax.legend(frameon=False, loc="upper left")
save(fig, "15_authentication.png")

with open(os.path.join(HERE, "analysis_results.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
