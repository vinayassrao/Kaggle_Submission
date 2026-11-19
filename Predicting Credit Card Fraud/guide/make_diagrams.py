"""Generates every diagram for the credit card fraud beginner guide (guide/images/*.png).

Reads the real data and results.csv / oof predictions, so numbers always match the latest runs.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from common import FLAGS, NUMS, OOF_DIR, RESULTS, load  # noqa: E402

IMG = os.path.join(HERE, "images")
os.makedirs(IMG, exist_ok=True)

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
LIGHT_BLUE, LIGHT_ORANGE, LIGHT_GRAY = "#cde2fb", "#fbe0d4", "#f0efec"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlecolor": INK,
})


def save(fig, name):
    fig.savefig(os.path.join(IMG, name), dpi=160, bbox_inches="tight")
    plt.close(fig)


def box(ax, x, y, w, h, text, fc, ec=None, tc=INK, fs=11, bold=False, r=0.04):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec or fc, lw=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs,
            fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2, color=MUTED):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8, mutation_scale=16))


def blank(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


train, test, y = load()
results = pd.read_csv(RESULTS)
results = results[results.iteration.str.match(r"v\d")].sort_values("iteration").reset_index(drop=True)

# 1. Imbalance: 1000 dots -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 2.8))
rate = y.mean()
n_fraud = int(round(rate * 1000))
cols, rows = 50, 20
xs, ys = np.meshgrid(np.arange(cols), np.arange(rows))
colors = np.array([AXIS] * 1000, dtype=object)
colors[np.random.default_rng(1).choice(1000, n_fraud, replace=False)] = ORANGE
ax.scatter(xs.ravel(), ys.ravel(), c=list(colors), s=28, edgecolor="none")
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title(f"1,000 transactions: only {n_fraud} are fraud (orange) = {rate * 100:.2f}%", loc="left")
save(fig, "01_imbalance.png")

# 2. Workflow / iterations -------------------------------------------------------------
fig, ax = blank(12.5, 2.6)
steps = [("v1", "Logistic\nregression"), ("v2", "LightGBM"), ("v3", "+ new\nfeatures"), ("v4", "Tuned\nLightGBM"),
         ("v5", "XGBoost"), ("v6", "CatBoost"), ("v7", "Ensemble\n(blend)")]
w, gap = 1.5, 0.25
for i, (v, s) in enumerate(steps):
    x = 0.1 + i * (w + gap)
    dark = v in ("v4", "v7")
    box(ax, x, 0.7, w, 1.3, f"{v}\n{s}", BLUE if dark else LIGHT_BLUE, tc="#ffffff" if dark else INK, fs=10.5, bold=True)
    if i < len(steps) - 1:
        arrow(ax, x + w + 0.02, 1.35, x + w + gap - 0.02, 1.35)
ax.text(0.1, 0.25, "Each version is scored the same way (5-fold CV AUC). Keep a change only if the score goes up by more than the noise.",
        color=INK2, fontsize=10)
save(fig, "02_iterations.png")

# 3. Red-flag fraud rates -------------------------------------------------------------
labels = {"ip_country_mismatch": "IP country mismatch", "is_ai_generated_scam_attempt": "AI scam attempt",
          "billing_shipping_mismatch": "Billing/shipping mismatch", "is_foreign_transaction": "Foreign transaction",
          "used_vpn": "Used a VPN", "is_new_merchant": "New merchant"}
rates = {labels[f]: (train.loc[train[f] == 1, "is_fraud"].mean() * 100, train.loc[train[f] == 0, "is_fraud"].mean() * 100)
         for f in FLAGS}
order = sorted(rates, key=lambda k: rates[k][0])
fig, ax = plt.subplots(figsize=(10, 3.8))
yy = np.arange(len(order))
ax.barh(yy + 0.2, [rates[k][1] for k in order], height=0.38, color=AXIS, label="Flag = No")
ax.barh(yy - 0.2, [rates[k][0] for k in order], height=0.38, color=ORANGE, label="Flag = Yes")
for i, k in enumerate(order):
    ax.text(rates[k][0] + 0.2, i - 0.2, f"{rates[k][0]:.1f}%", va="center", fontsize=9.5, color=INK)
    ax.text(rates[k][1] + 0.2, i + 0.2, f"{rates[k][1]:.1f}%", va="center", fontsize=9, color=INK2)
ax.set_yticks(yy); ax.set_yticklabels(order); ax.set_xlabel("% of transactions that are fraud")
ax.set_title("Each red flag makes fraud 4-12x more likely", loc="left"); ax.grid(axis="y", visible=False)
ax.legend(frameon=False, loc="lower right")
save(fig, "03_red_flags.png")

# 4. Fraud rate by auth method & merchant ------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [5, 12]})
for ax, col, ttl in [(axes[0], "auth_method", "By how the card was checked"),
                     (axes[1], "merchant_category", "By type of shop")]:
    r = train.groupby(col)["is_fraud"].mean().sort_values(ascending=False) * 100
    ax.bar(range(len(r)), r.values, color=BLUE, width=0.65)
    ax.set_xticks(range(len(r))); ax.set_xticklabels(r.index, rotation=35, ha="right", fontsize=9)
    for i, v in enumerate(r.values):
        ax.text(i, v + 0.1, f"{v:.1f}", ha="center", fontsize=8.5, color=INK)
    ax.axhline(rate * 100, color=MUTED, lw=1, ls="--"); ax.set_title(ttl, loc="left"); ax.grid(axis="x", visible=False)
axes[0].set_ylabel("% fraud")
axes[1].text(len(r) - 0.5, rate * 100 + 0.2, f"average {rate * 100:.1f}%", ha="right", color=MUTED, fontsize=9)
fig.tight_layout()
save(fig, "04_categories.png")

# 5. Single-feature AUC ---------------------------------------------------------------------
aucs = {c: max(roc_auc_score(y, train[c]), 1 - roc_auc_score(y, train[c])) for c in NUMS}
order = sorted(aucs, key=aucs.get)
fig, ax = plt.subplots(figsize=(10, 4.4))
ax.barh(order, [aucs[c] - 0.5 for c in order], left=0.5, color=BLUE, height=0.6)
for i, c in enumerate(order):
    ax.text(aucs[c] + 0.004, i, f"{aucs[c]:.3f}", va="center", fontsize=9.5)
ax.set_xlim(0.5, 0.8); ax.set_xlabel("AUC using only this one number (0.5 = useless coin flip)")
ax.set_title("How much does each number tell us on its own?", loc="left"); ax.grid(axis="y", visible=False)
save(fig, "05_single_feature_auc.png")

# 6. What AUC means: ranking line-up ------------------------------------------------------------
fig, ax = blank(11, 3.3)
ax.text(0.1, 3.0, "The model gives every transaction a 'suspicion score', then we line them up:", fontweight="bold")
good = ["F", "F", "F", "o", "F", "o", "o", "o", "o", "o", "o", "o"]
for i, g in enumerate(good):
    box(ax, 0.2 + i * 0.88, 1.6, 0.78, 0.8, "FRAUD" if g == "F" else "ok", ORANGE if g == "F" else LIGHT_GRAY,
        tc="#ffffff" if g == "F" else INK2, fs=8.5, bold=g == "F")
arrow(ax, 0.2, 1.3, 10.6, 1.3, INK2)
ax.text(0.2, 0.95, "most suspicious", color=INK2, fontsize=9.5)
ax.text(10.6, 0.95, "least suspicious", color=INK2, fontsize=9.5, ha="right")
ax.text(0.1, 0.25, "AUC = the chance that a random fraud is ranked above a random normal one. "
        "1.0 = perfect line-up, 0.5 = shuffled at random.", color=INK, fontsize=10)
save(fig, "06_auc_lineup.png")

# 7. ROC curves of each iteration ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 5.5))
palette = {"v1_logistic_regression": (MUTED, "v1 Logistic"), "v4_lightgbm_tuned": (BLUE, "v4 Tuned LightGBM"),
           "v7_ensemble": (ORANGE, "v7 Ensemble")}
for name, (c, lbl) in palette.items():
    p = os.path.join(OOF_DIR, f"{name}_oof.npy")
    if os.path.exists(p):
        oof = np.load(p)
        fpr, tpr, _ = roc_curve(y, oof)
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{lbl} (AUC {roc_auc_score(y, oof):.4f})")
ax.plot([0, 1], [0, 1], color=AXIS, lw=1.2, ls="--", label="Random guess (AUC 0.5)")
ax.set_xlabel("False alarms: % of normal transactions flagged")
ax.set_ylabel("Frauds caught: % of fraud flagged")
ax.set_title("ROC curve: catch more fraud with fewer false alarms", loc="left", fontsize=12)
ax.legend(frameon=False, loc="lower right", fontsize=9.5)
save(fig, "07_roc_curves.png")

# 8. Boosting idea -------------------------------------------------------------------------------
fig, ax = blank(11.5, 3.0)
items = [("Tree 1", "learns the\nbig pattern"), ("Tree 2", "fixes Tree 1's\nmistakes"),
         ("Tree 3", "fixes what's\nstill wrong"), ("...", "x 2,000\ntrees"), ("Sum of all", "final score")]
for i, (t, s) in enumerate(items):
    x = 0.1 + i * 2.3
    last = i == len(items) - 1
    box(ax, x, 1.0, 1.9, 1.5, f"{t}\n\n{s}", BLUE if last else LIGHT_BLUE, tc="#ffffff" if last else INK, fs=10, bold=last)
    if i < len(items) - 1:
        arrow(ax, x + 1.92, 1.75, x + 2.28, 1.75)
ax.text(0.1, 0.4, "Gradient boosting (LightGBM, XGBoost, CatBoost): small trees built one after another, each correcting the last.",
        color=INK2, fontsize=10)
save(fig, "08_boosting.png")

# 9. Overfitting: early stopping curve (illustrative from v2-style training) ----------------------
import lightgbm as lgb  # noqa: E402
from common import raw_features  # noqa: E402
X, _ = raw_features(train, test)
idx = np.random.default_rng(0).permutation(len(X))
tr, va = idx[: int(0.8 * len(X))], idx[int(0.8 * len(X)):]
curves = {}
for lbl, params in [("v2: big trees, fast learning", dict(learning_rate=0.05, num_leaves=31)),
                    ("v4: small trees, slow learning", dict(learning_rate=0.02, num_leaves=31, max_depth=4, min_child_samples=300,
                                                           colsample_bytree=0.4, subsample=0.7, subsample_freq=1, reg_lambda=10, cat_smooth=50))]:
    ev = {}
    m = lgb.LGBMClassifier(objective="binary", n_estimators=3000, random_state=42, verbose=-1, **params)
    m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], eval_metric="auc", callbacks=[lgb.record_evaluation(ev)])
    curves[lbl] = ev["valid_0"]["auc"]
fig, ax = plt.subplots(figsize=(10, 3.8))
for (lbl, c), col in zip(curves.items(), [ORANGE, BLUE]):
    ax.plot(np.arange(1, len(c) + 1), c, color=col, lw=2, label=lbl)
    b = int(np.argmax(c))
    ax.scatter([b + 1], [c[b]], color=col, s=60, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.text(b + 1, c[b] + 0.0006, f"best {c[b]:.4f} @ {b + 1} trees", color=INK, fontsize=9, ha="center")
ax.set_ylim(0.935, 0.952); ax.set_xlabel("Number of trees"); ax.set_ylabel("AUC on unseen data")
ax.set_title("Fast learners peak early then get worse (overfitting); slow learners climb higher", loc="left", fontsize=12)
ax.legend(frameon=False, loc="lower right")
save(fig, "09_learning_curves.png")

# 10. Iteration progress ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10.5, 4.2))
names = [r.split("_")[0] for r in results.iteration]
pretty = {"v1": "v1 Logistic\nregression", "v2": "v2 LightGBM", "v3": "v3 + features", "v4": "v4 Tuned\nLightGBM",
          "v5": "v5 XGBoost", "v6": "v6 CatBoost", "v7": "v7 Ensemble"}
best_so_far = results.cv_auc.cummax()
ax.plot(range(len(results)), best_so_far, color=AXIS, lw=1.5, ls="--", label="best so far", zorder=1)
ax.scatter(range(len(results)), results.cv_auc, color=BLUE, s=70, edgecolor=SURFACE, linewidth=2, zorder=3, label="this version")
for i, a in enumerate(results.cv_auc):
    ax.text(i, a + 0.0004, f"{a:.4f}", ha="center", fontsize=9.5, fontweight="bold")
ax.set_xticks(range(len(results))); ax.set_xticklabels([pretty.get(n, n) for n in names], fontsize=9)
ax.set_ylabel("5-fold CV AUC"); ax.set_title("Score of every version", loc="left")
lo = results.cv_auc.min()
ax.set_ylim(lo - 0.001, results.cv_auc.max() + 0.0015); ax.grid(axis="x", visible=False)
ax.legend(frameon=False, loc="lower right")
save(fig, "10_iterations_progress.png")

# 11. Rank averaging -----------------------------------------------------------------------------
fig, ax = blank(11, 3.2)
rows_ = [("Transaction", ["A", "B", "C", "D"]), ("LightGBM prob.", ["0.91", "0.40", "0.05", "0.02"]),
         ("CatBoost prob.", ["0.30", "0.55", "0.08", "0.01"]),
         ("LightGBM rank", ["1.00", "0.67", "0.33", "0.00"]), ("CatBoost rank", ["0.67", "1.00", "0.33", "0.00"]),
         ("Average rank", ["0.83", "0.83", "0.33", "0.00"])]
for r_, (lbl, vals) in enumerate(rows_):
    yv = 2.75 - r_ * 0.47
    head = r_ == 0
    hl = r_ == len(rows_) - 1
    ax.text(0.1, yv + 0.19, lbl, va="center", fontsize=10, fontweight="bold" if head or hl else "normal")
    for j, v in enumerate(vals):
        box(ax, 2.6 + j * 1.3, yv, 1.2, 0.4, v, INK2 if head else (BLUE if hl else LIGHT_GRAY),
            tc="#ffffff" if head or hl else INK, fs=9.5, bold=head or hl, r=0.02)
ax.text(8.1, 1.4, "Probabilities from different models\nare on different scales.\nRanks put them on the same scale\nbefore averaging.",
        color=INK2, fontsize=9.5, va="center")
save(fig, "11_rank_average.png")

# 12. Correlation between models' predictions --------------------------------------------------
models = [m for m in ["v1_logistic_regression", "v4_lightgbm_tuned", "v5_xgboost", "v6_catboost"]
          if os.path.exists(os.path.join(OOF_DIR, f"{m}_oof.npy"))]
if len(models) >= 2:
    from scipy.stats import spearmanr
    P = np.column_stack([np.load(os.path.join(OOF_DIR, f"{m}_oof.npy")) for m in models])
    C = spearmanr(P).correlation
    C = np.atleast_2d(C)
    short = [m.split("_", 1)[1].replace("_", " ") for m in models]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(C, cmap="Blues", vmin=0.8, vmax=1)
    for i in range(len(models)):
        for j in range(len(models)):
            ax.text(j, i, f"{C[i, j]:.3f}", ha="center", va="center", color="#ffffff" if C[i, j] > 0.93 else INK, fontsize=10)
    ax.set_xticks(range(len(models))); ax.set_xticklabels(short, rotation=30, ha="right")
    ax.set_yticks(range(len(models))); ax.set_yticklabels(short); ax.grid(False)
    ax.set_title("How similar are the models' rankings?", loc="left", fontsize=12)
    save(fig, "12_model_correlation.png")

print("Saved", len(os.listdir(IMG)), "images")
