"""Phase 2 diagrams for the beginner guide (images 21-24). Reads results.csv, oof/ and the key cache."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from common import OOF_DIR, RESULTS, load  # noqa: E402
from fe_keys import build_key_store, encode_keys  # noqa: E402

IMG = os.path.join(HERE, "images")
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, LIGHT_BLUE, LIGHT_ORANGE, LIGHT_GRAY = "#2a78d6", "#eb6834", "#cde2fb", "#fbe0d4", "#f0efec"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "font.family": "DejaVu Sans",
    "font.size": 11, "text.color": INK, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "axes.axisbelow": True, "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlecolor": INK})
from matplotlib.patches import FancyBboxPatch  # noqa: E402


def save(fig, name):
    fig.savefig(os.path.join(IMG, name), dpi=160, bbox_inches="tight"); plt.close(fig)


def box(ax, x, y, w, h, text, fc, tc=INK, fs=10.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.04", fc=fc, ec=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs, fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.8, mutation_scale=16))


train, test, y = load()
res = pd.read_csv(RESULTS).set_index("iteration")

# 21. Phase 2 progress ----------------------------------------------------------------------
steps = [("v4", "v4_lightgbm_tuned", "Tuned\nLightGBM"), ("v7", "v7_ensemble", "Old\nblend"), ("v9", "v9_lightgbm_woe", "+ per-value\nscores"),
         ("v11", "v11_ebm", "EBM"), ("v15", "v15_stack", "Stack 1"), ("v19", "v19_sparse_glm", "Sparse\nGLM"),
         ("v16", "v16_glm_lgb", "GLM-boost\n+ digits"), ("v17", "v17_glm_lgb10", "GLM-boost\n10 folds"), ("v23", "v23_stack", "Final\nstack")]
vals = [res.loc[k, "cv_auc"] for _, k, _ in steps]
fig, ax = plt.subplots(figsize=(11.5, 4.3))
ax.plot(range(len(vals)), np.maximum.accumulate(vals), color=AXIS, lw=1.5, ls="--", zorder=1, label="best so far")
ax.scatter(range(len(vals)), vals, color=[BLUE if v >= 0.951 else ORANGE if v >= 0.9485 else MUTED for v in vals], s=80,
           edgecolor=SURFACE, linewidth=2, zorder=3)
for i, v in enumerate(vals):
    ax.text(i, v + 0.00025, f"{v:.4f}", ha="center", fontsize=9.5, fontweight="bold")
ax.set_xticks(range(len(vals))); ax.set_xticklabels([f"{a}\n{c}" for a, _, c in steps], fontsize=8.5)
ax.axhline(0.944, color=ORANGE, lw=1, ls=":"); ax.text(len(vals) - 0.5, 0.9442, "simulated ceiling of v4 (0.944)", color=ORANGE, fontsize=9, ha="right")
ax.set_ylim(0.9435, 0.9525); ax.set_ylabel("5-fold CV AUC"); ax.grid(axis="x", visible=False)
ax.set_title("Breaking the plateau: from 0.9474 to 0.9516", loc="left")
save(fig, "21_phase2_progress.png")

# 22. GLM-boost concept --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 3.2)); ax.set_xlim(0, 12); ax.set_ylim(0, 3.2); ax.axis("off")
box(ax, 0.1, 1.2, 2.2, 1.2, "73 keys\nexact values + digits\n+ pairs", LIGHT_GRAY)
arrow(ax, 2.35, 1.8, 2.95, 1.8)
box(ax, 3.0, 1.2, 2.2, 1.2, "Target-encode\ninside each fold\n(np.bincount)", LIGHT_BLUE)
arrow(ax, 5.25, 1.8, 5.85, 1.8)
box(ax, 5.9, 1.2, 2.4, 1.2, "GLM (logistic reg.)\n= first draft\nAUC 0.9506", BLUE, tc="#ffffff", bold=True)
arrow(ax, 8.35, 1.8, 8.95, 1.8)
box(ax, 9.0, 1.2, 2.9, 1.2, "4-leaf LightGBM starts\nfrom the GLM's score,\nlearns only corrections\nAUC 0.9512", ORANGE, tc="#ffffff", bold=True)
ax.text(0.1, 0.45, "The trees don't have to re-learn what the GLM already knows. Every tiny tree fixes a small bend or interaction.", color=INK2, fontsize=10)
save(fig, "22_glm_boost.png")

# 23. Digit keys: example + single-key AUC -----------------------------------------------------
knames, kmat, kcnt, ncat, cents = build_key_store(train, test)
tr, va = next(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
Ea, Eb, Et = encode_keys(kmat, ncat, y, len(train), tr, va, n_cols=60)
auc = lambda j: max(roc_auc_score(y[va], Eb[:, j]), 1 - roc_auc_score(y[va], Eb[:, j]))
raw_auc = lambda c: max(roc_auc_score(y[va], train[c].values[va]), 1 - roc_auc_score(y[va], train[c].values[va]))
cols = ["amount_usd", "hours_since_last_txn", "account_balance_usd", "distance_from_home_km"]
parts = ["int", "tenths", "cents", "lead2"]
fig, axes = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={"width_ratios": [4, 6]})
ax = axes[0]; ax.axis("off"); ax.set_xlim(0, 4); ax.set_ylim(0, 4)
ax.text(0.05, 3.6, "amount_usd = 1234.56 becomes 6 keys:", fontweight="bold")
for i, (k, v) in enumerate([("integer part", "1234"), ("tens (int10)", "123"), ("tenths", "12345"), ("cents", "56"),
                            ("lead1 (length + 1st digit)", "41"), ("lead2 (length + 2 digits)", "412")]):
    ax.text(0.15, 3.05 - i * 0.48, f"{k:26s}", fontsize=10, family="monospace", color=INK2)
    ax.text(3.0, 3.05 - i * 0.48, v, fontsize=10.5, family="monospace", fontweight="bold")
ax = axes[1]
x = np.arange(len(cols)); w = 0.16
ax.bar(x - 2 * w, [raw_auc(c) for c in cols], w, color=AXIS, label="raw value")
for i, p in enumerate(parts):
    ax.bar(x + (i - 1) * w, [auc(knames.index(f"{c}__{p}")) for c in cols], w, color=[BLUE, LIGHT_BLUE, ORANGE, "#1baf7a"][i], label=p)
ax.set_xticks(x); ax.set_xticklabels([c.replace("_", " ") for c in cols], fontsize=9.5); ax.set_ylim(0.5, 0.6)
ax.set_ylabel("AUC on its own (fold 0)"); ax.grid(axis="x", visible=False); ax.legend(frameon=False, fontsize=9, ncol=5, loc="upper right")
ax.set_title("Digits carry signal the raw number doesn't", loc="left", fontsize=12)
fig.tight_layout()
save(fig, "23_digit_keys.png")

# 24. Calibration of v4 -------------------------------------------------------------------------
p = np.load(os.path.join(OOF_DIR, "v4_lightgbm_tuned_oof.npy"))
q = pd.qcut(pd.Series(p).rank(method="first"), 20, labels=False)
cal = pd.DataFrame({"q": q, "p": p, "y": y}).groupby("q").agg(pred=("p", "mean"), real=("y", "mean"))
fig, ax = plt.subplots(figsize=(6.2, 5.2))
ax.plot([1e-5, 1], [1e-5, 1], color=AXIS, ls="--", lw=1.2, label="perfect calibration")
ax.scatter(cal.pred, cal.real.clip(lower=1e-5), color=BLUE, s=55, edgecolor=SURFACE, linewidth=1.5, zorder=3, label="20 groups of v4 predictions")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(3e-5, 0.5); ax.set_ylim(1e-5, 0.5)
ax.set_xlabel("Average predicted fraud probability"); ax.set_ylabel("Real fraud rate")
ax.set_title("v4's probabilities are honest (calibrated)", loc="left", fontsize=12); ax.legend(frameon=False, loc="upper left", fontsize=9.5)
save(fig, "24_calibration.png")
print("phase 2 images saved")
