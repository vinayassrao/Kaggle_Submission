"""Generates every diagram used in the beginner guide (guide/images/*.png)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "images")
os.makedirs(IMG, exist_ok=True)
train = pd.read_csv(os.path.join(HERE, "..", "data", "train.csv"))

# Palette (light mode, validated reference palette)
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
LIGHT_BLUE, LIGHT_ORANGE, LIGHT_GRAY = "#cde2fb", "#fbe0d4", "#f0efec"
SURV, DIED = BLUE, ORANGE

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
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2, color=MUTED):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8, mutation_scale=16))


def blank(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


# 1. Workflow ---------------------------------------------------------------
fig, ax = blank(12, 2.6)
steps = ["1. Get the\ndata", "2. Look at\nthe data", "3. Clean\n(fill gaps)", "4. Make\nfeatures",
         "5. Train a\nmodel", "6. Check it\n(CV)", "7. Predict &\nsubmit"]
w, gap = 1.45, 0.22
for i, s in enumerate(steps):
    x = 0.1 + i * (w + gap)
    box(ax, x, 0.8, w, 1.2, s, LIGHT_BLUE if i < 4 else BLUE, tc=INK if i < 4 else "#ffffff", fs=10.5, bold=True)
    if i < len(steps) - 1:
        arrow(ax, x + w + 0.02, 1.4, x + w + gap - 0.02, 1.4)
ax.text(0.1 + 2 * (w + gap) - gap / 2, 0.45, "Preparing the data (most of the work!)", ha="center", color=INK2, fontsize=10)
ax.text(0.1 + 5.5 * (w + gap) - gap / 2, 0.45, "Machine learning", ha="center", color=INK2, fontsize=10)
save(fig, "01_workflow.png")

# 2. Train vs test ------------------------------------------------------------
fig, ax = blank(11.4, 3.6)
cols = ["Name", "Sex", "Age", "Class", "Survived"]
tr_rows = [["Braund, Mr. Owen", "male", "22", "3", "0"], ["Cumings, Mrs. John", "female", "38", "1", "1"],
           ["Heikkinen, Miss Laina", "female", "26", "3", "1"]]
te_rows = [["Kelly, Mr. James", "male", "34.5", "3", "?"], ["Wilkes, Mrs. James", "female", "47", "3", "?"],
           ["Myles, Mr. Thomas", "male", "62", "2", "?"]]
cw = [2.0, 0.65, 0.5, 0.55, 0.85]


def table(x0, title, rows, color, light):
    ax.text(x0, 3.3, title, fontsize=12, fontweight="bold", color=INK)
    y = 2.65
    x = x0
    for c, wdt in zip(cols, cw):
        box(ax, x, y, wdt - 0.04, 0.4, c, color, tc="#ffffff", fs=9.5, bold=True, r=0.02)
        x += wdt
    for r in rows:
        y -= 0.45
        x = x0
        for j, (v, wdt) in enumerate(zip(r, cw)):
            fc = light if j == 4 else LIGHT_GRAY
            box(ax, x, y, wdt - 0.04, 0.4, v, fc, fs=9, bold=(j == 4), r=0.02)
            x += wdt


table(0.1, "train.csv  (891 people, answers given)", tr_rows, BLUE, LIGHT_BLUE)
table(6.0, "test.csv  (418 people, answers hidden)", te_rows, ORANGE, LIGHT_ORANGE)
ax.text(0.1, 0.55, "The model LEARNS from these", color=INK2, fontsize=10.5)
ax.text(6.0, 0.55, "The model GUESSES the '?' for these", color=INK2, fontsize=10.5)
save(fig, "02_train_test.png")

# 3. Missing values -------------------------------------------------------------
full = pd.concat([train, pd.read_csv(os.path.join(HERE, "..", "data", "test.csv"))])
miss = (full.drop(columns="Survived").isna().mean() * 100).sort_values()
miss = miss[miss > 0]
fig, ax = plt.subplots(figsize=(8, 2.6))
ax.barh(miss.index, miss.values, color=BLUE, height=0.55)
for i, v in enumerate(miss.values):
    ax.text(v + 1, i, f"{v:.1f}% missing", va="center", color=INK2, fontsize=10)
ax.set_xlim(0, 100); ax.set_xlabel("% of passengers with a blank value")
ax.set_title("Which columns have gaps?", loc="left"); ax.grid(axis="y", visible=False)
save(fig, "03_missing.png")

# 4. Survival rates by sex / class / title -----------------------------------------
title = train["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
title = title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
train["Title"] = title.where(title.isin(["Mr", "Mrs", "Miss", "Master"]), "Rare")
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw={"width_ratios": [2, 3, 5]})
groups = [("Sex", ["female", "male"], "By sex"), ("Pclass", [1, 2, 3], "By ticket class"),
          ("Title", ["Mrs", "Miss", "Master", "Rare", "Mr"], "By title in name")]
for ax, (col, order, ttl) in zip(axes, groups):
    rates = train.groupby(col)["Survived"].mean().reindex(order) * 100
    labels = [f"Class {o}" if col == "Pclass" else str(o) for o in order]
    ax.bar(labels, rates.values, color=BLUE, width=0.6)
    for i, v in enumerate(rates.values):
        ax.text(i, v + 2, f"{v:.0f}%", ha="center", color=INK, fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, 105); ax.set_title(ttl, loc="left"); ax.grid(axis="x", visible=False)
    ax.axhline(38.4, color=MUTED, lw=1, ls="--")
axes[0].set_ylabel("% who survived")
axes[2].text(4.4, 41, "average 38%", color=MUTED, fontsize=9, ha="right")
fig.tight_layout()
save(fig, "04_survival_rates.png")

# 5. Age: survived vs died (stacked by age bin) -----------------------------------
bins = np.arange(0, 85, 5)
t = train.dropna(subset=["Age"])
s_counts, _ = np.histogram(t.loc[t.Survived == 1, "Age"], bins)
d_counts, _ = np.histogram(t.loc[t.Survived == 0, "Age"], bins)
fig, ax = plt.subplots(figsize=(10, 3.6))
centers = bins[:-1] + 2.5
ax.bar(centers, s_counts, width=4.6, color=SURV, label="Survived")
ax.bar(centers, d_counts, width=4.6, bottom=s_counts, color=DIED, label="Died", edgecolor=SURFACE, linewidth=2)
ax.set_xlabel("Age (years)"); ax.set_ylabel("Number of passengers")
ax.set_title("Children under ~10 survived much more often", loc="left")
ax.legend(frameon=False, loc="upper right"); ax.grid(axis="x", visible=False)
save(fig, "05_age.png")

# 6. One-hot encoding --------------------------------------------------------------
fig, ax = blank(10, 2.9)
ax.text(0.1, 2.6, "Before (words)", fontweight="bold")
box(ax, 0.1, 1.95, 1.6, 0.4, "Embarked", INK2, tc="#ffffff", bold=True, r=0.02)
for i, v in enumerate(["S", "C", "Q"]):
    box(ax, 0.1, 1.5 - i * 0.45, 1.6, 0.4, v, LIGHT_GRAY, r=0.02)
arrow(ax, 2.1, 1.3, 3.3, 1.3, BLUE)
ax.text(2.7, 1.45, "get_dummies", ha="center", color=BLUE, fontsize=10)
ax.text(3.6, 2.6, "After (numbers a computer can use)", fontweight="bold")
vals = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
for j, c in enumerate(["Embarked_S", "Embarked_C", "Embarked_Q"]):
    box(ax, 3.6 + j * 1.75, 1.95, 1.7, 0.4, c, BLUE, tc="#ffffff", bold=True, fs=9.5, r=0.02)
    for i in range(3):
        box(ax, 3.6 + j * 1.75, 1.5 - i * 0.45, 1.7, 0.4, str(vals[i][j]),
            LIGHT_BLUE if vals[i][j] else LIGHT_GRAY, bold=bool(vals[i][j]), r=0.02)
save(fig, "06_one_hot.png")

# 7. A real decision tree (numbers from the training data) ---------------------------
def rate(mask):
    return f"{train.loc[mask, 'Survived'].mean() * 100:.0f}% survived  (n={mask.sum()})"


fem, mal = train.Sex == "female", train.Sex == "male"
boy = mal & (train.Title == "Master")
c3 = train.Pclass == 3
fig, ax = blank(12, 4.6)
box(ax, 4.5, 3.6, 3, 0.8, f"Is the passenger male?\n{rate(train.Sex.notna())}", LIGHT_GRAY, ec=AXIS, fs=10)
box(ax, 1.0, 2.0, 3.6, 0.8, f"Is she in 3rd class?\n{rate(fem)}", LIGHT_GRAY, ec=AXIS, fs=10)
box(ax, 7.4, 2.0, 3.6, 0.8, f"Is he a boy (title 'Master')?\n{rate(mal)}", LIGHT_GRAY, ec=AXIS, fs=10)
leaves = [(0.0, fem & ~c3, "1st/2nd class woman", SURV), (2.6, fem & c3, "3rd class woman", SURV),
          (6.4, mal & ~boy, "Adult man", DIED), (9.0, boy, "Boy", SURV)]
for x, m, lbl, col in leaves:
    r_ = train.loc[m, "Survived"].mean()
    box(ax, x, 0.2, 2.5, 0.95, f"{lbl}\n{r_ * 100:.0f}% survived (n={m.sum()})\nGuess: {'LIVES' if r_ >= 0.5 else 'DIES'}",
        LIGHT_BLUE if r_ >= 0.5 else LIGHT_ORANGE, ec=col, fs=9.5)
arrow(ax, 5.3, 3.6, 3.2, 2.8); ax.text(3.9, 3.3, "No", color=INK2)
arrow(ax, 6.7, 3.6, 8.8, 2.8); ax.text(7.9, 3.3, "Yes", color=INK2)
arrow(ax, 2.2, 2.0, 1.3, 1.15); ax.text(1.35, 1.55, "No", color=INK2)
arrow(ax, 3.4, 2.0, 3.9, 1.15); ax.text(3.75, 1.55, "Yes", color=INK2)
arrow(ax, 8.6, 2.0, 7.7, 1.15); ax.text(7.75, 1.55, "No", color=INK2)
arrow(ax, 9.8, 2.0, 10.3, 1.15); ax.text(10.15, 1.55, "Yes", color=INK2)
save(fig, "07_decision_tree.png")

# 8. Random forest voting ---------------------------------------------------------
fig, ax = blank(11, 3.6)
box(ax, 0.1, 1.4, 1.9, 0.9, "New passenger\n(male, 3rd class,\nage 4)", LIGHT_GRAY, ec=AXIS, fs=9.5)
votes = ["Lives", "Lives", "Dies", "Lives", "Lives"]
for i, v in enumerate(votes):
    y = 2.95 - i * 0.62
    arrow(ax, 2.05, 1.85, 3.0, y + 0.22)
    box(ax, 3.05, y, 1.9, 0.45, f"Tree {i + 1}", LIGHT_GRAY, ec=AXIS, fs=10)
    arrow(ax, 5.0, y + 0.22, 5.6, y + 0.22)
    box(ax, 5.65, y, 1.1, 0.45, v, LIGHT_BLUE if v == "Lives" else LIGHT_ORANGE, fs=10, bold=True)
    arrow(ax, 6.8, y + 0.22, 7.7, 1.85)
box(ax, 7.75, 1.3, 3.1, 1.1, "Count the votes:\n4 Lives vs 1 Dies\nFinal answer: LIVES", BLUE, tc="#ffffff", fs=10.5, bold=True)
ax.text(3.05, 0.0, "Each tree sees a slightly different random slice of the data,\nso each makes different mistakes. Together they are wiser.",
        color=INK2, fontsize=10)
save(fig, "08_random_forest.png")

# 9. Cross-validation ----------------------------------------------------------------
fig, ax = blank(11, 3.5)
for r in range(5):
    y = 2.7 - r * 0.6
    ax.text(0.1, y + 0.2, f"Round {r + 1}", va="center", fontsize=10.5, color=INK)
    for c in range(5):
        test_ = c == r
        box(ax, 1.4 + c * 1.45, y, 1.38, 0.42, "Quiz" if test_ else "Study",
            ORANGE if test_ else LIGHT_BLUE, tc="#ffffff" if test_ else INK, fs=10, bold=test_, r=0.02)
    ax.text(8.85, y + 0.2, f"score {[83, 85, 82, 86, 84][r]}%", va="center", color=INK2, fontsize=10)
ax.text(1.4, 0.0, "Final CV score = average of the 5 rounds = 84%   (every passenger is quizzed exactly once)",
        color=INK, fontsize=10.5, fontweight="bold")
save(fig, "09_cross_validation.png")

# 10. Family group idea --------------------------------------------------------------
fig, ax = blank(11, 3.2)
ax.text(0.1, 2.9, "Same ticket / same surname = travelling together", fontweight="bold")
fam = [("Mother", "train: died", LIGHT_ORANGE), ("Daughter", "train: died", LIGHT_ORANGE),
       ("Son (boy)", "test: ?", LIGHT_GRAY)]
for i, (who, st, fc) in enumerate(fam):
    box(ax, 0.1 + i * 2.0, 1.6, 1.85, 0.9, f"{who}\n{st}", fc, ec=AXIS, fs=10)
arrow(ax, 6.1, 2.05, 7.0, 2.05)
box(ax, 7.05, 1.6, 3.8, 0.9, "Guess for the son: DIES\n(his family's women & children died)", ORANGE, tc="#ffffff", fs=10, bold=True)
ax.text(0.1, 0.6, "Families usually stayed together. If the mother and sisters did not make it,\n"
        "the young children with them probably did not either - and the other way round.", color=INK2, fontsize=10.5)
save(fig, "10_family_group.png")

# 11. Feature ablation (results from experiments.py) ---------------------------------
steps = ["Tutorial (class, sex,\nsiblings, parents)", "+ Title", "+ Age, Fare", "+ Family size,\nalone, child",
         "+ Port, deck, cabin", "+ Ticket size,\nfare per person", "+ Family-group\nsurvival"]
acc = [78.41, 82.45, 83.03, 83.23, 83.05, 83.46, 84.24]
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(range(len(acc)), acc, color=BLUE, lw=2, marker="o", markersize=8,
        markeredgecolor=SURFACE, markeredgewidth=2)
for i, a in enumerate(acc):
    ax.text(i, a + 0.35, f"{a:.1f}%", ha="center", fontsize=10, color=INK, fontweight="bold")
ax.set_xticks(range(len(acc))); ax.set_xticklabels(steps, fontsize=9)
ax.set_ylim(77, 86); ax.set_ylabel("Cross-validated accuracy (%)")
ax.set_title("Adding clues one at a time (random forest)", loc="left"); ax.grid(axis="x", visible=False)
save(fig, "11_feature_ablation.png")

# 12. Model comparison -----------------------------------------------------------------
models = ["Soft-vote\nensemble", "Gradient\nboosting", "Random\nforest", "Logistic\nregression",
          "Hist gradient\nboosting", "SVM", "Woman-child-\ngroup rule", "Gender rule\n(women live)"]
macc = [84.47, 84.42, 84.24, 84.15, 83.93, 83.70, 83.66, 78.68]
mstd = [2.10, 1.90, 2.14, 2.15, 2.07, 2.04, 2.05, 2.33]
fig, ax = plt.subplots(figsize=(10.5, 4))
ax.bar(range(len(macc)), macc, color=BLUE, width=0.6)
ax.errorbar(range(len(macc)), macc, yerr=mstd, fmt="none", ecolor=INK2, capsize=4, lw=1.2)
for i, a in enumerate(macc):
    ax.text(i + 0.33, a + 0.15, f"{a:.1f}", fontsize=9.5, color=INK)
ax.set_xticks(range(len(macc))); ax.set_xticklabels(models, fontsize=9)
ax.set_ylim(70, 90); ax.set_ylabel("CV accuracy (%)")
ax.set_title("Each model with its best clues: all very close (lines show +/- noise)", loc="left")
ax.grid(axis="x", visible=False)
save(fig, "12_models.png")

# 13. Overfitting -------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
rng = np.random.default_rng(3)
x = np.linspace(0, 10, 14); yv = 0.6 * x + rng.normal(0, 1.1, len(x))
xs = np.linspace(0, 10, 300)
fits = [("Too simple\n(underfitting)", np.full_like(xs, yv.mean())),
        ("Just right", np.polyval(np.polyfit(x, yv, 1), xs)),
        ("Memorised the noise\n(overfitting)", np.interp(xs, x, yv))]
for ax, (ttl, ys) in zip(axes, fits):
    ax.plot(xs, ys, color=BLUE, lw=2)
    ax.scatter(x, yv, color=ORANGE, s=40, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.set_title(ttl, loc="left", fontsize=11.5); ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
save(fig, "13_overfitting.png")

print("Saved", len(os.listdir(IMG)), "images to", IMG)
