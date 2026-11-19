"""Systematic search over Titanic improvements.

1. Feature ablation: add feature groups one at a time (random forest).
2. Model comparison on the full feature set.
3. Rule-based "woman-child-group" model (gender model + family-group fixes).
All scores are 5x repeated 5-fold CV. Group-survival features are computed
from training-fold labels only, so CV scores are not leaked.
Writes submission files for the best candidates into submissions/.
"""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "submissions")
os.makedirs(OUT_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(HERE, "data", "train.csv"))
test = pd.read_csv(os.path.join(HERE, "data", "test.csv"))
n_train = len(train)
full = pd.concat([train, test], ignore_index=True)  # Survived is NaN for test rows

# ---------------- static features (no labels used) ----------------
title = full["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
title = title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
full["Title"] = title.where(title.isin(["Mr", "Mrs", "Miss", "Master"]), "Rare")
full["IsMale"] = (full["Sex"] == "male").astype(int)
full["FamilySize"] = full["SibSp"] + full["Parch"] + 1
full["IsAlone"] = (full["FamilySize"] == 1).astype(int)
full["Age"] = full["Age"].fillna(full.groupby("Title")["Age"].transform("median"))
full["Fare"] = full["Fare"].fillna(full.groupby("Pclass")["Fare"].transform("median"))
full["LogFare"] = np.log1p(full["Fare"])
full["TicketCount"] = full.groupby("Ticket")["Ticket"].transform("count")
full["FarePerPerson"] = np.log1p(full["Fare"] / full["TicketCount"])
full["Embarked"] = full["Embarked"].fillna("S")
full["Deck"] = full["Cabin"].str[0].fillna("U").replace({"T": "U", "G": "U"})
full["HasCabin"] = full["Cabin"].notna().astype(int)
full["IsChild"] = (full["Age"] < 14).astype(int)
full["Surname"] = full["Name"].str.split(",").str[0]
# Family-group key: surname + class + ticket prefix + fare (groups real travelling families)
full["Group"] = (full["Surname"] + "-" + full["Pclass"].astype(str) + "-"
                 + full["Ticket"].str[:-1] + "-" + full["Fare"].round(1).astype(str))
# Women and children (boys) are the ones whose fate is shared within a family
full["WC"] = ((full["IsMale"] == 0) | (full["Title"] == "Master")).astype(int)


def group_survival(df, known_mask):
    """Survival rate of OTHER women/children in the same family or ticket group,
    using only labels where known_mask is True. Returns rate (0.5 if unknown) and a flag."""
    rate = pd.Series(0.5, index=df.index)
    known = pd.Series(0, index=df.index)
    for key in ["Group", "Ticket"]:
        lab = df["Survived"].where(known_mask & (df["WC"] == 1))
        s = lab.groupby(df[key]).transform("sum")
        c = lab.notna().groupby(df[key]).transform("sum")
        own = lab.fillna(0)
        own_c = lab.notna().astype(int)
        other_c = c - own_c
        other_rate = (s - own) / other_c.replace(0, np.nan)
        use = (other_c > 0) & (known == 0)
        rate[use] = other_rate[use]
        known[use] = 1
    return rate, known


FEATURE_SETS = {
    "1 tutorial (Pclass, Sex, SibSp, Parch)": ["Pclass", "IsMale", "SibSp", "Parch"],
    "2 + Title": ["Pclass", "IsMale", "SibSp", "Parch", "T_"],
    "3 + Age, Fare": ["Pclass", "IsMale", "SibSp", "Parch", "T_", "Age", "LogFare"],
    "4 + FamilySize, IsAlone, IsChild": ["Pclass", "IsMale", "SibSp", "Parch", "T_", "Age", "LogFare",
                                         "FamilySize", "IsAlone", "IsChild"],
    "5 + Embarked, Deck, HasCabin": ["Pclass", "IsMale", "SibSp", "Parch", "T_", "Age", "LogFare",
                                     "FamilySize", "IsAlone", "IsChild", "E_", "D_", "HasCabin"],
    "6 + Ticket size, fare per person": ["Pclass", "IsMale", "SibSp", "Parch", "T_", "Age", "LogFare",
                                         "FamilySize", "IsAlone", "IsChild", "E_", "D_", "HasCabin",
                                         "TicketCount", "FarePerPerson"],
    "7 + family-group survival": ["Pclass", "IsMale", "SibSp", "Parch", "T_", "Age", "LogFare",
                                  "FamilySize", "IsAlone", "IsChild", "E_", "D_", "HasCabin",
                                  "TicketCount", "FarePerPerson", "GroupRate", "GroupKnown"],
    "8 compact: Sex, Pclass, Title, Fam, group survival": ["Pclass", "IsMale", "T_", "FamilySize",
                                                           "LogFare", "GroupRate", "GroupKnown"],
}

dummies = pd.get_dummies(full[["Title", "Embarked", "Deck"]], prefix=["T", "E", "D"], dtype=int)
base = pd.concat([full, dummies], axis=1)


def matrix(df, spec):
    cols = []
    for c in spec:
        cols += [d for d in df.columns if d.startswith(c)] if c.endswith("_") else [c]
    return df[cols]


def build(known_mask):
    df = base.copy()
    df["GroupRate"], df["GroupKnown"] = group_survival(df, known_mask)
    return df


def wcg_predict(df, known_mask):
    """Woman-child-group rule: females survive, males die, except
    boys whose family women/children all survived -> survive,
    females whose family women/children all died -> die."""
    rate, known = group_survival(df, known_mask)
    pred = (df["IsMale"] == 0).astype(int)
    boys = (df["Title"] == "Master") & (known == 1) & (rate == 1)
    girls = (df["IsMale"] == 0) & (known == 1) & (rate == 0)
    pred[boys] = 1
    pred[girls] = 0
    return pred


MODELS = {
    "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)),
    "SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=1.0, gamma="scale", probability=True, random_state=1)),
    "Random forest": RandomForestClassifier(n_estimators=500, max_depth=6, min_samples_leaf=2, random_state=1),
    "Gradient boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                                    subsample=0.8, random_state=1),
    "Hist gradient boosting": HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200,
                                                             l2_regularization=1.0, random_state=1),
}
MODELS["Soft-vote ensemble"] = VotingClassifier(
    [(k, v) for k, v in MODELS.items() if k != "Hist gradient boosting"], voting="soft")

y = train["Survived"].astype(int).values
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1)
splits = list(cv.split(np.zeros(n_train), y))
train_mask_all = pd.Series(np.arange(len(full)) < n_train)

# Feature matrices per fold (group-survival built from that fold's training labels only)
fold_frames = []
for tr, va in splits:
    m = pd.Series(False, index=full.index)
    m.iloc[tr] = True
    fold_frames.append(build(m))


def cv_score(model, spec):
    accs = []
    for (tr, va), df in zip(splits, fold_frames):
        X = matrix(df, spec).iloc[:n_train].values
        model.fit(X[tr], y[tr])
        accs.append((model.predict(X[va]) == y[va]).mean())
    return np.mean(accs), np.std(accs)


results = []
print("=== 1. Feature ablation (random forest) ===")
rf = MODELS["Random forest"]
for name, spec in FEATURE_SETS.items():
    m, s = cv_score(rf, spec)
    results.append(("Random forest", name, m, s))
    print(f"{m:.4f} +/- {s:.4f}  {name}")

print("\n=== 2. Models on feature sets 5, 7 and 8 ===")
for fs in ["5 + Embarked, Deck, HasCabin", "7 + family-group survival",
           "8 compact: Sex, Pclass, Title, Fam, group survival"]:
    print(f"-- {fs}")
    for mname, model in MODELS.items():
        if mname == "Random forest":
            continue
        m, s = cv_score(model, FEATURE_SETS[fs])
        results.append((mname, fs, m, s))
        print(f"{m:.4f} +/- {s:.4f}  {mname}")

print("\n=== 3. Rule-based models ===")
gender = [((full["IsMale"].iloc[va] == 0).astype(int).values == y[va]).mean() for _, va in splits]
print(f"{np.mean(gender):.4f} +/- {np.std(gender):.4f}  Gender model (all women survive)")
wcg = []
for (tr, va), df in zip(splits, fold_frames):
    m = pd.Series(False, index=full.index)
    m.iloc[tr] = True
    wcg.append((wcg_predict(df, m).iloc[va].values == y[va]).mean())
print(f"{np.mean(wcg):.4f} +/- {np.std(wcg):.4f}  Woman-child-group rule")
results += [("Gender rule", "-", np.mean(gender), np.std(gender)),
            ("Woman-child-group rule", "-", np.mean(wcg), np.std(wcg))]

# ---------------- final fits on all training data -> submissions ----------------
final_df = build(train_mask_all)
pid = test["PassengerId"].values


def save(name, preds):
    path = os.path.join(OUT_DIR, f"{name}.csv")
    pd.DataFrame({"PassengerId": pid, "Survived": np.asarray(preds).astype(int)}).to_csv(path, index=False)
    return path


ranked = sorted(results, key=lambda r: -r[2])
print("\n=== Top 8 overall ===")
for mname, fs, m, s in ranked[:8]:
    print(f"{m:.4f} +/- {s:.4f}  {mname} | {fs}")

print("\n=== Submission files ===")
saved = set()
for mname, fs, m, s in ranked:
    if len(saved) >= 4:
        break
    key = f"{mname} | {fs}"
    if key in saved:
        continue
    slug = (mname + "_" + fs.split()[0]).lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
    if mname == "Woman-child-group rule":
        preds = wcg_predict(final_df, train_mask_all).iloc[n_train:].values
    elif mname == "Gender rule":
        preds = (test["Sex"] == "female").astype(int).values
    else:
        X = matrix(final_df, FEATURE_SETS[fs])
        model = MODELS[mname]
        model.fit(X.iloc[:n_train].values, y)
        preds = model.predict(X.iloc[n_train:].values)
    saved.add(key)
    print(f"CV {m:.4f}  ->  {save(slug, preds)}")
# Always also save the WCG rule, it often does better on the leaderboard than CV suggests
if "Woman-child-group rule | -" not in saved:
    print(f"CV {np.mean(wcg):.4f}  ->  {save('woman_child_group_rule', wcg_predict(final_df, train_mask_all).iloc[n_train:].values)}")
