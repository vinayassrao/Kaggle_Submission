"""Improved Titanic model: adds Title, Age, Fare, Embarked, FamilySize, IsAlone.

Compares against the tutorial baseline with 10-fold cross-validation,
then writes submission_improved.csv.
Works locally (data/ folder) or inside a Kaggle notebook (/kaggle/input/titanic/).
"""
import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

ON_KAGGLE = os.path.exists("/kaggle/input/titanic")
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
DATA_DIR = "/kaggle/input/titanic" if ON_KAGGLE else os.path.join(HERE, "data")
OUT_PATH = "submission.csv" if ON_KAGGLE else os.path.join(HERE, "submission_improved.csv")

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
full = pd.concat([train.drop(columns="Survived"), test], ignore_index=True)

# Title from the name, e.g. "Braund, Mr. Owen Harris" -> "Mr"
title = full["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
title = title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
full["Title"] = title.where(title.isin(["Mr", "Mrs", "Miss", "Master"]), "Rare")

# Family features
full["FamilySize"] = full["SibSp"] + full["Parch"] + 1
full["IsAlone"] = (full["FamilySize"] == 1).astype(int)

# Missing values: Age by title median, Fare by class median, Embarked by mode
full["Age"] = full["Age"].fillna(full.groupby("Title")["Age"].transform("median"))
full["Fare"] = full["Fare"].fillna(full.groupby("Pclass")["Fare"].transform("median"))
full["Embarked"] = full["Embarked"].fillna(full["Embarked"].mode()[0])

features = ["Pclass", "Sex", "Age", "Fare", "Embarked", "Title", "FamilySize", "IsAlone", "SibSp", "Parch"]
X_all = pd.get_dummies(full[features], columns=["Sex", "Embarked", "Title"], dtype=int)
X, X_test = X_all.iloc[: len(train)], X_all.iloc[len(train):]
y = train["Survived"]

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

baseline_X = pd.get_dummies(train[["Pclass", "Sex", "SibSp", "Parch"]], dtype=int)
baseline = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
base_scores = cross_val_score(baseline, baseline_X, y, cv=cv)

model = RandomForestClassifier(n_estimators=500, max_depth=6, min_samples_leaf=2, random_state=1)
scores = cross_val_score(model, X, y, cv=cv)

print(f"Baseline (tutorial) CV accuracy: {base_scores.mean():.4f} +/- {base_scores.std():.4f}")
print(f"Improved model     CV accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")

model.fit(X, y)
output = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": model.predict(X_test)})
output.to_csv(OUT_PATH, index=False)
print(f"Saved {len(output)} predictions to {OUT_PATH}")
