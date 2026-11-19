"""Titanic model from Alexis Cook's Kaggle tutorial.

Random forest on Pclass, Sex, SibSp, Parch -> submission.csv
Works locally (data/ folder) or inside a Kaggle notebook (/kaggle/input/titanic/).
"""
import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

DATA_DIR = "/kaggle/input/titanic" if os.path.exists("/kaggle/input/titanic") else os.path.join(os.path.dirname(__file__), "data")

train_data = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_data = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

y = train_data["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({"PassengerId": test_data.PassengerId, "Survived": predictions})
out_path = "submission.csv" if DATA_DIR.startswith("/kaggle") else os.path.join(os.path.dirname(__file__), "submission.csv")
output.to_csv(out_path, index=False)

print(f"Training accuracy: {model.score(X, y):.4f}")
print(f"Women survival rate (train): {train_data.loc[train_data.Sex == 'female', 'Survived'].mean():.4f}")
print(f"Men survival rate (train):   {train_data.loc[train_data.Sex == 'male', 'Survived'].mean():.4f}")
print(f"Your submission was successfully saved to {out_path} ({len(output)} rows)")
