# Titanic - Machine Learning from Disaster

Competition: https://www.kaggle.com/competitions/titanic

Predict which passengers survived the Titanic shipwreck.

## Setup

```bash
python -m pip install pandas scikit-learn matplotlib
```

Download `train.csv` and `test.csv` from the competition's Data tab into `data/`.

## Files

| File | What it does | CV accuracy | Public score |
|---|---|---|---|
| `titanic_model.py` | Tutorial random forest (Pclass, Sex, SibSp, Parch) -> `submission.csv` | 79.8% | **0.77511** |
| `titanic_improved.py` | + title, age, fare, family features -> `submission_improved.csv` | 83.2% | - |
| `experiments.py` | Tests every feature group and model, writes `submissions/` | up to 84.5% | - |
| `submissions/` | Ready-to-upload predictions from the best experiments | | |
| `guide/Titanic_Beginner_Guide.md` | Beginner's guide to the whole project, with diagrams | | |

Run any script from inside this folder, e.g. `python experiments.py`.
