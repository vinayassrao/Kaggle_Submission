"""Quick search over LightGBM regularisation settings (raw vs engineered features)."""
import lightgbm as lgb, numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from common import SEED, N_FOLDS, load, raw_features, engineered_features

train, test, y = load()
sets = {"raw": raw_features(train, test)[0], "eng": engineered_features(train, test)[0]}
base = dict(objective="binary", n_estimators=10000, random_state=SEED, verbose=-1, n_jobs=-1)
configs = {
    "A small trees":      dict(learning_rate=0.02, num_leaves=15, min_child_samples=200, colsample_bytree=0.5, subsample=0.8, subsample_freq=1, reg_lambda=5),
    "B tiny trees":       dict(learning_rate=0.02, num_leaves=7, min_child_samples=500, colsample_bytree=0.5, subsample=0.8, subsample_freq=1, reg_lambda=10),
    "C depth-limited":    dict(learning_rate=0.02, num_leaves=31, max_depth=4, min_child_samples=300, colsample_bytree=0.4, subsample=0.7, subsample_freq=1, reg_lambda=10, cat_smooth=50),
}
skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
for fs, X in sets.items():
    for name, cfg in configs.items():
        oof = np.zeros(len(X)); its = []
        for tr, va in skf.split(X, y):
            m = lgb.LGBMClassifier(**base, **cfg)
            m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], eval_metric="auc", callbacks=[lgb.early_stopping(300, verbose=False)])
            oof[va] = m.predict_proba(X.iloc[va])[:, 1]; its.append(m.best_iteration_)
        print(f"{fs:4s} {name:18s} AUC {roc_auc_score(y, oof):.5f}  trees~{int(np.mean(its))}", flush=True)
