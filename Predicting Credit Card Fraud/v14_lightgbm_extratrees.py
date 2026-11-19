"""v14 - LightGBM extra_trees (random split thresholds) on the v9 feature set: a lower-variance, more diverse booster."""
import pandas as pd

from common import SEED, load, raw_features
from fe_v9 import deterministic_features, run_cv_ff
from v9_lightgbm_woe import fit_predict

PARAMS = dict(objective="binary", metric="auc", n_estimators=30000, learning_rate=0.02, max_depth=5, num_leaves=24,
              min_child_samples=300, colsample_bytree=0.5, subsample=0.7, subsample_freq=1, reg_lambda=10,
              cat_smooth=50, max_bin=1023, extra_trees=True, random_state=SEED, verbose=-1, n_jobs=-1)

train, test, y = load()
X0, Xt0 = raw_features(train, test)
D, Dt = deterministic_features(train, test)
run_cv_ff("v14_lightgbm_extratrees", lambda *a: fit_predict(*a, params=PARAMS), pd.concat([X0, D], axis=1),
          pd.concat([Xt0, Dt], axis=1), y, train, test, woe_kwargs=dict(m=40.0),
          description="LightGBM extra_trees, depth 5 + v9 features")
