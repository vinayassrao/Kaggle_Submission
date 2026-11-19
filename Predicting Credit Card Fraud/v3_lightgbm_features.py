"""v3 - Same LightGBM as v2, but with engineered features (common.engineered_features)."""
from common import load, engineered_features, run_cv
from v2_lightgbm import fit_predict

train, test, y = load()
X, X_test = engineered_features(train, test)
print("features:", X.shape[1])
run_cv("v3_lightgbm_features", fit_predict, X, X_test, y, "LightGBM + engineered features (flag count, ratios, frequency encodings)")
