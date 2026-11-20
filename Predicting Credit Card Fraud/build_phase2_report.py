"""Builds PHASE2_REPORT.md from artifacts on disk (no training)."""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import HERE, OOF_DIR, SEED, load
from fe_keys import rank_logit

train, test, y = load()
f5 = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
f10 = list(StratifiedKFold(10, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
res = pd.read_csv(os.path.join(HERE, "results.csv")).set_index("iteration")
L = lambda m: np.load(os.path.join(OOF_DIR, f"{m}_oof.npy"))
fold5 = lambda o: np.array([roc_auc_score(y[va], o[va]) for _, va in f5])

MODELS = [  # (key, label, plan expectation, iterations note, test source)
    ("v11_ebm", "v11 EBM (Phase 1 best single)", "", "n/a", "5-fold mean"),
    ("v15_stack", "v15 stack (Phase 1 best)", "", "n/a", ""),
    ("v16_test_1seed", "v16 GLM-boost LGB, 1 seed (pilot)", "", "mean 1374 trees", "5-fold mean"),
    ("v16_glm_lgb_glm", "v16 GLM alone (73 TE logits + base, C=0.3)", "~0.9506", "newton-cholesky", "n/a"),
    ("v16_glm_lgb", "v16 GLM-boost 4-leaf LGB, 5-fold, 3 seeds", "0.9511–0.9514", "mean 1163 (407–1920); refit 1396×3", "full refit"),
    ("v17_glm_lgb10_glm", "v17 GLM alone, 10-fold", "", "newton-cholesky", "n/a"),
    ("v17_glm_lgb10", "v17 GLM-boost 4-leaf LGB, 10-fold, 2 seeds", "0.9515–0.9516", "mean 1245 (38–3090); refit 1382×2", "full refit"),
    ("v18_glm_xgb_glm", "v18 GLM alone", "", "newton-cholesky", "n/a"),
    ("v18_glm_xgb", "v18 GLM-boost depth-2 XGB, 5-fold, 3 seeds", "0.9508–0.9512", "mean 895 (359–1709); refit 1073×3", "full refit"),
    ("v19_sparse_glm", "v19 joint sparse one-hot GLM (62,172 cols, C=0.01)", "0.9501–0.9503", "lbfgs n_iter 134–152", "full refit"),
    ("v20_ebm_digits", "v20 EBM + cents + 60 TE logits", "0.9507–0.9512", "EBM", "full refit"),
    ("v21_mlp_bag", "v21 sklearn MLP (256,128), 12 epochs, 3 seeds", "~0.9495", "12 epochs fixed", "full refit"),
    ("v22_lgb4_raw", "v22 4-leaf LGB, raw 24 + 6 cents, no TE", "~0.9478–0.9481", "5380–6424 trees", "full refit"),
    ("v24_torch_mlp", "v24 PyTorch residual MLP (GPU), 10 epochs, 3 seeds", "(not in plan)", "10 epochs fixed", "full refit"),
    ("v23_stack", "**v23 final stack**", "0.9517–0.9520+", "n/a", "LogReg C=0.05 on rank-logits"),
]
v11f, v15f = fold5(L("v11_ebm")), fold5(L("v15_stack"))
rows = []
for key, lab, exp, it, src in MODELS:
    if not os.path.exists(os.path.join(OOF_DIR, f"{key}_oof.npy")):
        continue
    o = L(key); f = fold5(o)
    base = key.replace("_glm", "") if key.endswith("_glm") and key != "v19_sparse_glm" else key
    secs = res.loc[base, "seconds"] if base in res.index and key == base else np.nan
    rows.append({"model": lab, "concat_auc": roc_auc_score(y, o), "mean_fold_auc(5f seed42)": f.mean(),
                 **{f"f{i}": f[i] for i in range(5)}, "beats_v11": f"{(f > v11f).sum()}/5", "beats_v15": f"{(f > v15f).sum()}/5",
                 "pr_auc": average_precision_score(y, o), "plan_expected": exp, "iterations": it,
                 "runtime_s": "" if pd.isna(secs) or secs == 0 else int(secs)})
T = pd.DataFrame(rows)
o17 = L("v17_glm_lgb10")
f17_10 = [roc_auc_score(y[va], o17[va]) for _, va in f10]

md = []
w = md.append
w("# Phase 2 implementation report: Kaggle Playground S6E11\n")
w("Reply to the \"Phase 2 Review & Runbook\". Everything below was run on CPU (16 cores) plus an RTX 3050 Laptop GPU (6 GB, only for v24). "
  "Heavy jobs ran **one at a time**. Base folds: `StratifiedKFold(5, shuffle=True, random_state=42)`, except v17, which uses `StratifiedKFold(10, …, 42)`. "
  "Its fold columns below are its OOF scored on the 5-fold seed-42 split, for comparability. All numbers come from OOF arrays on disk.\n")
w("## Headline\n")
w("""- **Best single model: v17 (10-fold GLM-boost 4-leaf LightGBM), CV AUC 0.95149.** The plan expected 0.9515–0.9516, so it **matched**.
- **Best final: v23 stack, nested CV AUC 0.95156** (mean over stacker splits 43/44/45). The plan expected 0.9517–0.9520+, so it came in **~0.0001–0.0004 below**.
- Phase 1 best was 0.94941 (v15), so Phase 2 added **+0.00215**. v4 baseline 0.94737 → **+0.00419** total.
- **Every** Phase 2 base model beats v11 and v15 in 5/5 folds, except v21 (sklearn MLP), v22 (raw 4-leaf) and v24 (torch MLP).
- **The stack adds only +0.00007 over v17 alone.** The hill-climb combiner chooses **v17 alone** (weight 1.0). Adding v20 or v24 changed the stack by ≤ 0.00001.
- **Not submitted to the public LB yet** (the user will submit `submissions/v23_stack.csv`).
""")
w("\n## 1. Master score table\n")
w(T.round(5).to_markdown(index=False) + "\n")
w(f"\nv17's own 10-fold AUCs: {np.round(f17_10, 5).tolist()} (mean {np.mean(f17_10):.5f}, std {np.std(f17_10):.5f}).\n")
w("\n'GLM alone' rows = OOF of the newton-cholesky GLM that gives the init_score (averaged over the same encoding seeds). v16 and v18 share the same encodings and GLM, so their 'GLM alone' rows are identical by construction. The v23 row's concat AUC is its nested OOF on stacker split 43 (the headline 0.95156 is the mean over splits 43/44/45). Runtime = wall clock for CV + refit.\n")

w("""
## 2. Claims in the runbook vs what we measured

| Runbook claim | Measured | Verdict |
|---|---|---|
| Digit keys carry signal (e.g. `amount_usd` lead2 ≈ 0.583, distance cents ≈ 0.523) | fold-0 single-key AUC: amount lead2 **0.588**, amount int 0.580, hours tenths 0.576, hours cents 0.566, distance cents **0.523**, balance int 0.563 (raw values 0.541 / 0.558 / 0.507 / 0.502) | ✅ confirmed |
| GLM alone ≈ 0.9506 | v16 GLM 0.95063, v17 GLM (10-fold) 0.95087 | ✅ |
| v16 5-fold 3-seed ≈ 0.9511–0.9514 | **0.95121** (1-seed pilot 0.95115) | ✅ |
| v17 10-fold ≈ 0.9515–0.9516 | **0.95149** | ✅ |
| v18 depth-2 XGB ≈ 0.9508–0.9512 | **0.95111** | ✅ |
| v19 sparse GLM ≈ 0.9501–0.9503 | **0.95011** (62,172 cols, lbfgs converged in 134–152 iterations, ~45 s/fold) | ✅ |
| v20 EBM + digits ≈ 0.9507–0.9512 | **0.94995** | ❌ below (+0.0007 over v11 EBM only) |
| v21 sklearn MLP ≈ 0.9495 | **0.94259**, unstable folds 0.937–0.949 | ❌ well below |
| v22 raw 4-leaf ≈ 0.9478–0.9481, earns 0.08–0.09 stack weight | **0.94827**. LogReg-stack coef 0.127, hill weight 0 | ✅ score / ~ weight |
| Final stack 0.9517–0.9520+ | **0.95156** | ❌ slightly below |
| Zero-fraud zones +0.000036, 5/5 folds | **+0.000019 to +0.000021, 5/5 folds** (thr 13.1 in every fold, 0 train frauds in zone) | ✅ direction / smaller size |
""")

# stacking details
w("\n## 3. Stacking details (`v23_stack.py`)\n")
w("""- **Inputs:** every model's OOF probabilities → global `rank_logit` (`logit(rank/(n+1))`). Test → `rank_logit` of the **full-data refit** prediction where one exists (`_full.npy`), else the 5-fold mean (v11, v10). This follows the runbook's fix for the scale mismatch flagged in §1.4.
- **Combiners**, each scored by nested CV on three fresh splits (`StratifiedKFold(5, shuffle, seed s)`, s ∈ {43, 44, 45}), with predictions ranked within each validation fold:

| Combiner | nested AUC (43 / 44 / 45) | mean |
|---|---|---|
| LogReg C=0.01 (newton-cholesky) | 0.95150 / 0.95149 / 0.95147 | 0.95149 |
| **LogReg C=0.05** | 0.95157 / 0.95157 / 0.95154 | **0.95156** ← chosen |
| Hill climb (non-negative, rank01, ≤ 80 steps) | 0.95149 / 0.95149 / 0.95149 | 0.95149 (picks v17 only) |
| mix01 (50/50 rank of LR C=0.01 + hill) | 0.95155 / 0.95154 / 0.95154 | 0.95154 |

- **Members (8):** v16, v17, v18, v19, v21, v22, v11, v10. Full-data LogReg C=0.01 coefficients: v17 0.655, v16 0.287, v11 0.219, v22 0.127, v18 0.115, v19 0.042, v21 0.029, v10 −0.111.
- **Variants tried, not kept:** + v20 → 0.95155. + v24 → 0.95155 (v24 coef 0.004).
- **Zero-fraud zones:** z = (`merchant_risk_score` < thr) | (`velocity_score` ≤ 4.1 & `merchant_risk_score` ≤ 20). thr = minimum fraud `merchant_risk_score` in the training folds (13.1 in all 5 folds). Validated on the seed-42 folds using the nested (split-43) stack OOF: gain +0.000016 to +0.000030 per fold, 5/5 positive. Applied to **8,076 test rows** (pushed below all other rows, order kept inside the zone).
""")

# correlation
cm = ["v16_glm_lgb", "v17_glm_lgb10", "v18_glm_xgb", "v19_sparse_glm", "v20_ebm_digits", "v21_mlp_bag", "v22_lgb4_raw", "v24_torch_mlp", "v11_ebm", "v10_spline_logreg"]
C = spearmanr(np.column_stack([L(m) for m in cm])).correlation
w("\n## 4. Diversity (Spearman of OOF predictions)\n\n" + pd.DataFrame(C, index=cm, columns=[c.split("_")[0] for c in cm]).round(3).to_markdown() + "\n")
w("\nThe GLM-boost family (v16/v17/v18) is 0.99+ correlated within itself. The most different useful members are v22 (raw, no TE), v11 (EBM) and v10 (splines). The two MLPs are not different *enough* to earn weight.\n")

w("""
## 5. Implementation details and differences from the runbook

1. **`fe_keys.py`:** same 73 keys (24 exact, 36 digit, 13 pair), `np.rint(x*100)` + integer division, factorised on train+test, cached to `keys_cache.npz` (11 s to build). Nested `np.bincount` TE, smooth=20, inner 5-fold. **5.7 s per outer fold** for all 73 keys.
2. **GLM:** `StandardScaler(logit(E))` on 73 TE logits + base matrix (13 standardised numerics with log1p on amount/balance/hours, 6 flags, 35 one-hot cats), `LogisticRegression(C=0.3, newton-cholesky, tol=1e-7)`.
3. **Trees:** raw 24 + 6 cents + 60 TE rates + 60 key counts = 150 columns. LGB `num_leaves=4, lr 0.02, min_child_samples 100, feature_fraction 0.5, bagging 0.8, lambda_l2 1`, early stopping 300 on validation AUC with `init_score`. XGB `max_depth=2, min_child_weight 20, subsample 0.8, colsample 0.5, lambda 2`, `base_margin`.
4. **Seeds:** (teseed, model seed) = (0,42), (1,137), (2,2026) for v16/v18, and the first two for v17. Fold predictions = mean of margins over seeds. Test = fold-mean, plus a **full refit** at round(mean_iters × 1.20) (5-fold) or × 1.11 (10-fold).
5. **v19:** as specified (numerics, flags, cats, B-splines n_knots=8, one-hot `min_frequency=5` on the 13 numeric exact keys and the 36 digit keys, + 60 standardised TE logits), `C=0.01, lbfgs, tol 1e-6`.
6. **v20 / v21 / v22:** as specified. v20 was first killed by an out-of-memory error while another job ran (now strictly sequential), and the rerun matched the first run's folds exactly.
7. **v24 (extra, not in the runbook):** PyTorch 2.11 + CUDA 12.8 on the RTX 3050 (installed in a venv on E:). Inputs: quantile-normal numerics + cents, one-hots, clipped standardised logits of all 73 TE keys. Net: Linear(→256) + 2 residual blocks (LayerNorm, Linear, SiLU, Dropout 0.2, Linear), AdamW 2e-3 / wd 1e-4, OneCycle, batch 4096, **fixed 10 epochs**, 3 seeds. ~6 s per seed per fold. CV 0.94863, stable across seeds (±0.0005).
8. **Original 20k dataset:** not downloaded, following §1.5.
9. **GPU:** XGBoost (`device="cuda"`) and CatBoost (`task_type="GPU"`) work on this machine. LightGBM's pip build is CPU-only. v18 was run on CPU (before the GPU check).
""")

w("""
## 6. Saved artifacts

- `oof/<name>_oof.npy` = OOF probabilities. `<name>_test.npy` = 5-fold mean test probabilities. `<name>_full.npy` = full-refit test probabilities (v16–v22, v24). `<name>_glm_oof.npy` = GLM-alone OOF (v16–v18).
- `submissions/v23_stack.csv` = final (rank01 of the LogReg C=0.05 decision function, zones applied). Also `v16_glm_lgb.csv`, `v17_glm_lgb10.csv`, `v18_glm_xgb.csv`, `v19…v24`.
- Code: `fe_keys.py`, `v16_glm_boost.py` (v16/v17/v18 via CLI args), `p2_models.py` (v19–v22), `v24_torch_mlp.py`, `v23_stack.py`.

## 7. Open questions for the next phase

1. The stack is +0.00007 over v17, and hill climbing keeps v17 alone. Is the remaining headroom mainly **more v17-style bagging** (more encoding seeds, 10-fold for v16/v18), or a genuinely new feature family?
2. v20 (EBM + digits) and the MLPs fell short of the runbook's estimates. Are there settings (e.g. EBM with fewer TE columns, MLP with categorical embeddings or a GLM offset) that you've seen work on this data?
3. Is the 10-fold gain (v16 0.95121 → v17 0.95149) worth extending to **20-fold** for the GLM-boost, or will it plateau?
4. Any other synthesizer artifacts worth encoding (e.g. digit keys for `customer_age`/`card_age_months`, cross-column digit pairs)?
""")
open(os.path.join(HERE, "PHASE2_REPORT.md"), "w", encoding="utf-8").write("\n".join(md))
print("written")
