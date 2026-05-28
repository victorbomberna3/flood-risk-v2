# Flood Risk Prediction v2 — Two-Stage GBM Cascade + U-Net

**ESCP Hackathon 2026 · Group 2 · Train: Severn → Test: Northumbria**

---

## Results summary

| Model | Severn val QWK | Northumbria QWK | Northumbria Macro F1 |
|-------|---------------|-----------------|----------------------|
| v1 XGBoost baseline | — | — | 0.3121 |
| U-Net Run 2 (weighted CE) | 0.37 (F1-based) | 0.1530 | 0.2030 |
| U-Net Run 5 (focal + ordinal MSE) | **0.5578** | 0.1249 | 0.2067 |
| **Two-stage GBM cascade** | **0.7576** (5-fold CV) | **0.6865** (stress test) | — |

**Primary metric: QWK (Quadratic Weighted Kappa)** — penalises large ordinal mistakes quadratically.
Predicting class 0 when truth is 3 costs (3−0)²=9×; predicting class 4 when truth is 3 costs (4−3)²=1×.

---

## What we built and what we learned

### Stage 1 — U-Net (spatial deep learning)

We started by replacing the v1 XGBoost tabular approach with a multi-task U-Net that keeps pixels as 2D rasters, allowing the model to learn spatial context (neighbouring pixels are correlated — Tobler's law).

**What worked:** Severn validation QWK improved dramatically run-by-run (0.37 → 0.56), especially after switching to focal loss and adding an ordinal MSE auxiliary loss term.

**What failed:** Cross-region transfer. Northumbria QWK stayed at 0.12–0.15 regardless of how well the model performed on Severn. The U-Net was memorising Severn's spatial topology (river network shape, patch geometry) rather than learning transferable hydrological rules.

Key finding: **QWK < Macro F1 on Northumbria** — the model was not just wrong, it was making large ordinal errors (predicting "No Risk" on flood-prone pixels). Class 0 F1 ≈ 0.00 across all runs.

### Stage 2 — Two-stage GBM cascade

Motivated by the U-Net's cross-region failure, we implemented a two-stage gradient boosting pipeline:

**Stage A** — LightGBM binary classifier: *is there any flood risk at all?*
This dedicated binary model directly solves the class 0 collapse problem. A pixel predicted as "No Risk" by Stage A is never passed to Stage B.

**Stage B** — CatBoost + LightGBM regressor ensemble, one model per depth target (0.2m–1.2m):
CatBoost MultiClass with `eval_metric=WKappa` and balanced class weights. LightGBM regressor with `OptimizedRounder` (Nelder-Mead boundary tuning) converted to soft probabilities. Both models ensembled with sweep-optimised weights.

**Why it generalises:** GBMs learn rules like *"if HAND < 2m AND TWI > 12 → flood risk class 3"*. These rules are based on hydrological physics and transfer to Northumbria. The U-Net learned *what Severn's river network looks like*, which doesn't transfer.

**5-fold spatial block cross-validation:** 20km blocks prevent spatial autocorrelation leakage. Stage A and Stage B folds are aligned so no block appears in different fold indices across stages.

**tau_a threshold tuning:** Stage A's decision threshold is grid-searched from 0.05–0.95 to maximise mean QWK across all 5 depth targets on OOF predictions.

---

## Quick start (Colab)

```python
# Step 1 — Mount Drive, clone repo, install dependencies
from google.colab import drive
drive.mount('/content/drive')

import os, subprocess
if not os.path.exists('/content/flood-risk-v2'):
    subprocess.run(['git', 'clone',
                    'https://github.com/victorbomberna3/flood-risk-v2.git',
                    '/content/flood-risk-v2'])
else:
    subprocess.run(['git', '-C', '/content/flood-risk-v2', 'reset', '--hard', 'origin/main'])
    subprocess.run(['git', '-C', '/content/flood-risk-v2', 'pull'])

os.chdir('/content/flood-risk-v2')
!pip install -q -r requirements.txt   # includes catboost, optuna, lightgbm
```

### Run the GBM cascade (recommended — ~2h total)

```bash
# Step 2 — Build raster cache (once only, ~10 min)
!python scripts/01_build_rasters.py --config configs/default.yaml

# Step 5 — Train cascade
!python scripts/04_train_cascade.py \
    --config configs/default.yaml \
    --catboost-device GPU \
    --lgb-device cpu \
    --max-pixels 5000000

# Step 6 — Predict on Northumbria
!python scripts/05_predict_cascade.py --config configs/default.yaml
```

### Run the U-Net (optional — ~2-3h)

```bash
# Step 3 — Train U-Net (after building rasters above)
!python scripts/02_train.py --config configs/default.yaml

# Step 4 — Predict on Northumbria
!python scripts/03_predict.py --config configs/default.yaml
```

The two pipelines are **independent after Step 2** — raster cache is shared, everything else is separate.

---

## Project structure

```
flood-risk-v2/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml              # All hyperparameters (both pipelines)
│
├── cascade/                      # Two-stage GBM cascade package
│   ├── __init__.py
│   ├── cv.py                     # 20km spatial block GroupKFold
│   ├── models.py                 # Stage A (LightGBM) + Stage B (CatBoost + LGB)
│   ├── rounder.py                # OptimizedRounder + soft_probabilities
│   └── metrics.py                # QWK + per-class F1 evaluation
│
├── scripts/
│   ├── 01_build_rasters.py       # Build raster NPZ cache from .nc files
│   ├── 02_train.py               # Train U-Net on Severn
│   ├── 03_predict.py             # U-Net inference on Northumbria
│   ├── 04_train_cascade.py       # Train GBM cascade on Severn (5-fold CV)
│   └── 05_predict_cascade.py     # Cascade inference on Northumbria
│
├── colab_runner.ipynb            # Colab notebook (Steps 1-7)
│
├── data_io.py                    # Load .nc → raster tensors + NPZ cache
├── features.py                   # Terrain + weather feature engineering
├── normalize.py                  # Per-region robust normalisation
├── dataset.py                    # PyTorch Dataset + spatial block CV
├── model.py                      # U-Net multi-task architecture
├── losses.py                     # Focal loss + ordinal MSE auxiliary term
├── trainer.py                    # Training loop (QWK monitoring)
├── predict.py                    # Sliding-window inference + TTA
├── evaluate.py                   # QWK + Macro F1 + confusion matrices
└── utils.py                      # Config loading, seeding, device helpers
```

---

## Data expected

Place the four `.nc` files in `/content/drive/MyDrive/Hackathon II Group 2/`:

```
flood_risk_terrain_severn.nc
flood_risk_terrain_northumbria.nc
era5_land_severn.nc
era5_land_northumbria.nc
```

---

## Engineered features (29 total)

### Terrain (21 features)
All relative — no absolute elevation or coordinates, which encode regional identity and cause domain shift.

| Feature | Description |
|---------|-------------|
| `dtm_rel_3/11/51` | Elevation relative to local mean (60m / 220m / 1km windows) |
| `slope`, `aspect_sin`, `aspect_cos` | Gradient magnitude and direction (circular encoded) |
| `curvature` | Laplacian — negative = concave = water pools |
| `roughness_3/11` | Local elevation std (texture) |
| `log_flow_acc` | Upstream catchment area (log-scaled) |
| `flow_acc_lag_3x3/11x11` | Neighbourhood mean of log_flow_acc |
| `flow_acc_anomaly` | Local convergence hotspot: pixel vs neighbourhood |
| `twi` | Topographic Wetness Index = ln(flow_acc / tan(slope)) |
| `hand` | Height Above Nearest Drainage — strongest single predictor |
| `log_dist_channel` | Log distance to nearest channel |
| `near_water` | Binary: within water body extent |
| `waw` | Water/wetness category (0–5 ordinal) |
| `imd` | Imperviousness 0–100% |
| `flow_dir_sin`, `flow_dir_cos` | Flow direction (circular encoded) |

### Weather (8 features)
ERA5-Land 10-year climatology aggregated to static statistics — appropriate for annual exceedance probability targets.

| Feature | Description |
|---------|-------------|
| `precip_roll1d/7d/30d_max` | Peak 1-day / 7-day / 30-day rainfall |
| `precip_p95_7d` | 95th percentile of 7-day rolling rainfall |
| `precip_intensity` | max_1d / mean_30d — region-invariant extremeness |
| `runoff_roll7d_max` | Peak 7-day surface runoff |
| `runoff_intensity` | Runoff extremeness ratio |
| `soil_moist_max` | Maximum soil moisture (saturation proxy) |

---

## Model details

### U-Net
- Architecture: 4-level encoder-decoder with skip connections
- Input: `(29, 256, 256)` patch → 5 output heads `(5, 256, 256)`
- Loss: Focal loss (γ=2.0) + ordinal MSE auxiliary term (weight=0.5)
- Validation: Spatial block split, checkpoint on best val QWK
- Best val QWK: **0.5578** (epoch 58/60)

### Two-stage GBM cascade
- Stage A: LightGBM binary, Optuna HPO (25 trials), 5-fold spatial CV
- Stage B: CatBoost MultiClass (`eval_metric=WKappa`) + LightGBM regressor with `OptimizedRounder`, ensembled by QWK-optimised weight sweep
- Threshold: tau_a grid-searched 0.05–0.95 to maximise OOF mean QWK
- CV QWK: **0.7576** | Northumbria stress-test QWK: **0.6865**

---

## Why QWK over Macro F1

The teacher's instruction: *"predicting 4 when 3 is true ≠ predicting 1"* — ordinal distance matters.

Macro F1 treats all misclassifications equally. QWK penalises by the square of the ordinal distance:

| True | Pred | Macro F1 penalty | QWK penalty |
|------|------|-----------------|-------------|
| 3 | 4 | 1 (missed) | (4−3)²/16 = 0.06 |
| 3 | 1 | 1 (missed) | (3−1)²/16 = 0.25 |
| 3 | 0 | 1 (missed) | (3−0)²/16 = 0.56 |

For a public safety application — where predicting "No Risk" on a flood-prone area could cost lives — QWK is the correct metric.

---

## Key differences vs v1

| Aspect | v1 (XGBoost) | v2 U-Net | v2 Cascade |
|--------|-------------|----------|------------|
| Data layout | Tabular rows | 2D rasters | Tabular (pixel-level) |
| Spatial context | None | Convolution (5km receptive field) | Spatial lag features |
| Cross-region transfer | Poor | Poor (memorises Severn) | Good (physics-based rules) |
| Class 0 handling | Single model | Fails (F1≈0.00) | Dedicated binary stage |
| Ordinal awareness | None | Ordinal MSE loss | CatBoost WKappa eval metric |
| Validation | Random split | Spatial block split | 5-fold spatial block CV |
| Primary metric | Macro F1 | QWK | QWK |
| Northumbria QWK | — | 0.1249 | **0.6865** |
