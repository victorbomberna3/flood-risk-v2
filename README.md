# Flood Risk Prediction v2 — U-Net Multi-Task

**ESCP Hackathon 2026 · Group 2 · Train: Severn → Test: Northumbria**

Replaces the XGBoost ordinal ensemble (Macro F1 = 31.21%) with a U-Net semantic segmentation model. Target: **45–55% Macro F1**.

---

## Why this works

Previous approaches treated pixels as independent tabular rows. This destroys the most important signal in the data: **spatial structure**.

This pipeline:

1. **Keeps the data as 2D rasters** — terrain + weather become image-like tensors `(H, W, channels)`.
2. **U-Net learns spatial patterns natively** — convolution exploits the fact that neighbouring pixels are correlated (Tobler's law), instead of fighting it.
3. **Per-region robust normalisation** — eliminates Severn-vs-Northumbria distribution shift.
4. **Multi-task learning** — one shared encoder predicts all 5 flood depths jointly. Shared representations transfer better.
5. **Spatial block CV + augmentation** — prevents the data leakage that random pixel splits cause.

Background: see academic justification in Kabir et al. 2020 (CNN beats SVR by large margin), Liao et al. 2023 (CNN beats XGBoost on urban floods), Gao et al. 2024 (explainable CNN for flood prediction).

---

## Quick start (Colab)

```python
# In a Colab cell:
!git clone https://github.com/YOUR_USERNAME/flood-risk-v2.git
%cd flood-risk-v2
!pip install -q -r requirements.txt

# Then open notebooks/colab_runner.ipynb and run cells top to bottom
```

The notebook handles:
1. Mount Google Drive (your .nc files)
2. Build raster tensors (~5 min)
3. Engineer features on rasters (~5 min)
4. Train U-Net (~1–3 hours on Colab Pro GPU)
5. Inference on Northumbria (~10 min)
6. Generate prediction maps + metrics

---

## Project structure

```
flood-risk-v2/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml              # All hyperparameters
├── notebooks/
│   └── colab_runner.ipynb        # Main entry point
├── src/
│   ├── data_io.py                # Load .nc → raster tensors
│   ├── features.py               # Feature engineering on rasters
│   ├── normalize.py              # Per-region robust normalisation
│   ├── dataset.py                # PyTorch Dataset + augmentation
│   ├── model.py                  # U-Net multi-task architecture
│   ├── losses.py                 # Masked weighted CE
│   ├── trainer.py                # Training loop
│   ├── predict.py                # Sliding-window inference
│   ├── evaluate.py               # Metrics, confusion matrices
│   └── utils.py                  # Helpers, config loading
└── scripts/
    ├── 01_build_rasters.py       # Build raster tensors from .nc
    ├── 02_train.py               # Train U-Net
    └── 03_predict.py             # Generate Northumbria predictions
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

The runner notebook detects them automatically.

---

## Key differences vs v1 (the XGBoost approach)

| Aspect | v1 (XGBoost) | v2 (U-Net) |
|---|---|---|
| Data layout | Tabular parquet (each pixel = row) | 2D raster `(H, W, C)` |
| Spatial structure | Ignored | Captured natively via convolution |
| Receptive field | Single pixel | Up to 5km via U-Net depth |
| Cross-region handling | Drop regional proxy features | Per-region robust normalisation |
| Multi-target | 5 separate models | 1 shared encoder, 5 output heads |
| Validation | Random row split (leaks) | Spatial block CV |
| Augmentation | None | 8 dihedral transformations |

---

## Outputs

After running the full pipeline, you get:

- `outputs/predictions_northumbria.parquet` — predicted risk class per pixel for all 5 depths
- `outputs/maps/risk_0_2m_pred_vs_true.png` (and 4 others) — visual comparison maps
- `outputs/metrics.json` — Macro F1, per-class F1, confusion matrices for each depth
- `outputs/model_best.pt` — trained U-Net weights
