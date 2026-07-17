# Inference of PM2.5 at Held-Out Locations: XGBoost vs. AGNN

Code for my MSc thesis on hourly PM2.5 inferencing at unmonitored locations, using reference stations, low-cost sensor (LCS) data, and ERA5 / ERA5-Land / EAC4 variables (2021–2024). Both models are evaluated under a leave-one-station-out (LOSO) protocol over 31 stations: train on 2021–2022, validate on 2023, test on the held-out station's 2024 hours.

## Repository layout

**Data preparation** 

| File | What it does |
|---|---|
| `eda_and_knn_imputation.ipynb` | EDA, cleaning of `ref_pm25` / `lcs_median_pm25`, and multivariate KNN imputation (k=5) per station → `data_imputed/` |
| `feature_engineering.ipynb` | Engineered features: derived meteorology, cyclic time, neighbour readings + IDW mean, lags, upwind/downwind blocks, convection–diffusion terms, LCS corrections |

**`XGB/`** XGBoost notebooks, one per subquestion

**`AGNN/`** AGNN scripts

| Subquestion | XGBoost | AGNN |
|---|---|---|
| SQ1: full model under LOSO | `01_sq1_full_model_loso.ipynb` | `01_sq1_full_model_loso.py` |
| SQ2: leave-group-out feature ablations | `02_sq2_feature_ablations.ipynb` | `02_sq2_feature_ablations.py` |
| SQ3: distance bands (remove stations within 0/15/35/60 km) | `03_sq3_distance_bands.ipynb` | `03_sq3_distance_bands.py` |
| SQ4: seasonal performance | `04_sq4_seasonal_performance.ipynb` | `04_sq4_seasonal_performance.py` |

`AGNN/agnn_model.py` holds the AGNN model code (encoder/decoder classes adapted from Phung et al., 2022, plus the LOSO fold dataset and training helpers), and `AGNN/05_agnn_results.ipynb` builds the AGNN result tables and figures from the saved CSVs.

## Outputs

Each script/notebook writes per-station metrics (RMSE, MAE, R²), predictions, hourly metrics, and summary tables to `results/tables/`. Trained AGNN model pairs are archived in `results/models/` and reloaded on a rerun, so evaluation does not retrain.

