"""
SQ2: leave group out ablations (train and evaluate)

Trains the 9 AGNN ablation variants per held out station, evaluates each on the
held out station's 2024 hours, and saves both the models and the result tables.

Variants (GEO and GEO+Time are skipped because lat/lon and year are not AGNN
inputs): full, no_lcs, no_era5, no_eac4, no_era5_eac4, no_era5_eac4_lcs,
no_network, no_time, raw. Removing a node group changes the graph encoder
input, removing a climate group changes the decoder input. The raw model adds
lcs_median_pm25 back, exactly like the XGBoost raw model.

Saved to results/models/ : sq2_<variant>_<station>_{stdgi,decoder}.pt
Saved to results/tables/ : per variant sq2_<variant>_per_station_metrics.csv,
sq2_<variant>_predictions.csv, sq2_<variant>_hourly_metrics.csv, plus
sq2_summary.csv and sq2_wilcoxon_vs_full.csv (paired Wilcoxon of every variant
against the full model, BH corrected).

The expensive per fold feature rebuild is done once per fold and shared by all
variants. Within one fresh fold, variants with identical node features share
one trained encoder (the encoder never sees climate features), the shared
encoder is saved under every variant's own file name. Variant model pairs that
already exist are loaded rather than retrained. 
"""

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import wilcoxon

from agnn_model import (
    Attention_STDGI, Local_Global_Decoder, FoldDataset, EarlyStopping,
    seed_everything, load_model, train_stdgi_one_epoch, train_decoder_one_epoch,
    decoder_val_loss, predict_decoder, save_model,
)

# location of imputed station CSVs and the station coordinates 
DATA_DIR = r"C:\Users\Storm Anderson\Documents\UVA\Inferencing\XGboost\data_imputed"
COORDS_CSV = r"C:\Users\Storm Anderson\Documents\UVA\Inferencing\XGboost\station_cell_map.csv"

# per fold models are archived here (loaded on a rerun so evaluation does not retrain) all evaluation result CSVs go to results/tables
MODELS_DIR = os.path.join("results", "models")
OUT_TABLES = os.path.join("results", "tables")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUT_TABLES, exist_ok=True)

# time split, same protocol as the XGBoost notebooks
TRAIN_YEAR_MAX = 2022   # training years are 2021 and 2022
VAL_YEAR = 2023         # 2023 is only used for early stopping of the decoder
TEST_YEAR = 2024        # 2024 is the held out test year
FILL_YEAR_MAX = 2022    # fill statistics come from these years only (like feature_engineering)

USE_GPU = True
DEVICE = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
SEED = 42

# neighbour geometry settings, identical to the XGBoost notebooks
K_NEAREST = 5
K_QUAD = 8
IDW_POWER = 2.0
LAG_HOURS = [1, 3, 6, 24]

# AGNN hyperparameters
SEQUENCE_LENGTH = 12
BATCH_SIZE = 32
OUTPUT_STDGI = 60
EN_HID1 = 128
EN_HID2 = 128
DIS_HID = 16
CNN_HID_DIM = 64
FC_HID_DIM = 128
N_LAYERS_RNN = 1
RNN_TYPE = "GRU"
LR_STDGI = 1e-3
LR_DECODER = 1e-3
NUM_EPOCHS_STDGI = 20
NUM_EPOCHS_DECODER = 30
PATIENCE = 5
STDGI_NOISE_MIN = 0.4
STDGI_NOISE_MAX = 0.7

# preprocessing settings
CLIP_PCT = 95.0         # upper clip for feature channels, like the original repo
Y_SCALE_PCT = 99.9      # denominator percentile for the (never clipped) target scaling
VAL_STRIDE_HOURS = 24   # validation uses one window per day per pool station
DIST_FLOOR_KM = 0.1

pd.set_option("display.max_columns", None)
print("device:", DEVICE)


# Feature groups, AGNN feature roles and the ablation variants


NETWORK_FEATURES = [
    "nb_1", "nb_2", "nb_3", "nb_4", "nb_5",
    "idw_neighbour",
    "nb1_lag_1h", "nb1_lag_3h", "nb1_lag_6h", "nb1_lag_24h",
    "nb1_24h_mean", "nb1_24h_std",
    "pm_upwind", "pm_downwind", "n_upwind", "n_downwind",
    "pm_gradient", "pm_advection", "pm_diffusion",
    "pm_upwind_isna", "pm_downwind_isna",
]

LCS_FEATURES = ["lcs_adv", "lcs_adv_coef", "f_rh", "lcs_x_frh", "lcs_rh_corrected"]

ERA5_FEATURES = [
    "t2m", "rh", "wind_speed", "wind_dir_sin", "wind_dir_cos",
    "sp", "ssr", "strd", "tp", "slhf", "sshf",
    "blh", "low_blh", "ventilation_coef", "stagnant", "stagnation_hours",
]

EAC4_FEATURES = [
    "aod550", "bcaod550", "duaod550", "ssaod550", "gtco3", "pm2p5",
    "co_1000hPa", "go3_1000hPa", "no2_1000hPa", "so2_1000hPa",
    "co_700hPa", "go3_700hPa", "no2_700hPa", "so2_700hPa",
    "co_500hPa", "go3_500hPa", "no2_500hPa", "so2_500hPa",
]

TIME_SINCOS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "dow_sin", "dow_cos"]

NODE_FEATURES = ["ref_pm25"] + NETWORK_FEATURES + LCS_FEATURES + EAC4_FEATURES
CLIMATE_FEATURES = ERA5_FEATURES + TIME_SINCOS

# the raw model additionally needs the Veli corrected LCS median as the one allowed semi raw LCS input
MASTER_NODE_FEATURES = NODE_FEATURES + ["lcs_median_pm25"]

NO_CLIP_COLS = [
    "low_blh", "stagnant", "n_upwind", "n_downwind",
    "pm_upwind_isna", "pm_downwind_isna",
    "wind_dir_sin", "wind_dir_cos",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "dow_sin", "dow_cos",
]

# raw meteorology for the raw model: ERA5 without the 4 boundary layer derivations
RAW_ERA5 = [
    "t2m", "rh", "wind_speed", "wind_dir_sin", "wind_dir_cos",
    "sp", "ssr", "strd", "tp", "slhf", "sshf", "blh",
]


def node_without(groups_to_drop):
    dropped = []
    for group in groups_to_drop:
        for col in group:
            dropped.append(col)
    kept = []
    for col in NODE_FEATURES:
        if col not in dropped:
            kept.append(col)
    return kept


# every variant is a pair (node feature list, climate feature list)
ABLATIONS = {}
ABLATIONS["full"] = (list(NODE_FEATURES), list(CLIMATE_FEATURES))
ABLATIONS["no_lcs"] = (node_without([LCS_FEATURES]), list(CLIMATE_FEATURES))
ABLATIONS["no_era5"] = (list(NODE_FEATURES), list(TIME_SINCOS))
ABLATIONS["no_eac4"] = (node_without([EAC4_FEATURES]), list(CLIMATE_FEATURES))
ABLATIONS["no_era5_eac4"] = (node_without([EAC4_FEATURES]), list(TIME_SINCOS))
ABLATIONS["no_era5_eac4_lcs"] = (node_without([EAC4_FEATURES, LCS_FEATURES]), list(TIME_SINCOS))
ABLATIONS["no_network"] = (node_without([NETWORK_FEATURES]), list(CLIMATE_FEATURES))
ABLATIONS["no_time"] = (list(NODE_FEATURES), list(ERA5_FEATURES))
ABLATIONS["raw"] = (["ref_pm25", "nb_1", "nb_2", "nb_3", "nb_4", "nb_5", "lcs_median_pm25"] + EAC4_FEATURES,
                    RAW_ERA5 + TIME_SINCOS)

# plotting order and labels, same as the XGBoost notebook minus the two GEO
# variants (lat/lon are not AGNN inputs, so there is nothing to remove)
ABL_ORDER = ["no_lcs", "no_era5", "no_eac4", "no_era5_eac4", "no_era5_eac4_lcs",
             "no_network", "no_time", "raw"]
ABL_LABEL = {
    "no_lcs": "LCS",
    "no_era5": "ERA5",
    "no_eac4": "EAC4",
    "no_era5_eac4": "ERA5+EAC4",
    "no_era5_eac4_lcs": "ERA5+EAC4+LCS",
    "no_network": "Neighbours",
    "no_time": "Time",
    "raw": "Engineered features",
}

for name in ABLATIONS:
    node_names, clim_names = ABLATIONS[name]
    print("%-18s %2d node + %2d climate features" % (name, len(node_names), len(clim_names)))


# Load the data (identical to the XGBoost notebooks)

t0 = time.time()
station_folders = sorted(os.listdir(DATA_DIR))

stations = {}
for folder in station_folders:
    folder_path = os.path.join(DATA_DIR, folder)
    if not os.path.isdir(folder_path):
        continue  # skips loose files 
    df = pd.read_csv(os.path.join(folder_path, folder + ".csv"))
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df["station_id"] = folder
    stations[folder] = df

STATION_IDS = list(stations.keys())

coords = pd.read_csv(COORDS_CSV)
coords = coords[coords["station_id"].isin(STATION_IDS)].set_index("station_id")[["lat", "lon"]]

for name in STATION_IDS:
    stations[name]["lat"] = coords.loc[name, "lat"]
    stations[name]["lon"] = coords.loc[name, "lon"]

panel = pd.DataFrame()
for name in STATION_IDS:
    df = stations[name]
    panel[name] = pd.Series(df["ref_pm25"].values, index=df["datetime"])
panel = panel.sort_index()

N_HOURS = len(panel)
YEAR_VEC = stations[STATION_IDS[0]]["year"].to_numpy()
TRAIN_ROWS = YEAR_VEC <= TRAIN_YEAR_MAX

TRAIN_T = [int(t) for t in np.where(YEAR_VEC <= TRAIN_YEAR_MAX)[0] if t >= SEQUENCE_LENGTH - 1]
VAL_T = [int(t) for t in np.where(YEAR_VEC == VAL_YEAR)[0][::VAL_STRIDE_HOURS]]
# every 2024 hour is scored, the 12 hour input windows of the first hours simply reach back into late 2023
TEST_T = [int(t) for t in np.where(YEAR_VEC == TEST_YEAR)[0]]

print("stations:", len(stations), "| panel shape:", panel.shape,
      "| loaded in", round(time.time() - t0, 1), "s")


# Helper functions (geometry and rebuild identical to the XGBoost notebooks)

def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(a))


def bearing_rad(lat1, lon1, lat2, lon2):
    # compass bearing from point 1 to point 2, in radians from north, clockwise
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(phi2)
    y = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlon)
    return np.mod(np.arctan2(x, y), 2.0 * np.pi)


def angle_diff(a, b):
    # shortest signed angle a minus b, in [-pi, pi]
    return np.mod(a - b + np.pi, 2.0 * np.pi) - np.pi


def nearest_neighbours(target_id, available_ids, k):
    # the k nearest stations to target_id, chosen only from available_ids, never itself
    lat0 = coords.loc[target_id, "lat"]
    lon0 = coords.loc[target_id, "lon"]
    rows = []
    for other in available_ids:
        if other == target_id:
            continue
        lat1 = coords.loc[other, "lat"]
        lon1 = coords.loc[other, "lon"]
        rows.append({
            "id": other,
            "lat": lat1,
            "lon": lon1,
            "dist_km": float(haversine_km(lon0, lat0, lon1, lat1)),
            "bearing_rad": float(bearing_rad(lat0, lon0, lat1, lon1)),
        })
    rows.sort(key=lambda r: r["dist_km"])
    return rows[:k]


def rebuild_network_features(station_id, df_station, available_ids):
    # identical to the XGBoost notebooks
    df = df_station.copy()
    n = len(df)
    meta5 = nearest_neighbours(station_id, available_ids, K_NEAREST)
    meta8 = nearest_neighbours(station_id, available_ids, K_QUAD)

    for i in range(K_NEAREST):
        if i < len(meta5):
            df["nb_" + str(i + 1)] = panel[meta5[i]["id"]].reindex(df["datetime"]).values
        else:
            df["nb_" + str(i + 1)] = np.nan

    num = np.zeros(n)
    den = np.zeros(n)
    for i in range(len(meta5)):
        d = max(meta5[i]["dist_km"], 0.1)
        w = 1.0 / d ** IDW_POWER
        vals = df["nb_" + str(i + 1)].values
        ok = ~np.isnan(vals)
        num = num + np.where(ok, vals * w, 0.0)
        den = den + np.where(ok, w, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        df["idw_neighbour"] = np.where(den > 0, num / den, np.nan)

    if len(meta5) > 0:
        nb1 = panel[meta5[0]["id"]]
        for h in LAG_HOURS:
            df["nb1_lag_" + str(h) + "h"] = nb1.reindex(df["datetime"] - pd.Timedelta(hours=h)).values
        roll_mean = nb1.rolling("24h", closed="left", min_periods=12).mean()
        roll_std = nb1.rolling("24h", closed="left", min_periods=12).std()
        df["nb1_24h_mean"] = roll_mean.reindex(df["datetime"]).values
        df["nb1_24h_std"] = roll_std.reindex(df["datetime"]).values
    else:
        for h in LAG_HOURS:
            df["nb1_lag_" + str(h) + "h"] = np.nan
        df["nb1_24h_mean"] = np.nan
        df["nb1_24h_std"] = np.nan

    cone_half_rad = np.radians(60.0)
    theta_to = np.mod(np.arctan2(df["u10"].values, df["v10"].values), 2.0 * np.pi)
    theta_from = np.mod(theta_to + np.pi, 2.0 * np.pi)
    up_num = np.zeros(n)
    up_den = np.zeros(n)
    up_count = np.zeros(n, dtype=int)
    dn_num = np.zeros(n)
    dn_den = np.zeros(n)
    dn_count = np.zeros(n, dtype=int)
    for i in range(len(meta5)):
        pm = df["nb_" + str(i + 1)].values
        valid = ~np.isnan(pm)
        w = 1.0 / max(meta5[i]["dist_km"], 0.1)
        beta = meta5[i]["bearing_rad"]
        in_up = (np.abs(angle_diff(beta, theta_from)) <= cone_half_rad) & valid
        in_dn = (np.abs(angle_diff(beta, theta_to)) <= cone_half_rad) & valid
        up_num = up_num + np.where(in_up, pm * w, 0.0)
        up_den = up_den + np.where(in_up, w, 0.0)
        up_count = up_count + in_up.astype(int)
        dn_num = dn_num + np.where(in_dn, pm * w, 0.0)
        dn_den = dn_den + np.where(in_dn, w, 0.0)
        dn_count = dn_count + in_dn.astype(int)
    with np.errstate(invalid="ignore", divide="ignore"):
        df["pm_upwind"] = np.where(up_den > 0, up_num / up_den, np.nan)
        df["pm_downwind"] = np.where(dn_den > 0, dn_num / dn_den, np.nan)
    df["n_upwind"] = up_count
    df["n_downwind"] = dn_count

    if len(meta8) > 0:
        lat0 = coords.loc[station_id, "lat"]
        lon0 = coords.loc[station_id, "lon"]
        A = np.zeros((len(meta8), 6))
        for i in range(len(meta8)):
            dx = (meta8[i]["lon"] - lon0) * 111.32 * np.cos(np.radians(lat0))
            dy = (meta8[i]["lat"] - lat0) * 110.57
            A[i, 0] = 1.0
            A[i, 1] = dx
            A[i, 2] = dy
            A[i, 3] = dx * dx
            A[i, 4] = dy * dy
            A[i, 5] = dx * dy
        A_pinv = np.linalg.pinv(A)
        values = np.zeros((n, len(meta8)))
        for i in range(len(meta8)):
            values[:, i] = panel[meta8[i]["id"]].reindex(df["datetime"]).values
        coef = values @ A_pinv.T
        bad = np.isnan(values).any(axis=1)
        coef[bad, :] = np.nan
        grad_x = coef[:, 1]
        grad_y = coef[:, 2]
        df["pm_gradient"] = np.sqrt(grad_x ** 2 + grad_y ** 2)
        df["pm_advection"] = -(df["u10"].values * grad_x + df["v10"].values * grad_y) * 3.6
        df["pm_diffusion"] = 2.0 * coef[:, 3] + 2.0 * coef[:, 4]
    else:
        df["pm_gradient"] = np.nan
        df["pm_advection"] = np.nan
        df["pm_diffusion"] = np.nan

    df["pm_upwind_isna"] = df["pm_upwind"].isna().astype(int)
    df["pm_downwind_isna"] = df["pm_downwind"].isna().astype(int)
    return df


def fit_feature_scaling(cube, feature_names):
    # clip and MinMax statistics per channel, training years of the pool only
    stats = []
    for f in range(len(feature_names)):
        name = feature_names[f]
        train_vals = cube[TRAIN_ROWS, :, f].astype(np.float64)
        if name in NO_CLIP_COLS:
            clip = None
        else:
            clip = float(np.percentile(train_vals, CLIP_PCT))
            train_vals = np.minimum(train_vals, clip)
        mn = float(train_vals.min())
        mx = float(train_vals.max())
        stats.append({"name": name, "clip": clip, "min": mn, "max": mx})
    return stats


def apply_feature_scaling(cube, stats):
    scaled = cube.astype(np.float32).copy()
    for f in range(len(stats)):
        st = stats[f]
        col = scaled[..., f]
        if st["clip"] is not None:
            col = np.minimum(col, st["clip"])
        if st["max"] > st["min"]:
            col = 2.0 * (col - st["min"]) / (st["max"] - st["min"]) - 1.0
        else:
            col = np.zeros_like(col)
        scaled[..., f] = col
    return scaled


def build_fold_data(held_out, radius_km):
    # same as script 01, but the node cube carries the master feature list
    # (69 feature roles + lcs_median_pm25 for the raw variant)
    lat0 = coords.loc[held_out, "lat"]
    lon0 = coords.loc[held_out, "lon"]

    removed = []
    for other in STATION_IDS:
        if other == held_out:
            continue
        d = float(haversine_km(lon0, lat0, coords.loc[other, "lon"], coords.loc[other, "lat"]))
        if radius_km > 0 and d <= radius_km:
            removed.append(other)

    available = []
    for s in STATION_IDS:
        if s != held_out and s not in removed:
            available.append(s)

    if len(available) == 0:
        raise ValueError("radius " + str(radius_km) + " km removes every station around " + held_out)

    pool_tables = {}
    for s in available:
        pool_tables[s] = rebuild_network_features(s, stations[s], available)

    for col in NETWORK_FEATURES:
        train_parts = []
        for s in available:
            df = pool_tables[s]
            train_parts.append(df.loc[df["year"] <= FILL_YEAR_MAX, col])
        med = float(pd.concat(train_parts).median())
        if not np.isfinite(med):
            med = 0.0
        for s in available:
            pool_tables[s][col] = pool_tables[s][col].fillna(med)

    P = len(available)
    node_raw = np.zeros((N_HOURS, P, len(MASTER_NODE_FEATURES)), dtype=np.float32)
    clim_raw = np.zeros((N_HOURS, P, len(CLIMATE_FEATURES)), dtype=np.float32)
    for j in range(P):
        s = available[j]
        node_raw[:, j, :] = pool_tables[s][MASTER_NODE_FEATURES].to_numpy(dtype=np.float32)
        clim_raw[:, j, :] = pool_tables[s][CLIMATE_FEATURES].to_numpy(dtype=np.float32)

    y_raw = node_raw[:, :, 0].astype(np.float64).copy()

    node_stats = fit_feature_scaling(node_raw, MASTER_NODE_FEATURES)
    node_scaled = apply_feature_scaling(node_raw, node_stats)
    clim_stats = fit_feature_scaling(clim_raw, CLIMATE_FEATURES)
    clim_scaled = apply_feature_scaling(clim_raw, clim_stats)

    clim_target_raw = stations[held_out][CLIMATE_FEATURES].to_numpy(dtype=np.float32)
    clim_target_scaled = apply_feature_scaling(clim_target_raw, clim_stats)

    y_train_vals = y_raw[TRAIN_ROWS, :]
    y_min = float(y_train_vals.min())
    y_den = float(np.percentile(y_train_vals, Y_SCALE_PCT))
    if y_den <= y_min:
        y_den = y_min + 1.0
    y_scaled = (2.0 * (y_raw - y_min) / (y_den - y_min) - 1.0).astype(np.float32)

    dist_pool = np.zeros((P, P))
    for a in range(P):
        for b in range(P):
            dist_pool[a, b] = haversine_km(coords.loc[available[a], "lon"], coords.loc[available[a], "lat"],
                                           coords.loc[available[b], "lon"], coords.loc[available[b], "lat"])
    dist_target = np.zeros(P)
    for a in range(P):
        dist_target[a] = haversine_km(lon0, lat0,
                                      coords.loc[available[a], "lon"], coords.loc[available[a], "lat"])

    y_target_raw = stations[held_out]["ref_pm25"].to_numpy(dtype=float)

    return {
        "held_out": held_out,
        "available": available,
        "removed": removed,
        "node_scaled": node_scaled,
        "clim_scaled": clim_scaled,
        "clim_target_scaled": clim_target_scaled,
        "y_scaled": y_scaled,
        "y_min": y_min,
        "y_den": y_den,
        "dist_pool": dist_pool,
        "dist_target": dist_target,
        "y_target_raw": y_target_raw,
    }


def compute_metrics(y_true, y_pred):
    # RMSE, MAE and R2 on the rows where both values exist
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[ok]
    y_pred = y_pred[ok]
    if len(y_true) == 0:
        return {"n_hours": 0, "RMSE": np.nan, "MAE": np.nan, "R2": np.nan}
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot > 0:
        r2 = 1.0 - ss_res / ss_tot
    else:
        r2 = np.nan
    return {"n_hours": int(len(y_true)), "RMSE": rmse, "MAE": mae, "R2": r2}


def hourly_metrics_table(pred_df):
    rows = []
    for dt_value, group in pred_df.groupby("datetime"):
        m = compute_metrics(group["y_true"], group["y_pred"])
        rows.append({"datetime": dt_value, "n_stations": m["n_hours"],
                     "RMSE": m["RMSE"], "MAE": m["MAE"], "R2": m["R2"]})
    out = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
    return out


def bh_correction(p_values):
    p = pd.Series(p_values, dtype=float)
    m = p.notna().sum()
    q = (p * m / p.rank(method="first")).clip(upper=1.0)
    return q


def run_variant(fold, node_names, clim_names, encoder_cache, stdgi_path, decoder_path):
    # Train one ablation variant on this fold, or load it if both model files
    # already exist, then predict every 2024 hour of the held out station.
    # Encoders are cached per node feature set inside the fold when trained
    # fresh, because the unsupervised encoder never sees the climate features,
    # so variants that only change the climate block reuse the same encoder,
    # the shared encoder is still saved under each variant's own file name.
    seed_everything(SEED)
    P = len(fold["available"])

    node_idx = [MASTER_NODE_FEATURES.index(c) for c in node_names]
    clim_idx = [CLIMATE_FEATURES.index(c) for c in clim_names]
    node = fold["node_scaled"][:, :, node_idx]
    clim = fold["clim_scaled"][:, :, clim_idx]
    clim_target = fold["clim_target_scaled"][:, clim_idx]

    test_ds = FoldDataset("test", node, fold["y_scaled"], clim, clim_target,
                          fold["dist_pool"], fold["dist_target"],
                          SEQUENCE_LENGTH, TEST_T, dist_floor_km=DIST_FLOOR_KM)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    resumed = os.path.exists(stdgi_path) and os.path.exists(decoder_path)
    if resumed:
        stdgi = Attention_STDGI(in_ft=len(node_names), out_ft=OUTPUT_STDGI,
                                en_hid1=EN_HID1, en_hid2=EN_HID2, dis_hid=DIS_HID,
                                stdgi_noise_min=STDGI_NOISE_MIN, stdgi_noise_max=STDGI_NOISE_MAX).to(DEVICE)
        load_model(stdgi, stdgi_path)
        stdgi.eval()
        decoder = Local_Global_Decoder(len(node_names) + OUTPUT_STDGI, 1,
                                       n_layers_rnn=N_LAYERS_RNN, rnn=RNN_TYPE,
                                       cnn_hid_dim=CNN_HID_DIM, fc_hid_dim=FC_HID_DIM,
                                       n_features=len(clim_names),
                                       num_input_stat=P).to(DEVICE)
        load_model(decoder, decoder_path)
        decoder.eval()
        stdgi_epochs = 0
        decoder_epochs = 0
        best_val = float("nan")
    else:
        train_ds = FoldDataset("train", node, fold["y_scaled"], clim, clim_target,
                               fold["dist_pool"], fold["dist_target"],
                               SEQUENCE_LENGTH, TRAIN_T, dist_floor_km=DIST_FLOOR_KM)
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

        val_pairs = []
        for j in range(P):
            for t in VAL_T:
                val_pairs.append((t, j))
        val_ds = FoldDataset("val", node, fold["y_scaled"], clim, clim_target,
                             fold["dist_pool"], fold["dist_target"],
                             SEQUENCE_LENGTH, VAL_T, pair_list=val_pairs, dist_floor_km=DIST_FLOOR_KM)
        val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=True)

        scratch_e = os.path.join(MODELS_DIR, "_scratch_sq2_stdgi.pt")
        scratch_d = os.path.join(MODELS_DIR, "_scratch_sq2_decoder.pt")

        # stage 1: encoder (reused when this fold already trained one for this node set)
        cache_key = tuple(node_names)
        if cache_key in encoder_cache:
            stdgi, stdgi_epochs = encoder_cache[cache_key]
        else:
            stdgi = Attention_STDGI(in_ft=len(node_names), out_ft=OUTPUT_STDGI,
                                    en_hid1=EN_HID1, en_hid2=EN_HID2, dis_hid=DIS_HID,
                                    stdgi_noise_min=STDGI_NOISE_MIN, stdgi_noise_max=STDGI_NOISE_MAX).to(DEVICE)
            opt_e = torch.optim.Adam(stdgi.encoder.parameters(), lr=LR_STDGI)
            opt_d = torch.optim.Adam(stdgi.disc.parameters(), lr=LR_STDGI)
            bce = nn.BCELoss()
            es_e = EarlyStopping(patience=PATIENCE, path=scratch_e)
            stdgi_epochs = 0
            for epoch in range(NUM_EPOCHS_STDGI):
                if es_e.early_stop:
                    break
                loss = train_stdgi_one_epoch(stdgi, train_dl, opt_e, opt_d, bce, DEVICE, n_steps=2)
                es_e(loss, stdgi)
                stdgi_epochs = epoch + 1
            load_model(stdgi, scratch_e)
            encoder_cache[cache_key] = (stdgi, stdgi_epochs)
        stdgi.eval()

        # stage 2: decoder
        decoder = Local_Global_Decoder(len(node_names) + OUTPUT_STDGI, 1,
                                       n_layers_rnn=N_LAYERS_RNN, rnn=RNN_TYPE,
                                       cnn_hid_dim=CNN_HID_DIM, fc_hid_dim=FC_HID_DIM,
                                       n_features=len(clim_names),
                                       num_input_stat=P).to(DEVICE)
        opt_dec = torch.optim.Adam(decoder.parameters(), lr=LR_DECODER)
        mse = nn.MSELoss()
        es_d = EarlyStopping(patience=PATIENCE, path=scratch_d)
        decoder_epochs = 0
        for epoch in range(NUM_EPOCHS_DECODER):
            if es_d.early_stop:
                break
            train_loss = train_decoder_one_epoch(stdgi, decoder, train_dl, mse, opt_dec, DEVICE)
            val_loss = decoder_val_loss(stdgi, decoder, val_dl, mse, DEVICE)
            es_d(val_loss, decoder)
            decoder_epochs = epoch + 1
        load_model(decoder, scratch_d)
        best_val = float(es_d.best_score)

        save_model(stdgi, stdgi_path)
        save_model(decoder, decoder_path)

    # predict every 2024 hour and inverse the target scaling
    preds_scaled = predict_decoder(stdgi, decoder, test_dl, DEVICE)
    y_pred = (preds_scaled + 1.0) / 2.0 * (fold["y_den"] - fold["y_min"]) + fold["y_min"]

    del decoder, test_ds, test_dl
    if resumed:
        del stdgi          # a loaded encoder is not shared, free it; cached ones stay for the fold
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return y_pred, stdgi_epochs, decoder_epochs, best_val, resumed



# Train (or load) all 9 variants per held out station, evaluate on 2024

all_per_station = {}
all_preds = {}
for name in ABLATIONS:
    all_per_station[name] = []
    all_preds[name] = []

t0 = time.time()
for k, held_out in enumerate(STATION_IDS):
    fold = build_fold_data(held_out, radius_km=0)
    encoder_cache = {}
    test_rows = stations[held_out][stations[held_out]["year"] == TEST_YEAR]

    for name in ABLATIONS:
        node_names, clim_names = ABLATIONS[name]
        stdgi_path = os.path.join(MODELS_DIR, "sq2_" + name + "_" + held_out + "_stdgi.pt")
        decoder_path = os.path.join(MODELS_DIR, "sq2_" + name + "_" + held_out + "_decoder.pt")
        y_pred, e1, e2, best_val, resumed = run_variant(fold, node_names, clim_names, encoder_cache,
                                                        stdgi_path, decoder_path)

        preds = pd.DataFrame({
            "station_id": held_out,
            "datetime": test_rows["datetime"].values,
            "y_true": test_rows["ref_pm25"].values,
            "y_pred": y_pred,
        })
        all_preds[name].append(preds)

        m = compute_metrics(preds["y_true"], preds["y_pred"])
        m["station_id"] = held_out
        all_per_station[name].append(m)

        status = "loaded " if resumed else "trained"
        minutes = (time.time() - t0) / 60.0
        print("   %s %-18s RMSE %.3f | %.1f min" % (status, name, m["RMSE"], minutes))

    encoder_cache.clear()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # crash safe: rewrite the per variant per station and prediction tables after every fold
    for name in ABLATIONS:
        ps = pd.DataFrame(all_per_station[name])[["station_id", "n_hours", "RMSE", "MAE", "R2"]]
        ps.round(4).to_csv(os.path.join(OUT_TABLES, "sq2_" + name + "_per_station_metrics.csv"), index=False)
        pd.concat(all_preds[name], ignore_index=True).to_csv(
            os.path.join(OUT_TABLES, "sq2_" + name + "_predictions.csv"), index=False, float_format="%.4f")

    print("[%2d/%d] %s done | full RMSE %.3f | %.1f min"
          % (k + 1, len(STATION_IDS), held_out, all_per_station["full"][-1]["RMSE"], (time.time() - t0) / 60.0))

print()
print("all variants evaluated")


# Hourly metrics, summary and Wilcoxon tests


per_station_tables = {}
for name in ABLATIONS:
    ps = pd.DataFrame(all_per_station[name])[["station_id", "n_hours", "RMSE", "MAE", "R2"]]
    per_station_tables[name] = ps
    preds = pd.concat(all_preds[name], ignore_index=True)
    hourly = hourly_metrics_table(preds)
    hourly.to_csv(os.path.join(OUT_TABLES, "sq2_" + name + "_hourly_metrics.csv"), index=False, float_format="%.4f")
    print(name, "saved | mean station RMSE %.3f" % ps["RMSE"].mean())

rows = []
for name in ["full"] + ABL_ORDER:
    ps = per_station_tables[name]
    node_names, clim_names = ABLATIONS[name]
    rows.append({
        "ablation": name,
        "label": "full model" if name == "full" else ABL_LABEL[name],
        "n_features": len(node_names) + len(clim_names),
        "mean_RMSE": ps["RMSE"].mean(),
        "mean_MAE": ps["MAE"].mean(),
        "mean_R2": ps["R2"].mean(),
        "median_RMSE": ps["RMSE"].median(),
    })
summary = pd.DataFrame(rows)
full_rmse = float(summary.loc[summary["ablation"] == "full", "mean_RMSE"].iloc[0])
full_r2 = float(summary.loc[summary["ablation"] == "full", "mean_R2"].iloc[0])
summary["dRMSE_vs_full"] = summary["mean_RMSE"] - full_rmse
summary["dRMSE_pct"] = 100.0 * summary["dRMSE_vs_full"] / full_rmse
summary["dR2_vs_full"] = summary["mean_R2"] - full_r2
summary.round(4).to_csv(os.path.join(OUT_TABLES, "sq2_summary.csv"), index=False)
print()
print(summary.round(3).to_string(index=False))

full_ps = per_station_tables["full"]
rows = []
for name in ABL_ORDER:
    ps = per_station_tables[name]
    merged = full_ps[["station_id", "RMSE"]].merge(ps[["station_id", "RMSE"]],
                                                   on="station_id", suffixes=("_full", "_abl"))
    diff = (merged["RMSE_abl"] - merged["RMSE_full"]).values
    nonzero = diff[diff != 0]
    if len(nonzero) >= 3:
        p = float(wilcoxon(nonzero).pvalue)
    else:
        p = np.nan
    rows.append({"ablation": name, "label": ABL_LABEL[name],
                 "n_stations": len(diff),
                 "n_worse_than_full": int((diff > 0).sum()),
                 "median_dRMSE": float(np.median(diff)),
                 "p": p})
wil = pd.DataFrame(rows)
wil["q_BH"] = bh_correction(wil["p"])
wil["significant"] = wil["q_BH"] < 0.05
wil.round(6).to_csv(os.path.join(OUT_TABLES, "sq2_wilcoxon_vs_full.csv"), index=False)
print(wil.round(4).to_string(index=False))
print()
print("saved the sq2 per variant tables, sq2_summary.csv and sq2_wilcoxon_vs_full.csv")


# remove the scratch checkpoints, only the final per fold files matter
for scratch in ["_scratch_sq2_stdgi.pt", "_scratch_sq2_decoder.pt"]:
    scratch_path = os.path.join(MODELS_DIR, scratch)
    if os.path.exists(scratch_path):
        os.remove(scratch_path)
