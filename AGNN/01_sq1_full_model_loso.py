"""
SQ1: AGNN full model under LOSO (train and evaluate)

Trains the full AGNN (encoder + decoder) once per held out station under the
same leave one station out protocol as the XGBoost notebook 01, evaluates it on
the held out station's 2024 hours, and saves both the models and the result
tables.

Saved per fold to results/models/ (archived, reloaded on a rerun so a second
run re-evaluates without retraining):
    sq1_<station>_stdgi.pt      the unsupervised STDGI encoder
    sq1_<station>_decoder.pt    the supervised localglobal attention decoder

Saved to results/tables/ (the evaluation results the thesis figures are built
from, same file names as the XGBoost notebooks):
    sq1_per_station_metrics.csv   RMSE, MAE, R2 per held out station
    sq1_predictions.csv           y_true and y_pred per station hour
    sq1_hourly_metrics.csv        RMSE, MAE, R2 across stations per 2024 hour
    sq1_summary.csv               mean and median over the 31 stations

The figures are made by 05_agnn_evaluation_and_figures.ipynb, which only reads
these CSVs. 

Protocol per fold (31 folds): the held out station is removed everywhere and
the neighbour features of the 30 pool stations are rebuilt from the pool only
(identical code to the XGBoost notebooks). NaN are filled with the pool
training year median. Features are clipped at the training year 95th percentile
(bounded columns exempt) and MinMax scaled to (-1, 1) with pool training
statistics; the PM2.5 target is scaled linearly and never clipped. The encoder
trains on 2021 to 2022 with early stopping on its own loss, the decoder on 2021
to 2022 with early stopping on the 2023 validation loss. Folds whose two model
files already exist are not retrained, only re-evaluated.
"""

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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


# Feature groups (the 69 XGBoost features) and the AGNN feature roles


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

# AGNN feature roles. Node features describe the input stations (PM2.5 must be
# first, it is the quantity the graph interpolates). Climate features describe
# the target location and are the only thing the decoder knows there.
NODE_FEATURES = ["ref_pm25"] + NETWORK_FEATURES + LCS_FEATURES + EAC4_FEATURES
CLIMATE_FEATURES = ERA5_FEATURES + TIME_SINCOS

# columns whose values are already bounded, clipping them at the 95th
# percentile would destroy them (a 0/1 flag with fewer than 5% ones would be
# clipped to all zero), so they are scaled but not clipped
NO_CLIP_COLS = [
    "low_blh", "stagnant", "n_upwind", "n_downwind",
    "pm_upwind_isna", "pm_downwind_isna",
    "wind_dir_sin", "wind_dir_cos",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "dow_sin", "dow_cos",
]

print("node features:", len(NODE_FEATURES), "| climate features:", len(CLIMATE_FEATURES))


# Load the data (identical to the XGBoost notebooks)


t0 = time.time()
station_folders = sorted(os.listdir(DATA_DIR))

stations = {}
for folder in station_folders:
    folder_path = os.path.join(DATA_DIR, folder)
    if not os.path.isdir(folder_path):
        continue  # skips loose files like imputation_report.csv
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

# wide panel of the imputed reference PM2.5: rows are hours, columns are stations
panel = pd.DataFrame()
for name in STATION_IDS:
    df = stations[name]
    panel[name] = pd.Series(df["ref_pm25"].values, index=df["datetime"])
panel = panel.sort_index()

N_HOURS = len(panel)
YEAR_VEC = stations[STATION_IDS[0]]["year"].to_numpy()
TRAIN_ROWS = YEAR_VEC <= TRAIN_YEAR_MAX

# target hour lists for the three dataset modes
TRAIN_T = [int(t) for t in np.where(YEAR_VEC <= TRAIN_YEAR_MAX)[0] if t >= SEQUENCE_LENGTH - 1]
VAL_T = [int(t) for t in np.where(YEAR_VEC == VAL_YEAR)[0][::VAL_STRIDE_HOURS]]
# every 2024 hour is scored; the 12 hour input windows of the first hours
# simply reach back into late 2023
TEST_T = [int(t) for t in np.where(YEAR_VEC == TEST_YEAR)[0]]

print("stations:", len(stations), "| panel shape:", panel.shape,
      "| loaded in", round(time.time() - t0, 1), "s")
print("train hours:", len(TRAIN_T), "| val windows per station:", len(VAL_T))


# Helper functions

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
    # Returns a copy of the station table in which every neighbour derived feature is
    # recomputed from only the stations in available_ids. This is the leakage control:
    # the held out station (and any station removed by a distance band) is not in
    # available_ids, so its measurements cannot reach any feature in this fold.
    # Identical to the XGBoost notebooks. Values that cannot be computed stay NaN,
    # they are filled afterwards because the AGNN cannot handle NaN.
    df = df_station.copy()
    n = len(df)
    meta5 = nearest_neighbours(station_id, available_ids, K_NEAREST)
    meta8 = nearest_neighbours(station_id, available_ids, K_QUAD)

    # 1. neighbour readings nb_1 to nb_5
    for i in range(K_NEAREST):
        if i < len(meta5):
            df["nb_" + str(i + 1)] = panel[meta5[i]["id"]].reindex(df["datetime"]).values
        else:
            df["nb_" + str(i + 1)] = np.nan

    # 2. inverse distance weighted mean of the neighbour readings
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

    # 3. lagged block from the nearest neighbour (left closed, so no future values)
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

    # 4. upwind and downwind neighbour averages (120 degree cones)
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

    # 5. convection diffusion terms from a quadratic surface fit on the 8 nearest
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

    # 6. fresh missing indicators for the two cone features
    df["pm_upwind_isna"] = df["pm_upwind"].isna().astype(int)
    df["pm_downwind_isna"] = df["pm_downwind"].isna().astype(int)
    return df


# scaling helpers

def fit_feature_scaling(cube, feature_names):
    # clip and MinMax statistics per channel, computed on the training years of
    # the pool stations only. cube shape is [hours, stations, features].
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
    # works for [hours, stations, features] and [hours, features]
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


# fold construction

def build_fold_data(held_out, radius_km):
    # Builds everything one AGNN fold needs. Station selection is identical to
    # the XGBoost build_fold_table. The held out station is never a node, so
    # only its climate columns and its raw ref_pm25 (the evaluation truth) are
    # taken from its table, its node features are never used anywhere.
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

    # rebuild the neighbour features of every pool station from the pool only
    pool_tables = {}
    for s in available:
        pool_tables[s] = rebuild_network_features(s, stations[s], available)

    # fill NaN in the rebuilt columns with the pool median over the training
    # years, the same rule feature_engineering.ipynb used for the stored table
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

    # stack the pool into [hours, stations, features] cubes
    P = len(available)
    node_raw = np.zeros((N_HOURS, P, len(NODE_FEATURES)), dtype=np.float32)
    clim_raw = np.zeros((N_HOURS, P, len(CLIMATE_FEATURES)), dtype=np.float32)
    for j in range(P):
        s = available[j]
        node_raw[:, j, :] = pool_tables[s][NODE_FEATURES].to_numpy(dtype=np.float32)
        clim_raw[:, j, :] = pool_tables[s][CLIMATE_FEATURES].to_numpy(dtype=np.float32)

    y_raw = node_raw[:, :, 0].astype(np.float64).copy()  # unclipped pool ref_pm25

    # scale features (fit on pool stations, training years)
    node_stats = fit_feature_scaling(node_raw, NODE_FEATURES)
    node_scaled = apply_feature_scaling(node_raw, node_stats)
    clim_stats = fit_feature_scaling(clim_raw, CLIMATE_FEATURES)
    clim_scaled = apply_feature_scaling(clim_raw, clim_stats)

    # climate at the held out target, scaled with the pool statistics
    clim_target_raw = stations[held_out][CLIMATE_FEATURES].to_numpy(dtype=np.float32)
    clim_target_scaled = apply_feature_scaling(clim_target_raw, clim_stats)

    # target scaling: linear, never clipped, denominator is the training p99.9
    # so one extreme fireworks hour does not squash all typical values
    y_train_vals = y_raw[TRAIN_ROWS, :]
    y_min = float(y_train_vals.min())
    y_den = float(np.percentile(y_train_vals, Y_SCALE_PCT))
    if y_den <= y_min:
        y_den = y_min + 1.0
    y_scaled = (2.0 * (y_raw - y_min) / (y_den - y_min) - 1.0).astype(np.float32)

    # distances
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
    # RMSE, MAE and R2 across the held out stations, for every evaluated hour
    rows = []
    for dt_value, group in pred_df.groupby("datetime"):
        m = compute_metrics(group["y_true"], group["y_pred"])
        rows.append({"datetime": dt_value, "n_stations": m["n_hours"],
                     "RMSE": m["RMSE"], "MAE": m["MAE"], "R2": m["R2"]})
    out = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
    return out


def bh_correction(p_values):
    # BH correction, same implementation as the XGBoost notebooks
    p = pd.Series(p_values, dtype=float)
    m = p.notna().sum()
    q = (p * m / p.rank(method="first")).clip(upper=1.0)
    return q


def run_fold(fold, stdgi_path, decoder_path):
    # Train the encoder and decoder for this fold, or load them if both model
    # files already exist (a rerun then re-evaluates without retraining), then
    # predict every 2024 hour of the held out station. The prediction is made
    # here, in the same run that owns the model, so the scaling seen at train
    # time and at prediction time is guaranteed identical. Returns the 2024
    # prediction in ug/m3, the epoch counts, the best validation loss, and
    # whether the models were loaded rather than trained.
    seed_everything(SEED)
    P = len(fold["available"])

    test_ds = FoldDataset("test", fold["node_scaled"], fold["y_scaled"], fold["clim_scaled"],
                          fold["clim_target_scaled"], fold["dist_pool"], fold["dist_target"],
                          SEQUENCE_LENGTH, TEST_T, dist_floor_km=DIST_FLOOR_KM)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    resumed = os.path.exists(stdgi_path) and os.path.exists(decoder_path)
    if resumed:
        # load the archived models and skip training
        stdgi = Attention_STDGI(in_ft=len(NODE_FEATURES), out_ft=OUTPUT_STDGI,
                                en_hid1=EN_HID1, en_hid2=EN_HID2, dis_hid=DIS_HID,
                                stdgi_noise_min=STDGI_NOISE_MIN, stdgi_noise_max=STDGI_NOISE_MAX).to(DEVICE)
        load_model(stdgi, stdgi_path)
        stdgi.eval()
        decoder = Local_Global_Decoder(len(NODE_FEATURES) + OUTPUT_STDGI, 1,
                                       n_layers_rnn=N_LAYERS_RNN, rnn=RNN_TYPE,
                                       cnn_hid_dim=CNN_HID_DIM, fc_hid_dim=FC_HID_DIM,
                                       n_features=len(CLIMATE_FEATURES),
                                       num_input_stat=P).to(DEVICE)
        load_model(decoder, decoder_path)
        decoder.eval()
        stdgi_epochs = 0
        decoder_epochs = 0
        best_val = float("nan")
    else:
        train_ds = FoldDataset("train", fold["node_scaled"], fold["y_scaled"], fold["clim_scaled"],
                               fold["clim_target_scaled"], fold["dist_pool"], fold["dist_target"],
                               SEQUENCE_LENGTH, TRAIN_T, dist_floor_km=DIST_FLOOR_KM)
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

        val_pairs = []
        for j in range(P):
            for t in VAL_T:
                val_pairs.append((t, j))
        val_ds = FoldDataset("val", fold["node_scaled"], fold["y_scaled"], fold["clim_scaled"],
                             fold["clim_target_scaled"], fold["dist_pool"], fold["dist_target"],
                             SEQUENCE_LENGTH, VAL_T, pair_list=val_pairs, dist_floor_km=DIST_FLOOR_KM)
        val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=True)

        scratch_e = os.path.join(MODELS_DIR, "_scratch_sq1_stdgi.pt")
        scratch_d = os.path.join(MODELS_DIR, "_scratch_sq1_decoder.pt")

        # stage 1: unsupervised encoder, early stop on its own training loss
        stdgi = Attention_STDGI(in_ft=len(NODE_FEATURES), out_ft=OUTPUT_STDGI,
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
        stdgi.eval()

        # stage 2: supervised decoder, early stop on the 2023 validation loss
        decoder = Local_Global_Decoder(len(NODE_FEATURES) + OUTPUT_STDGI, 1,
                                       n_layers_rnn=N_LAYERS_RNN, rnn=RNN_TYPE,
                                       cnn_hid_dim=CNN_HID_DIM, fc_hid_dim=FC_HID_DIM,
                                       n_features=len(CLIMATE_FEATURES),
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

        # both stages finished, write the final model files
        save_model(stdgi, stdgi_path)
        save_model(decoder, decoder_path)

    # predict every 2024 hour and inverse the target scaling
    preds_scaled = predict_decoder(stdgi, decoder, test_dl, DEVICE)
    y_pred = (preds_scaled + 1.0) / 2.0 * (fold["y_den"] - fold["y_min"]) + fold["y_min"]

    del stdgi, decoder, test_ds, test_dl
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return y_pred, stdgi_epochs, decoder_epochs, best_val, resumed



# Train (or load) each fold, evaluate on 2024, save the result tables


per_station_rows = []
pred_parts = []
t0 = time.time()

for k, held_out in enumerate(STATION_IDS):
    stdgi_path = os.path.join(MODELS_DIR, "sq1_" + held_out + "_stdgi.pt")
    decoder_path = os.path.join(MODELS_DIR, "sq1_" + held_out + "_decoder.pt")

    fold = build_fold_data(held_out, radius_km=0)
    y_pred, stdgi_epochs, decoder_epochs, best_val, resumed = run_fold(fold, stdgi_path, decoder_path)

    test_rows = stations[held_out][stations[held_out]["year"] == TEST_YEAR]
    preds = pd.DataFrame({
        "station_id": held_out,
        "datetime": test_rows["datetime"].values,
        "y_true": test_rows["ref_pm25"].values,
        "y_pred": y_pred,
    })
    pred_parts.append(preds)

    m = compute_metrics(preds["y_true"], preds["y_pred"])
    m["station_id"] = held_out
    per_station_rows.append(m)

    # crash safe: rewrite the per station metrics and predictions after every fold
    per_station = pd.DataFrame(per_station_rows)[["station_id", "n_hours", "RMSE", "MAE", "R2"]]
    per_station.round(4).to_csv(os.path.join(OUT_TABLES, "sq1_per_station_metrics.csv"), index=False)
    pd.concat(pred_parts, ignore_index=True).to_csv(
        os.path.join(OUT_TABLES, "sq1_predictions.csv"), index=False, float_format="%.4f")

    status = "loaded " if resumed else "trained"
    minutes = (time.time() - t0) / 60.0
    print("[%2d/%d] %s %s: RMSE %.3f | %.1f min"
          % (k + 1, len(STATION_IDS), status, held_out, m["RMSE"], minutes))

predictions = pd.concat(pred_parts, ignore_index=True)

# hourly metrics 
hourly = hourly_metrics_table(predictions)
hourly.to_csv(os.path.join(OUT_TABLES, "sq1_hourly_metrics.csv"), index=False, float_format="%.4f")

summary = pd.DataFrame([
    {"aggregation": "mean over stations",
     "RMSE": per_station["RMSE"].mean(),
     "MAE": per_station["MAE"].mean(),
     "R2": per_station["R2"].mean()},
    {"aggregation": "median over stations",
     "RMSE": per_station["RMSE"].median(),
     "MAE": per_station["MAE"].median(),
     "R2": per_station["R2"].median()},
])
summary.round(4).to_csv(os.path.join(OUT_TABLES, "sq1_summary.csv"), index=False)

print()
print("done:", len(predictions), "predicted station hours")
print(summary.round(3).to_string(index=False))
print()
print("saved sq1_per_station_metrics.csv, sq1_predictions.csv, sq1_hourly_metrics.csv, sq1_summary.csv")

# remove the scratch checkpoints, only the final per fold files matter
for scratch in ["_scratch_sq1_stdgi.pt", "_scratch_sq1_decoder.pt"]:
    scratch_path = os.path.join(MODELS_DIR, scratch)
    if os.path.exists(scratch_path):
        os.remove(scratch_path)
