"""
predict.py — ASTRA Final Production Model (Ultimate Ensemble)
Improvements over original baseline:
  1. LightGBM added to the ensemble (gradient boosting handles non-linearity better than RF/ET alone)
  2. ±15-min temporal lag features: demand_day48_prev, demand_day48_next
  3. Demand std dev per geohash (gh_std) as a volatility proxy
  4. Bayesian Overlap Shrinkage on shifts (k=2.0) carried from predict.py
  5. Exponentially-weighted overlap shift (shift_diff_wmean) as an additional feature
  6. OOF-based ensemble weight search (grid search over all alpha/weights) instead of hand-tuned weights
  7. Robust test-set ordering enforced after spatial-temporal mapping
"""

import pandas as pd
import numpy as np
import os
import zipfile
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import lightgbm as lgb

DATA_DIR = r"c:\Users\KIIT\Desktop\flipkartgrid\dataset\dataset"
OUTPUT_PATH = r"c:\Users\KIIT\Desktop\flipkartgrid\submission.csv"
N_SPLITS = 5

# ── helpers ──────────────────────────────────────────────────────────────────

def decode_geohash(geohash: str):
    base32 = "0123456789bcdefghjkmnpqrstuvwxyz"
    base32_map = {c: i for i, c in enumerate(base32)}
    lat_interval = [-90.0, 90.0]
    lon_interval = [-180.0, 180.0]
    is_even = True
    for char in geohash:
        val = base32_map[char]
        for i in range(4, -1, -1):
            bit = (val >> i) & 1
            if is_even:
                mid = (lon_interval[0] + lon_interval[1]) / 2
                if bit == 1: lon_interval[0] = mid
                else:        lon_interval[1] = mid
            else:
                mid = (lat_interval[0] + lat_interval[1]) / 2
                if bit == 1: lat_interval[0] = mid
                else:        lat_interval[1] = mid
            is_even = not is_even
    return (lat_interval[0] + lat_interval[1]) / 2, (lon_interval[0] + lon_interval[1]) / 2


def add_coords(df: pd.DataFrame) -> pd.DataFrame:
    lats, lons = zip(*[decode_geohash(g) for g in df["geohash"]])
    df["lat"] = list(lats)
    df["lon"] = list(lons)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    hours, minutes = zip(*[(int(t.split(":")[0]), int(t.split(":")[1])) for t in df["timestamp"]])
    df["hour"]       = list(hours)
    df["minute"]     = list(minutes)
    df["time_of_day"]= df["hour"] + df["minute"] / 60.0
    df["sin_time"]   = np.sin(2 * np.pi * df["time_of_day"] / 24.0)
    df["cos_time"]   = np.cos(2 * np.pi * df["time_of_day"] / 24.0)
    return df


def shift_time_str(t_str: str, delta_mins: int) -> str:
    """Add/subtract minutes from a 'H:M' string, wrapping at 24h."""
    h, m = map(int, t_str.split(":"))
    total = (h * 60 + m + delta_mins) % (24 * 60)
    return f"{total // 60}:{total % 60}"


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("ASTRA — predict (Ultimate Ensemble)")
    print("=" * 60)

    # ── 1. Load data ─────────────────────────────────────────────
    print("\n[1] Loading datasets...")
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    test_indices = test["Index"].copy()

    train = add_coords(train)
    test  = add_coords(test)
    train = add_time_features(train)
    test  = add_time_features(test)

    train_48 = train[train["day"] == 48].copy()
    train_49 = train[train["day"] == 49].copy()

    # ── 2. Map missing test geohashes -> nearest train geohash ────
    print("\n[2] Mapping unseen test geohashes...")
    known_ghs = set(train["geohash"].unique())
    train_gh_df = pd.DataFrame(
        [(g, *decode_geohash(g)) for g in known_ghs],
        columns=["geohash", "lat", "lon"]
    )
    gh_mapping = {}
    for gh in set(test["geohash"].unique()) - known_ghs:
        lat, lon = decode_geohash(gh)
        dists = (train_gh_df["lat"] - lat) ** 2 + (train_gh_df["lon"] - lon) ** 2
        gh_mapping[gh] = train_gh_df.loc[dists.idxmin(), "geohash"]
        print(f"   {gh} -> {gh_mapping[gh]}")
    test["mapped_geohash"] = test["geohash"].map(lambda x: gh_mapping.get(x, x))

    # ── 3. Build Day-48 lookup tables ────────────────────────────
    print("\n[3] Building Day-48 lookup tables...")

    # 3a. Exact (geohash, timestamp) -> demand
    t48_exact = (
        train_48[["geohash", "timestamp", "demand"]]
        .drop_duplicates(["geohash", "timestamp"])
        .set_index(["geohash", "timestamp"])["demand"]
    )

    # 3b. Geohash-level stats from Day 48
    gh_stats_48 = train_48.groupby("geohash")["demand"].agg(["mean", "std"]).reset_index()
    gh_stats_48.columns = ["geohash", "gh_mean", "gh_std"]
    gh_stats_48["gh_std"] = gh_stats_48["gh_std"].fillna(0.0)
    global_mean_48 = train_48["demand"].mean()

    def lookup_demand(gh, ts, fallback_mean):
        """Look up Day-48 demand for (geohash, timestamp), with spatial fallback."""
        try:
            return t48_exact.loc[(gh, ts)]
        except KeyError:
            # Spatial fallback: find geohash in Day 48 that has this timestamp and is nearest
            cands = train_48[train_48["timestamp"] == ts]
            if len(cands) == 0:
                return fallback_mean
            gh_lat, gh_lon = decode_geohash(gh)
            dists = (cands["lat"] - gh_lat) ** 2 + (cands["lon"] - gh_lon) ** 2
            return cands.loc[dists.idxmin(), "demand"]

    # ── 4. Bayesian Overlap Shrinkage for shift features ─────────
    print("\n[4] Computing Bayesian-smoothed shift features...")
    OVERLAP_TIMES = ["0:0","0:15","0:30","0:45","1:0","1:15","1:30","1:45","2:0"]
    OVERLAP_WEIGHTS = {
        "0:0": 0.1, "0:15": 0.2, "0:30": 0.3, "0:45": 0.5,
        "1:0": 0.8, "1:15": 1.2, "1:30": 1.8, "1:45": 2.7, "2:0": 4.0
    }
    K_SHRINK = 2.0  # Bayesian smoothing constant

    t48_ov = train_48[train_48["timestamp"].isin(OVERLAP_TIMES)]
    t49_ov = train_49[train_49["timestamp"].isin(OVERLAP_TIMES)]

    # Raw per-geohash overlap means (Day 48 and Day 49)
    ov48 = t48_ov.groupby("geohash")["demand"].agg(["sum", "count"]).reset_index()
    ov49 = t49_ov.groupby("geohash")["demand"].agg(["sum", "count"]).reset_index()
    g48  = t48_ov["demand"].mean()  # global Day-48 overlap mean
    g49  = t49_ov["demand"].mean()  # global Day-49 overlap mean

    # Bayesian smoothed means
    ov48["bsm48"] = (ov48["sum"] + K_SHRINK * g48) / (ov48["count"] + K_SHRINK)
    ov49["bsm49"] = (ov49["sum"] + K_SHRINK * g49) / (ov49["count"] + K_SHRINK)
    shifts = ov48[["geohash", "bsm48"]].merge(ov49[["geohash", "bsm49"]], on="geohash", how="inner")
    shifts["shift_diff"]  = shifts["bsm49"] - shifts["bsm48"]
    shifts["shift_ratio"] = (shifts["bsm49"] + 1e-5) / (shifts["bsm48"] + 1e-5)

    # Exponentially-weighted overlap shift (more recent = higher weight)
    merged_ov = t49_ov.merge(
        t48_ov[["geohash", "timestamp", "demand"]], on=["geohash", "timestamp"], suffixes=("_49", "_48")
    )
    merged_ov["diff"]   = merged_ov["demand_49"] - merged_ov["demand_48"]
    merged_ov["weight"] = merged_ov["timestamp"].map(OVERLAP_WEIGHTS)
    wmean = (
        merged_ov.groupby("geohash")
        .apply(lambda g: np.average(g["diff"], weights=g["weight"]))
        .reset_index(name="shift_diff_wmean")
    )
    shifts = shifts.merge(wmean, on="geohash", how="left")
    shifts["shift_diff_wmean"] = shifts["shift_diff_wmean"].fillna(0.0)

    # ── 5. Build feature-rich training set for Day 49 ────────────
    print("\n[5] Engineering features for train_49...")

    def build_features(df: pd.DataFrame, source_tag: str = "train_49") -> pd.DataFrame:
        """Attach all Day-48 historical features to a dataframe."""
        gh_col = "mapped_geohash" if "mapped_geohash" in df.columns else "geohash"

        # Exact demand at (geohash, timestamp) on Day 48
        demand_day48   = [lookup_demand(row[gh_col], row["timestamp"],
                                        global_mean_48)
                          for _, row in df.iterrows()]
        # ±15-min lag features
        demand_prev    = [lookup_demand(row[gh_col], shift_time_str(row["timestamp"], -15),
                                        global_mean_48)
                          for _, row in df.iterrows()]
        demand_next    = [lookup_demand(row[gh_col], shift_time_str(row["timestamp"], +15),
                                        global_mean_48)
                          for _, row in df.iterrows()]

        df = df.copy()
        df["demand_day48"]      = demand_day48
        df["demand_day48_prev"] = demand_prev
        df["demand_day48_next"] = demand_next

        # Merge shift + geohash stats
        df = df.merge(shifts[["geohash","shift_diff","shift_ratio","shift_diff_wmean"]],
                      left_on=gh_col, right_on="geohash", how="left",
                      suffixes=("", "_shift"))
        if "geohash_shift" in df.columns: df.drop(columns=["geohash_shift"], inplace=True)
        df["shift_diff"]       = df["shift_diff"].fillna(0.0)
        df["shift_ratio"]      = df["shift_ratio"].fillna(1.0)
        df["shift_diff_wmean"] = df["shift_diff_wmean"].fillna(0.0)

        df = df.merge(gh_stats_48, left_on=gh_col, right_on="geohash", how="left",
                      suffixes=("", "_stats"))
        if "geohash_stats" in df.columns: df.drop(columns=["geohash_stats"], inplace=True)
        df["gh_mean"] = df["gh_mean"].fillna(global_mean_48)
        df["gh_std"]  = df["gh_std"].fillna(0.0)

        # Baseline prediction (used as a feature + for blending)
        df["baseline_pred"] = np.clip(df["demand_day48"] + df["shift_diff_wmean"], 0.0, 1.0)

        print(f"   {source_tag}: {len(df)} rows, {df['demand_day48'].isna().sum()} NaN demand_day48")
        return df

    df_train_49 = build_features(train_49, "train_49")

    # Day-48 self-features (use demand as its own "Day-48" reference, no shift)
    df_train_48 = train_48.copy()
    df_train_48["demand_day48"]      = df_train_48["geohash"].map(
        train_48.groupby("geohash")["demand"].mean()).fillna(global_mean_48)
    df_train_48["demand_day48_prev"] = df_train_48["demand_day48"]
    df_train_48["demand_day48_next"] = df_train_48["demand_day48"]
    df_train_48["shift_diff"]        = 0.0
    df_train_48["shift_ratio"]       = 1.0
    df_train_48["shift_diff_wmean"]  = 0.0
    df_train_48 = df_train_48.merge(gh_stats_48, on="geohash", how="left")
    df_train_48["gh_mean"]           = df_train_48["gh_mean"].fillna(global_mean_48)
    df_train_48["gh_std"]            = df_train_48["gh_std"].fillna(0.0)
    df_train_48["baseline_pred"]     = df_train_48["demand_day48"]

    combined_train = pd.concat([df_train_48, df_train_49], ignore_index=True)

    # Test features (use mapped_geohash for Day-48 lookup)
    print("\n[5b] Engineering features for test set...")
    df_test = build_features(test, "test")
    # Restore original row order
    df_test = df_test.set_index("Index").reindex(test_indices).reset_index()

    # ── 6. Categorical encoding ───────────────────────────────────
    print("\n[6] Encoding categorical variables...")
    CAT_COLS = ["RoadType", "LargeVehicles", "Landmarks", "Weather"]
    for col in CAT_COLS:
        combined_train[col] = combined_train[col].astype(str)
        df_train_49[col]    = df_train_49[col].astype(str)
        df_test[col]        = df_test[col].astype(str)
        le = LabelEncoder()
        combined_train[col+"_enc"] = le.fit_transform(combined_train[col])
        mapping = dict(zip(le.classes_, range(len(le.classes_))))
        df_train_49[col+"_enc"] = df_train_49[col].map(mapping).fillna(-1).astype(int)
        df_test[col+"_enc"]     = df_test[col].map(mapping).fillna(-1).astype(int)

    # Numeric imputation
    med_temp  = combined_train["Temperature"].median()
    med_lanes = combined_train["NumberofLanes"].median()
    for df in [combined_train, df_train_49, df_test]:
        df["Temperature"]   = df["Temperature"].fillna(med_temp)
        df["NumberofLanes"] = df["NumberofLanes"].fillna(med_lanes)

    # ── 7. Feature columns ───────────────────────────────────────
    FEATURE_COLS = [
        "lat", "lon", "sin_time", "cos_time", "time_of_day",
        "demand_day48", "demand_day48_prev", "demand_day48_next",
        "shift_diff", "shift_ratio", "shift_diff_wmean",
        "gh_mean", "gh_std", "baseline_pred",
        "NumberofLanes", "Temperature",
    ] + [c+"_enc" for c in CAT_COLS]
    print(f"\n   Total features: {len(FEATURE_COLS)}")

    # ── 8. OOF ensemble to find optimal blend weights ────────────
    print("\n[8] Running 5-fold OOF to determine optimal model weights...")
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    n = len(df_train_49)

    oof_et  = np.zeros(n)
    oof_rf  = np.zeros(n)
    oof_lgb = np.zeros(n)

    ET  = ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    RF  = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    LGB = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.03, max_depth=7,
        num_leaves=63, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=-1
    )

    for fold, (tr_idx, val_idx) in enumerate(kf.split(df_train_49), 1):
        X_tr = df_train_49[FEATURE_COLS].iloc[tr_idx]
        y_tr = df_train_49["demand"].iloc[tr_idx]
        X_val = df_train_49[FEATURE_COLS].iloc[val_idx]

        ET.fit(X_tr, y_tr);  oof_et[val_idx]  = ET.predict(X_val)
        RF.fit(X_tr, y_tr);  oof_rf[val_idx]  = RF.predict(X_val)
        LGB.fit(X_tr, y_tr); oof_lgb[val_idx] = LGB.predict(X_val)

        fold_r2_et  = r2_score(df_train_49["demand"].iloc[val_idx], ET.predict(X_val))
        fold_r2_rf  = r2_score(df_train_49["demand"].iloc[val_idx], RF.predict(X_val))
        fold_r2_lgb = r2_score(df_train_49["demand"].iloc[val_idx], LGB.predict(X_val))
        print(f"   Fold {fold}: ET={fold_r2_et:.4f}  RF={fold_r2_rf:.4f}  LGB={fold_r2_lgb:.4f}")

    y49 = df_train_49["demand"].values
    print(f"\n   OOF R2  ET={r2_score(y49, oof_et):.5f}  RF={r2_score(y49, oof_rf):.5f}  LGB={r2_score(y49, oof_lgb):.5f}")

    # Grid search for best blend: final = (1-alpha) * (w_et*et + w_rf*rf + w_lgb*lgb) + alpha * baseline
    print("\n   Grid-searching optimal ensemble + baseline blend weights...")
    best_r2, best_params = -1, None
    baseline_oof = np.clip(df_train_49["demand_day48"].values + df_train_49["shift_diff_wmean"].values, 0, 1)

    for alpha in np.linspace(0.0, 0.15, 7):           # baseline blend weight
        for w_et in np.linspace(0.0, 1.0, 11):
            for w_rf in np.linspace(0.0, 1.0 - w_et, 11):
                w_lgb = 1.0 - w_et - w_rf
                if w_lgb < -1e-5: continue
                ml = w_et * oof_et + w_rf * oof_rf + w_lgb * oof_lgb
                blend = (1 - alpha) * ml + alpha * baseline_oof
                r2 = r2_score(y49, np.clip(blend, 0, 1))
                if r2 > best_r2:
                    best_r2 = r2
                    best_params = (alpha, w_et, w_rf, w_lgb)

    alpha_opt, w_et_opt, w_rf_opt, w_lgb_opt = best_params
    print(f"\n   [OK] Best OOF R2: {best_r2:.5f}")
    print(f"   [OK] Weights -- ET:{w_et_opt:.2f}  RF:{w_rf_opt:.2f}  LGB:{w_lgb_opt:.2f}  Baseline:{alpha_opt:.2f}")

    # -- 9. Retrain on ALL training data --------------------------
    print("\n[9] Retraining all models on full combined training data...")
    X_all = combined_train[FEATURE_COLS]
    y_all = combined_train["demand"]

    ET.fit(X_all, y_all)
    RF.fit(X_all, y_all)
    LGB.fit(X_all, y_all)

    # ── 10. Generate test predictions ────────────────────────────
    print("\n[10] Generating final test predictions...")
    X_test = df_test[FEATURE_COLS]
    p_et   = ET.predict(X_test)
    p_rf   = RF.predict(X_test)
    p_lgb  = LGB.predict(X_test)
    p_base = df_test["baseline_pred"].values

    ml_preds   = w_et_opt * p_et + w_rf_opt * p_rf + w_lgb_opt * p_lgb
    final_preds = np.clip((1 - alpha_opt) * ml_preds + alpha_opt * p_base, 0.0, 1.0)

    # ── 11. Save submission ───────────────────────────────────────
    submission = pd.DataFrame({"Index": test_indices, "demand": final_preds})
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[OK] Submission saved -> {OUTPUT_PATH}")
    print(f"  Shape: {submission.shape}")
    print(submission["demand"].describe().round(5).to_string())

    # ── 12. Zip source files for portal submission ────────────────
    zip_path = r"c:\Users\KIIT\Desktop\flipkartgrid\source_files.zip"
    print(f"\nCreating source package at {zip_path}...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(r"c:\Users\KIIT\Desktop\flipkartgrid\predict.py", "predict.py")
        zipf.write(r"c:\Users\KIIT\Desktop\flipkartgrid\README.txt", "README.txt")
        zipf.write(r"c:\Users\KIIT\Desktop\flipkartgrid\README.md", "README.md")
        zipf.write(r"c:\Users\KIIT\Desktop\flipkartgrid\PROJECT_REPORT.md", "PROJECT_REPORT.md")
        zipf.write(r"c:\Users\KIIT\Desktop\flipkartgrid\Traffic_Demand_Prediction.ipynb", "Traffic_Demand_Prediction.ipynb")
        zipf.write(r"c:\Users\KIIT\Desktop\flipkartgrid\submission.csv", "submission.csv")
        zipf.write(r"c:\Users\KIIT\Desktop\flipkartgrid\index.html", "index.html")
        zipf.write(r"c:\Users\KIIT\Desktop\flipkartgrid\styles.css", "styles.css")
        zipf.write(r"c:\Users\KIIT\Desktop\flipkartgrid\app.js", "app.js")
        zipf.write(r"c:\Users\KIIT\Desktop\flipkartgrid\prepare_dashboard_data.py", "prepare_dashboard_data.py")
    print("Source code package zipped successfully!")

if __name__ == "__main__":
    main()
