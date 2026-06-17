import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

def decode_geohash(geohash):
    base32 = "0123456789bcdefghjkmnpqrstuvwxyz"
    base32_map = {char: i for i, char in enumerate(base32)}
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
                else: lon_interval[1] = mid
            else:
                mid = (lat_interval[0] + lat_interval[1]) / 2
                if bit == 1: lat_interval[0] = mid
                else: lat_interval[1] = mid
            is_even = not is_even
    return (lat_interval[0] + lat_interval[1]) / 2, (lon_interval[0] + lon_interval[1]) / 2

def main():
    print("Loading data...")
    data_dir = r"c:\Users\KIIT\Desktop\flipkartgrid\dataset\dataset"
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test  = pd.read_csv(os.path.join(data_dir, "test.csv"))
    test_indices = test['Index'].copy()

    # Spatial coords
    def add_coords(df):
        lats, lons = [], []
        for g in df['geohash']:
            lat, lon = decode_geohash(g)
            lats.append(lat)
            lons.append(lon)
        df['lat'] = lats
        df['lon'] = lons
        return df
    train = add_coords(train)
    test = add_coords(test)

    # Map missing test geohashes to nearest train geohash
    gh_train = set(train['geohash'].unique())
    missing   = list(set(test['geohash'].unique()) - gh_train)
    gh_map    = {}
    if missing:
        print(f"Mapping {len(missing)} missing geohashes...")
        tc  = [(g, *decode_geohash(g)) for g in gh_train]
        tdf = pd.DataFrame(tc, columns=['geohash','lat','lon'])
        for g in missing:
            lat, lon = decode_geohash(g)
            tdf['d'] = np.sqrt((tdf['lat']-lat)**2+(tdf['lon']-lon)**2)
            gh_map[g] = tdf.sort_values('d').iloc[0]['geohash']
    test['mapped_gh'] = test['geohash'].map(lambda x: gh_map.get(x, x))

    train_48 = train[train['day']==48].copy()
    train_49 = train[train['day']==49].copy()

    # Compute shifts
    print("Computing daily shifts...")
    overlap_times = ['0:0', '0:15', '0:30', '0:45', '1:0', '1:15', '1:30', '1:45', '2:0']
    train_48_overlap = train_48[train_48['timestamp'].isin(overlap_times)]
    train_49_overlap = train_49[train_49['timestamp'].isin(overlap_times)]
    
    merged_ov = train_49_overlap.merge(
        train_48_overlap[['geohash', 'timestamp', 'demand']],
        on=['geohash', 'timestamp'],
        suffixes=('_49', '_48')
    )
    merged_ov['diff'] = merged_ov['demand_49'] - merged_ov['demand_48']
    
    # 1. Mean shift
    mean_shift = merged_ov.groupby('geohash')['diff'].mean().reset_index().rename(columns={'diff': 'shift_diff_mean'})
    
    # 2. Shift at 2:0
    shift_2 = merged_ov[merged_ov['timestamp'] == '2:0'][['geohash', 'diff']].rename(columns={'diff': 'shift_diff_2_0'})
    
    # 3. Weighted mean shift (exponential weight)
    time_weights = {
        '0:0': 0.1, '0:15': 0.2, '0:30': 0.3, '0:45': 0.5,
        '1:0': 0.8, '1:15': 1.2, '1:30': 1.8, '1:45': 2.7, '2:0': 4.0
    }
    merged_ov['weight'] = merged_ov['timestamp'].map(time_weights)
    wmean_shift = merged_ov.groupby('geohash').apply(
        lambda g: np.average(g['diff'], weights=g['weight'])
    ).reset_index(name='shift_diff_wmean')
    
    # Merge all shifts
    shifts = mean_shift.merge(shift_2, on='geohash', how='left')
    shifts = shifts.merge(wmean_shift, on='geohash', how='left')
    
    global_shift_mean = shifts['shift_diff_mean'].mean()
    global_shift_2_0 = shifts['shift_diff_2_0'].mean()
    global_shift_wmean = shifts['shift_diff_wmean'].mean()

    # Geohash-level stats from day 48
    gh_stats = train_48.groupby('geohash')['demand'].agg(['mean','std']).reset_index()
    gh_stats.columns = ['geohash','gh_mean','gh_std']
    global_mean = train_48['demand'].mean()

    # Feature engineering function
    def build(df, geohash_col):
        d = df.copy()
        
        # Day 48 demand at exact same timestamp
        d48_demand = train_48[['geohash','timestamp','demand']].rename(
            columns={'demand':'demand_day48','geohash':'_g'})
        d = d.merge(d48_demand, left_on=[geohash_col,'timestamp'],
                    right_on=['_g','timestamp'], how='left').drop(columns=['_g'],errors='ignore')

        # Shifts
        d = d.merge(shifts.rename(columns={'geohash':'_g'}),
                    left_on=geohash_col, right_on='_g', how='left').drop(columns=['_g'],errors='ignore')
        d['shift_diff_mean'] = d['shift_diff_mean'].fillna(global_shift_mean)
        d['shift_diff_2_0'] = d['shift_diff_2_0'].fillna(global_shift_2_0)
        d['shift_diff_wmean'] = d['shift_diff_wmean'].fillna(global_shift_wmean)

        # Geohash stats
        d = d.merge(gh_stats.rename(columns={'geohash':'_g'}),
                    left_on=geohash_col, right_on='_g', how='left').drop(columns=['_g'],errors='ignore')

        # Fill demand_day48 with geohash mean / global mean
        d['demand_day48'] = d['demand_day48'].fillna(d['gh_mean']).fillna(global_mean)
        
        # Explicit predictions
        d['pred_mean'] = np.clip(d['demand_day48'] + d['shift_diff_mean'], 0.0, 1.0)
        d['pred_2_0'] = np.clip(d['demand_day48'] + d['shift_diff_2_0'], 0.0, 1.0)
        d['pred_wmean'] = np.clip(d['demand_day48'] + d['shift_diff_wmean'], 0.0, 1.0)

        return d

    print("Building features...")
    df49  = build(train_49, 'geohash')
    df_test = build(test, 'mapped_gh')

    # Encode categoricals
    cat_cols = ['RoadType','LargeVehicles','Landmarks']
    le_maps  = {}
    for col in cat_cols:
        df49[col] = df49[col].astype(str)
        le = LabelEncoder()
        df49[col+'_enc'] = le.fit_transform(df49[col])
        le_maps[col] = dict(zip(le.classes_, range(len(le.classes_))))
        
        df_test[col] = df_test[col].astype(str)
        df_test[col+'_enc'] = df_test[col].map(le_maps[col]).fillna(-1).astype(int)

    med_l = df49['NumberofLanes'].median()
    df49['NumberofLanes'] = df49['NumberofLanes'].fillna(med_l)
    df_test['NumberofLanes'] = df_test['NumberofLanes'].fillna(med_l)

    # Feature set
    feats = [
        'lat','lon','demand_day48','shift_diff_mean','shift_diff_2_0','shift_diff_wmean','gh_mean','gh_std',
        'pred_mean','pred_2_0','pred_wmean','NumberofLanes'
    ] + [c+'_enc' for c in cat_cols]
    print(f"Feature count: {len(feats)}")

    X_49 = df49[feats]
    y_49 = df49['demand']
    X_test = df_test[feats]

    # --- Ensemble: ET + RF + LGB on Day 49 only ---
    print("Training models...")

    # ExtraTrees
    et = ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    et.fit(X_49, y_49)
    print("  ExtraTrees done")

    # RandomForest
    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_49, y_49)
    print("  RandomForest done")

    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=150, learning_rate=0.05, max_depth=6, num_leaves=31,
        random_state=42, n_jobs=-1, verbosity=-1
    )
    lgb_model.fit(X_49, y_49)
    print("  LightGBM done")

    # Generate predictions
    print("Generating predictions...")
    p_et = et.predict(X_test)
    p_rf = rf.predict(X_test)
    p_lgb = lgb_model.predict(X_test)

    # ML Blend
    preds_ml = 0.20 * p_et + 0.08 * p_rf + 0.72 * p_lgb
    
    # Pure Blend
    pred_pure = 0.5 * df_test['pred_mean'] + 0.5 * df_test['pred_wmean']
    
    # Final Blend
    final = 0.95 * preds_ml + 0.05 * pred_pure
    final = np.clip(final, 0.0, 1.0)

    sub = pd.DataFrame({'Index': test_indices, 'demand': final})
    out = r"c:\Users\KIIT\Desktop\flipkartgrid\submission.csv"
    sub.to_csv(out, index=False)
    print(f"Submission saved: {out}")
    print("Shape:", sub.shape)
    print(sub.head())

if __name__ == '__main__':
    main()
