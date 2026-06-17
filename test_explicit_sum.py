import os
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
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
    data_dir = r"c:\Users\KIIT\Desktop\flipkartgrid\dataset\dataset"
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    
    # Decode geohashes
    lats, lons = [], []
    for g in train['geohash']:
        lat, lon = decode_geohash(g)
        lats.append(lat)
        lons.append(lon)
    train['lat'] = lats
    train['lon'] = lons
    
    # Split day 48 and day 49
    train_48 = train[train['day'] == 48].copy()
    train_49 = train[train['day'] == 49].copy()
    
    # Compute shifts
    overlap_times = ['0:0', '0:15', '0:30', '0:45', '1:0', '1:15', '1:30', '1:45', '2:0']
    train_48_overlap = train_48[train_48['timestamp'].isin(overlap_times)]
    train_49_overlap = train_49[train_49['timestamp'].isin(overlap_times)]
    
    # Merge overlap to compute shifts
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
    shifts_df = mean_shift.merge(shift_2, on='geohash', how='left')
    shifts_df = shifts_df.merge(wmean_shift, on='geohash', how='left')
    
    geohash_stats = train_48.groupby('geohash')['demand'].mean().reset_index()
    geohash_stats.columns = ['geohash', 'gh_mean']
    global_mean = train_48['demand'].mean()
    
    # Map features to day 49
    df_train_49 = train_49.merge(train_48[['geohash', 'timestamp', 'demand']], on=['geohash', 'timestamp'], how='left', suffixes=('', '_day48'))
    df_train_49 = df_train_49.merge(shifts_df, on='geohash', how='left')
    df_train_49 = df_train_49.merge(geohash_stats, on='geohash', how='left')
    
    # Fill NAs
    df_train_49['shift_diff_mean'] = df_train_49['shift_diff_mean'].fillna(0.0)
    df_train_49['shift_diff_2_0'] = df_train_49['shift_diff_2_0'].fillna(0.0)
    df_train_49['shift_diff_wmean'] = df_train_49['shift_diff_wmean'].fillna(0.0)
    df_train_49['demand_day48'] = df_train_49['demand_day48'].fillna(df_train_49['gh_mean']).fillna(global_mean)
    
    # Explicit sums
    df_train_49['pred_mean'] = np.clip(df_train_49['demand_day48'] + df_train_49['shift_diff_mean'], 0, 1)
    df_train_49['pred_2_0'] = np.clip(df_train_49['demand_day48'] + df_train_49['shift_diff_2_0'], 0, 1)
    df_train_49['pred_wmean'] = np.clip(df_train_49['demand_day48'] + df_train_49['shift_diff_wmean'], 0, 1)
    
    # Categoricals
    cat_cols = ['RoadType', 'LargeVehicles', 'Landmarks']
    for col in cat_cols:
        df_train_49[col] = df_train_49[col].astype(str)
        le = LabelEncoder()
        df_train_49[col + '_enc'] = le.fit_transform(df_train_49[col])
        
    med_lanes = df_train_49['NumberofLanes'].median()
    df_train_49['NumberofLanes'] = df_train_49['NumberofLanes'].fillna(med_lanes)
    
    # Features
    feats = [
        'lat', 'lon', 'demand_day48', 'shift_diff_mean', 'shift_diff_2_0', 'shift_diff_wmean', 'gh_mean',
        'pred_mean', 'pred_2_0', 'pred_wmean', 'NumberofLanes'
    ] + [c + '_enc' for c in cat_cols]
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_et = np.zeros(len(df_train_49))
    oof_lgb = np.zeros(len(df_train_49))
    
    et = ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    lgb_model = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.05, max_depth=6, num_leaves=31, random_state=42, n_jobs=-1, verbosity=-1)
    
    for tr_idx, val_idx in kf.split(df_train_49):
        X_tr, y_tr = df_train_49[feats].iloc[tr_idx], df_train_49['demand'].iloc[tr_idx]
        X_val = df_train_49[feats].iloc[val_idx]
        
        et.fit(X_tr, y_tr)
        oof_et[val_idx] = et.predict(X_val)
        
        lgb_model.fit(X_tr, y_tr)
        oof_lgb[val_idx] = lgb_model.predict(X_val)
        
    print(f"ET OOF R2 with explicit sums: {r2_score(df_train_49['demand'], oof_et):.5f}")
    print(f"LGB OOF R2 with explicit sums: {r2_score(df_train_49['demand'], oof_lgb):.5f}")
    
    # Best Blend
    ml_blend = 0.3 * oof_et + 0.7 * oof_lgb
    print(f"ML Blend OOF R2 with explicit sums: {r2_score(df_train_49['demand'], ml_blend):.5f}")

if __name__ == '__main__':
    main()
