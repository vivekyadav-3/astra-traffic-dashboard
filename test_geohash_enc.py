import os
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

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
    
    # Temporal features
    hours, minutes = [], []
    for t in train['timestamp']:
        h, m = map(int, t.split(':'))
        hours.append(h)
        minutes.append(m)
    train['hour'] = hours
    train['minute'] = minutes
    train['time_of_day'] = train['hour'] + train['minute'] / 60.0
    train['sin_time'] = np.sin(2 * np.pi * train['time_of_day'] / 24.0)
    train['cos_time'] = np.cos(2 * np.pi * train['time_of_day'] / 24.0)
    
    # Split day 48 and day 49
    train_48 = train[train['day'] == 48].copy()
    train_49 = train[train['day'] == 49].copy()
    
    # Compute shifts
    overlap_times = ['0:0', '0:15', '0:30', '0:45', '1:0', '1:15', '1:30', '1:45', '2:0']
    train_48_overlap = train_48[train_48['timestamp'].isin(overlap_times)]
    train_49_overlap = train_49[train_49['timestamp'].isin(overlap_times)]
    
    overlap_48 = train_48_overlap.groupby('geohash')['demand'].mean().reset_index()
    overlap_48.columns = ['geohash', 'mean_overlap_48']
    
    overlap_49 = train_49_overlap.groupby('geohash')['demand'].mean().reset_index()
    overlap_49.columns = ['geohash', 'mean_overlap_49']
    
    shifts = overlap_48.merge(overlap_49, on='geohash', how='inner')
    shifts['shift_diff'] = shifts['mean_overlap_49'] - shifts['mean_overlap_48']
    shifts['shift_ratio'] = (shifts['mean_overlap_49'] + 1e-5) / (shifts['mean_overlap_48'] + 1e-5)
    
    geohash_stats = train_48.groupby('geohash')['demand'].mean().reset_index()
    geohash_stats.columns = ['geohash', 'gh_mean']
    global_mean = train_48['demand'].mean()
    
    # Map features to day 49
    df_train_49 = train_49.merge(train_48[['geohash', 'timestamp', 'demand']], on=['geohash', 'timestamp'], how='left', suffixes=('', '_day48'))
    df_train_49 = df_train_49.merge(shifts, on='geohash', how='left')
    df_train_49 = df_train_49.merge(geohash_stats, on='geohash', how='left')
    
    df_train_49['shift_diff'] = df_train_49['shift_diff'].fillna(0.0)
    df_train_49['shift_ratio'] = df_train_49['shift_ratio'].fillna(1.0)
    df_train_49['demand_day48'] = df_train_49['demand_day48'].fillna(df_train_49['gh_mean']).fillna(global_mean)
    
    # Encode categoricals
    cat_cols = ['RoadType', 'LargeVehicles', 'Landmarks', 'Weather']
    for col in cat_cols:
        df_train_49[col] = df_train_49[col].astype(str)
        le = LabelEncoder()
        df_train_49[col + '_enc'] = le.fit_transform(df_train_49[col])
        
    # Also encode geohash itself
    le_gh = LabelEncoder()
    df_train_49['geohash_enc'] = le_gh.fit_transform(df_train_49['geohash'])
        
    med_temp = df_train_49['Temperature'].median()
    med_lanes = df_train_49['NumberofLanes'].median()
    df_train_49['Temperature'] = df_train_49['Temperature'].fillna(med_temp)
    df_train_49['NumberofLanes'] = df_train_49['NumberofLanes'].fillna(med_lanes)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    base_feats = [
        'lat', 'lon', 'sin_time', 'cos_time', 'time_of_day',
        'demand_day48', 'shift_diff', 'shift_ratio', 'gh_mean',
        'NumberofLanes', 'Temperature'
    ] + [c + '_enc' for c in cat_cols]
    
    new_feats = base_feats + ['geohash_enc']
    
    print("\nRunning KFold CV with Base Features:")
    preds_base = np.zeros(len(df_train_49))
    et = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    for tr_idx, val_idx in kf.split(df_train_49):
        X_tr, y_tr = df_train_49[base_feats].iloc[tr_idx], df_train_49['demand'].iloc[tr_idx]
        X_val = df_train_49[base_feats].iloc[val_idx]
        et.fit(X_tr, y_tr)
        preds_base[val_idx] = et.predict(X_val)
    print(f"Base R2: {r2_score(df_train_49['demand'], preds_base):.5f}")
    
    print("\nRunning KFold CV with geohash_enc:")
    preds_new = np.zeros(len(df_train_49))
    for tr_idx, val_idx in kf.split(df_train_49):
        X_tr, y_tr = df_train_49[new_feats].iloc[tr_idx], df_train_49['demand'].iloc[tr_idx]
        X_val = df_train_49[new_feats].iloc[val_idx]
        et.fit(X_tr, y_tr)
        preds_new[val_idx] = et.predict(X_val)
    print(f"With geohash_enc R2: {r2_score(df_train_49['demand'], preds_new):.5f}")

if __name__ == '__main__':
    main()
