import os
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
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
    
    lats, lons = [], []
    for g in train['geohash']:
        lat, lon = decode_geohash(g)
        lats.append(lat); lons.append(lon)
    train['lat'] = lats; train['lon'] = lons
    
    def add_time_features(df):
        hours, minutes = [], []
        for t in df['timestamp']:
            h, m = map(int, t.split(':'))
            hours.append(h); minutes.append(m)
        df['hour'] = hours; df['minute'] = minutes
        df['time_of_day'] = df['hour'] + df['minute'] / 60.0
        df['sin_time'] = np.sin(2 * np.pi * df['time_of_day'] / 24.0)
        df['cos_time'] = np.cos(2 * np.pi * df['time_of_day'] / 24.0)
        return df

    train = add_time_features(train)
    
    train_48 = train[train['day'] == 48].copy()
    train_49 = train[train['day'] == 49].copy()
    
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
    
    df_train_49 = train_49.merge(train_48[['geohash', 'timestamp', 'demand']], on=['geohash', 'timestamp'], how='left', suffixes=('', '_day48'))
    df_train_49 = df_train_49.merge(shifts, on='geohash', how='left')
    df_train_49 = df_train_49.merge(geohash_stats, on='geohash', how='left')
    
    df_train_49['shift_diff'] = df_train_49['shift_diff'].fillna(0.0)
    df_train_49['shift_ratio'] = df_train_49['shift_ratio'].fillna(1.0)
    df_train_49['demand_day48'] = df_train_49['demand_day48'].fillna(df_train_49['gh_mean']).fillna(global_mean)
    
    df_train_48 = train_48.copy()
    df_train_48['demand_day48'] = df_train_48['geohash'].map(train_48.groupby('geohash')['demand'].mean())
    df_train_48['shift_diff'] = 0.0; df_train_48['shift_ratio'] = 1.0
    df_train_48 = df_train_48.merge(geohash_stats, on='geohash', how='left')
    
    combined_train = pd.concat([df_train_48, df_train_49], ignore_index=True)
    
    cat_cols = ['RoadType', 'LargeVehicles', 'Landmarks', 'Weather']
    for col in cat_cols:
        combined_train[col] = combined_train[col].astype(str)
        df_train_49[col] = df_train_49[col].astype(str)
        le = LabelEncoder()
        combined_train[col + '_enc'] = le.fit_transform(combined_train[col])
        mapping = dict(zip(le.classes_, range(len(le.classes_))))
        df_train_49[col + '_enc'] = df_train_49[col].map(mapping).fillna(-1).astype(int)
        
    med_temp = combined_train['Temperature'].median()
    med_lanes = combined_train['NumberofLanes'].median()
    combined_train['Temperature'] = combined_train['Temperature'].fillna(med_temp)
    combined_train['NumberofLanes'] = combined_train['NumberofLanes'].fillna(med_lanes)
    df_train_49['Temperature'] = df_train_49['Temperature'].fillna(med_temp)
    df_train_49['NumberofLanes'] = df_train_49['NumberofLanes'].fillna(med_lanes)
    
    feature_cols = [
        'lat', 'lon', 'sin_time', 'cos_time', 'time_of_day',
        'demand_day48', 'shift_diff', 'shift_ratio', 'gh_mean',
        'NumberofLanes', 'Temperature'
    ] + [c + '_enc' for c in cat_cols]
    
    geohashes = df_train_49['geohash'].unique()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # We will test default vs regularized parameters
    param_sets = [
        {'min_samples_leaf': 1, 'max_features': 1.0},
        {'min_samples_leaf': 2, 'max_features': 0.8},
        {'min_samples_leaf': 3, 'max_features': 0.7},
        {'min_samples_leaf': 5, 'max_features': 0.6}
    ]
    
    for p in param_sets:
        print(f"\nEvaluating with parameters: {p}")
        oof_et_49 = np.zeros(len(df_train_49))
        oof_rf_49 = np.zeros(len(df_train_49))
        oof_et_comb = np.zeros(len(df_train_49))
        oof_rf_comb = np.zeros(len(df_train_49))
        
        for tr_gh_idx, val_gh_idx in kf.split(geohashes):
            tr_ghs = geohashes[tr_gh_idx]
            val_ghs = geohashes[val_gh_idx]
            
            train_fold_49 = df_train_49[df_train_49['geohash'].isin(tr_ghs)]
            train_fold_comb = combined_train[combined_train['geohash'].isin(tr_ghs)]
            val_fold = df_train_49[df_train_49['geohash'].isin(val_ghs)]
            val_indices = val_fold.index
            
            et_49 = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1, **p)
            et_49.fit(train_fold_49[feature_cols], train_fold_49['demand'])
            oof_et_49[val_indices] = et_49.predict(val_fold[feature_cols])
            
            rf_49 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, **p)
            rf_49.fit(train_fold_49[feature_cols], train_fold_49['demand'])
            oof_rf_49[val_indices] = rf_49.predict(val_fold[feature_cols])
            
            et_comb = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1, **p)
            et_comb.fit(train_fold_comb[feature_cols], train_fold_comb['demand'])
            oof_et_comb[val_indices] = et_comb.predict(val_fold[feature_cols])
            
            rf_comb = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, **p)
            rf_comb.fit(train_fold_comb[feature_cols], train_fold_comb['demand'])
            oof_rf_comb[val_indices] = rf_comb.predict(val_fold[feature_cols])
            
        orig_blend = 0.3 * oof_et_49 + 0.2 * oof_rf_49 + 0.3 * oof_et_comb + 0.2 * oof_rf_comb
        print(f"  ET Day 49 OOF R2: {r2_score(df_train_49['demand'], oof_et_49):.5f}")
        print(f"  RF Day 49 OOF R2: {r2_score(df_train_49['demand'], oof_rf_49):.5f}")
        print(f"  ET Comb OOF R2:   {r2_score(df_train_49['demand'], oof_et_comb):.5f}")
        print(f"  RF Comb OOF R2:   {r2_score(df_train_49['demand'], oof_rf_comb):.5f}")
        print(f"  Blend OOF R2:     {r2_score(df_train_49['demand'], orig_blend):.5f}")

if __name__ == '__main__':
    main()
