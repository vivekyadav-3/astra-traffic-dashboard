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
    
    # Map features to day 48
    df_train_48 = train_48.copy()
    df_train_48['demand_day48'] = df_train_48['geohash'].map(train_48.groupby('geohash')['demand'].mean())
    df_train_48['shift_diff'] = 0.0
    df_train_48['shift_ratio'] = 1.0
    df_train_48 = df_train_48.merge(geohash_stats, on='geohash', how='left')
    
    combined_train = pd.concat([df_train_48, df_train_49], ignore_index=True)
    
    # Encode categoricals
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
    
    # Out of fold validation on Day 49 geohashes
    # Note: since the test set has unseen geohashes and future timestamps,
    # spatial KFold (splitting by geohashes) is the most realistic CV!
    geohashes = df_train_49['geohash'].unique()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_et_49 = np.zeros(len(df_train_49))
    oof_rf_49 = np.zeros(len(df_train_49))
    oof_et_comb = np.zeros(len(df_train_49))
    oof_rf_comb = np.zeros(len(df_train_49))
    
    for tr_gh_idx, val_gh_idx in kf.split(geohashes):
        tr_ghs = geohashes[tr_gh_idx]
        val_ghs = geohashes[val_gh_idx]
        
        # Split train/val by geohash
        train_fold_49 = df_train_49[df_train_49['geohash'].isin(tr_ghs)]
        train_fold_comb = combined_train[combined_train['geohash'].isin(tr_ghs)]
        val_fold = df_train_49[df_train_49['geohash'].isin(val_ghs)]
        val_indices = val_fold.index
        
        # Train models
        et_49 = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        et_49.fit(train_fold_49[feature_cols], train_fold_49['demand'])
        oof_et_49[val_indices] = et_49.predict(val_fold[feature_cols])
        
        rf_49 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_49.fit(train_fold_49[feature_cols], train_fold_49['demand'])
        oof_rf_49[val_indices] = rf_49.predict(val_fold[feature_cols])
        
        et_comb = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        et_comb.fit(train_fold_comb[feature_cols], train_fold_comb['demand'])
        oof_et_comb[val_indices] = et_comb.predict(val_fold[feature_cols])
        
        rf_comb = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_comb.fit(train_fold_comb[feature_cols], train_fold_comb['demand'])
        oof_rf_comb[val_indices] = rf_comb.predict(val_fold[feature_cols])
        
    print(f"ET Day 49 OOF R2: {r2_score(df_train_49['demand'], oof_et_49):.5f}")
    print(f"RF Day 49 OOF R2: {r2_score(df_train_49['demand'], oof_rf_49):.5f}")
    print(f"ET Combined OOF R2: {r2_score(df_train_49['demand'], oof_et_comb):.5f}")
    print(f"RF Combined OOF R2: {r2_score(df_train_49['demand'], oof_rf_comb):.5f}")
    
    # Original blend: 0.3 * ET_Day49 + 0.2 * RF_Day49 + 0.3 * ET_Comb + 0.2 * RF_Comb
    orig_blend = 0.3 * oof_et_49 + 0.2 * oof_rf_49 + 0.3 * oof_et_comb + 0.2 * oof_rf_comb
    print(f"Original Blend OOF R2: {r2_score(df_train_49['demand'], orig_blend):.5f}")
    
    # Grid search for the best spatial CV blend weights
    best_r2 = -1
    best_weights = None
    for w1 in np.linspace(0, 1, 11):
        for w2 in np.linspace(0, 1 - w1, 11):
            for w3 in np.linspace(0, 1 - w1 - w2, 11):
                w4 = 1.0 - w1 - w2 - w3
                if w4 < -1e-5:
                    continue
                blend = w1 * oof_et_49 + w2 * oof_rf_49 + w3 * oof_et_comb + w4 * oof_rf_comb
                r2 = r2_score(df_train_49['demand'], blend)
                if r2 > best_r2:
                    best_r2 = r2
                    best_weights = (w1, w2, w3, w4)
                    
    print(f"\nBest Blend Weights found:")
    print(f"  ET Day 49: {best_weights[0]:.2f}")
    print(f"  RF Day 49: {best_weights[1]:.2f}")
    print(f"  ET Comb  : {best_weights[2]:.2f}")
    print(f"  RF Comb  : {best_weights[3]:.2f}")
    print(f"  Best OOF R2: {best_r2:.5f}")

if __name__ == '__main__':
    main()
