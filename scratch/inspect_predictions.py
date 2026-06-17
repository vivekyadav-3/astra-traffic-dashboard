import os
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

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
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    
    # Decoding geohashes
    lats, lons = [], []
    for g in train['geohash']:
        lat, lon = decode_geohash(g)
        lats.append(lat)
        lons.append(lon)
    train['lat'] = lats
    train['lon'] = lons
    
    lats, lons = [], []
    for g in test['geohash']:
        lat, lon = decode_geohash(g)
        lats.append(lat)
        lons.append(lon)
    test['lat'] = lats
    test['lon'] = lons
    
    # Map missing test geohashes
    geohashes_train = set(train['geohash'].unique())
    geohashes_test = set(test['geohash'].unique())
    missing_geohashes = list(geohashes_test - geohashes_train)
    
    if len(missing_geohashes) > 0:
        train_coords = []
        for gh in geohashes_train:
            lat, lon = decode_geohash(gh)
            train_coords.append((gh, lat, lon))
        train_df = pd.DataFrame(train_coords, columns=['geohash', 'lat', 'lon'])
        
        gh_mapping = {}
        for gh in missing_geohashes:
            lat, lon = decode_geohash(gh)
            train_df['dist'] = np.sqrt((train_df['lat'] - lat)**2 + (train_df['lon'] - lon)**2)
            nearest_gh = train_df.sort_values('dist').iloc[0]['geohash']
            gh_mapping[gh] = nearest_gh
        test['mapped_geohash'] = test['geohash'].map(lambda x: gh_mapping.get(x, x))
    else:
        test['mapped_geohash'] = test['geohash']
        
    # Temporal features
    def add_time_features(df):
        hours, minutes = [], []
        for t in df['timestamp']:
            h, m = map(int, t.split(':'))
            hours.append(h)
            minutes.append(m)
        df['hour'] = hours
        df['minute'] = minutes
        df['time_of_day'] = df['hour'] + df['minute'] / 60.0
        df['sin_time'] = np.sin(2 * np.pi * df['time_of_day'] / 24.0)
        df['cos_time'] = np.cos(2 * np.pi * df['time_of_day'] / 24.0)
        return df

    train = add_time_features(train)
    test = add_time_features(test)
    
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
    
    # Map train_49
    df_train_49 = train_49.merge(train_48[['geohash', 'timestamp', 'demand']], on=['geohash', 'timestamp'], how='left', suffixes=('', '_day48'))
    df_train_49 = df_train_49.merge(shifts, on='geohash', how='left')
    df_train_49 = df_train_49.merge(geohash_stats, on='geohash', how='left')
    
    df_train_49['shift_diff'] = df_train_49['shift_diff'].fillna(0.0)
    df_train_49['shift_ratio'] = df_train_49['shift_ratio'].fillna(1.0)
    df_train_49['demand_day48'] = df_train_49['demand_day48'].fillna(df_train_49['gh_mean']).fillna(global_mean)
    
    # Map train_48
    df_train_48 = train_48.copy()
    df_train_48['demand_day48'] = df_train_48['geohash'].map(train_48.groupby('geohash')['demand'].mean())
    df_train_48['shift_diff'] = 0.0
    df_train_48['shift_ratio'] = 1.0
    df_train_48 = df_train_48.merge(geohash_stats, on='geohash', how='left')
    
    # Map test
    df_test = test.merge(
        train_48[['geohash', 'timestamp', 'demand']], 
        left_on=['mapped_geohash', 'timestamp'], 
        right_on=['geohash', 'timestamp'], 
        how='left', 
        suffixes=('', '_day48_raw')
    )
    if 'geohash_day48_raw' in df_test.columns:
        df_test = df_test.drop(columns=['geohash_day48_raw'])
    df_test = df_test.merge(shifts, left_on='mapped_geohash', right_on='geohash', how='left')
    df_test = df_test.merge(geohash_stats, left_on='mapped_geohash', right_on='geohash', how='left')
    
    df_test['shift_diff'] = df_test['shift_diff'].fillna(0.0)
    df_test['shift_ratio'] = df_test['shift_ratio'].fillna(1.0)
    df_test['demand_day48'] = df_test['demand'].fillna(df_test['gh_mean']).fillna(global_mean)
    if 'demand' in df_test.columns:
        df_test = df_test.drop(columns=['demand'])
        
    combined_train = pd.concat([df_train_48, df_train_49], ignore_index=True)
    
    # Categoricals
    cat_cols = ['RoadType', 'LargeVehicles', 'Landmarks', 'Weather']
    for col in cat_cols:
        combined_train[col] = combined_train[col].astype(str)
        df_train_49[col] = df_train_49[col].astype(str)
        df_test[col] = df_test[col].astype(str)
        le = LabelEncoder()
        combined_train[col + '_enc'] = le.fit_transform(combined_train[col])
        mapping = dict(zip(le.classes_, range(len(le.classes_))))
        df_train_49[col + '_enc'] = df_train_49[col].map(mapping).fillna(-1).astype(int)
        df_test[col + '_enc'] = df_test[col].map(mapping).fillna(-1).astype(int)
        
    med_temp = combined_train['Temperature'].median()
    med_lanes = combined_train['NumberofLanes'].median()
    combined_train['Temperature'] = combined_train['Temperature'].fillna(med_temp)
    combined_train['NumberofLanes'] = combined_train['NumberofLanes'].fillna(med_lanes)
    df_train_49['Temperature'] = df_train_49['Temperature'].fillna(med_temp)
    df_train_49['NumberofLanes'] = df_train_49['NumberofLanes'].fillna(med_lanes)
    df_test['Temperature'] = df_test['Temperature'].fillna(med_temp)
    df_test['NumberofLanes'] = df_test['NumberofLanes'].fillna(med_lanes)
    
    feature_cols = [
        'lat', 'lon', 'sin_time', 'cos_time', 'time_of_day',
        'demand_day48', 'shift_diff', 'shift_ratio', 'gh_mean',
        'NumberofLanes', 'Temperature'
    ] + [c + '_enc' for c in cat_cols]
    
    print("Training models...")
    et_day49 = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    et_day49.fit(df_train_49[feature_cols], df_train_49['demand'])
    
    et_comb = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    et_comb.fit(combined_train[feature_cols], combined_train['demand'])
    
    # Train residual model on Day 49 only
    # residual = demand_day49 - demand_day48
    df_train_49['residual'] = df_train_49['demand'] - df_train_49['demand_day48']
    
    # We train ExtraTrees to predict the residual
    # We exclude demand_day48 from features for residual model to avoid overdependence/leakage,
    # or keep it? Let's check with it included first.
    et_res_day49 = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    et_res_day49.fit(df_train_49[feature_cols], df_train_49['residual'])
    
    # Let's pick a geohash with substantial daytime demand variance
    gh_counts = train_48.groupby('geohash')['demand'].std().sort_values(ascending=False)
    target_gh = gh_counts.index[0]
    print(f"\nAnalyzing predictions for geohash {target_gh}:")
    
    # Get test rows for this geohash
    gh_test = df_test[df_test['geohash'] == target_gh].sort_values('time_of_day')
    
    preds_day49 = et_day49.predict(gh_test[feature_cols])
    preds_comb = et_comb.predict(gh_test[feature_cols])
    
    # Predict residual and reconstruct demand
    preds_res = et_res_day49.predict(gh_test[feature_cols])
    preds_res_demand = np.clip(gh_test['demand_day48'] + preds_res, 0.0, 1.0)
    
    # Get the day 48 values for comparison
    gh_48 = train_48[train_48['geohash'] == target_gh].sort_values('time_of_day')
    
    print(f"gh_test shape before merge: {gh_test.shape}")
    print(f"gh_48 shape before merge: {gh_48.shape}")
    
    # Let's perform a clean merge using only timestamp to avoid duplicate geohash columns confusion
    gh_test_clean = gh_test.copy()
    gh_test_clean['demand_48'] = gh_test_clean['timestamp'].map(dict(zip(gh_48['timestamp'], gh_48['demand'])))
    
    result = pd.DataFrame({
        'time': gh_test_clean['timestamp'],
        'demand_48': gh_test_clean['demand_48'],
        'day49_only': preds_day49,
        'combined': preds_comb,
        'residual_model': preds_res_demand,
        'pure_shift': gh_test_clean['demand_48'] + gh_test_clean['shift_diff']
    })
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(result.to_string())

if __name__ == '__main__':
    main()
