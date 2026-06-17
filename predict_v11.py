import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
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
                if bit == 1:
                    lon_interval[0] = mid
                else:
                    lon_interval[1] = mid
            else:
                mid = (lat_interval[0] + lat_interval[1]) / 2
                if bit == 1:
                    lat_interval[0] = mid
                else:
                    lat_interval[1] = mid
            is_even = not is_even
            
    lat = (lat_interval[0] + lat_interval[1]) / 2
    lon = (lon_interval[0] + lon_interval[1]) / 2
    return lat, lon

def main():
    print("Loading datasets...")
    data_dir = r"c:\Users\KIIT\Desktop\flipkartgrid\dataset\dataset"
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    
    # Save a copy of test indices
    test_indices = test['Index'].copy()
    
    # 1. Decode geohashes to lat/lon
    print("Decoding geohashes...")
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
    
    # 2. Map missing test geohashes to nearest known training geohashes
    geohashes_train = set(train['geohash'].unique())
    geohashes_test = set(test['geohash'].unique())
    missing_geohashes = list(geohashes_test - geohashes_train)
    
    if len(missing_geohashes) > 0:
        print(f"Mapping {len(missing_geohashes)} missing test geohashes to nearest train geohashes...")
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
        
    # 3. Add temporal features
    print("Constructing temporal features...")
    def add_time_features(df):
        hours = []
        minutes = []
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
    
    # Split train into day 48 and day 49
    train_48 = train[train['day'] == 48].copy()
    train_49 = train[train['day'] == 49].copy()
    
    # 4. Compute daily shifts using overlapping period (0:0 to 2:0)
    print("Calculating daily shift parameters...")
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
    
    # 5. Compute overall day 48 stats per geohash
    geohash_stats = train_48.groupby('geohash')['demand'].mean().reset_index()
    geohash_stats.columns = ['geohash', 'gh_mean']
    global_mean = train_48['demand'].mean()
    
    # 6. Map features to day 49 training set
    print("Mapping historical demand features to train_49...")
    df_train_49 = train_49.merge(train_48[['geohash', 'timestamp', 'demand']], on=['geohash', 'timestamp'], how='left', suffixes=('', '_day48'))
    df_train_49 = df_train_49.merge(shifts, on='geohash', how='left')
    df_train_49 = df_train_49.merge(geohash_stats, on='geohash', how='left')
    
    df_train_49['shift_diff'] = df_train_49['shift_diff'].fillna(0.0)
    df_train_49['shift_ratio'] = df_train_49['shift_ratio'].fillna(1.0)
    df_train_49['demand_day48'] = df_train_49['demand_day48'].fillna(df_train_49['gh_mean']).fillna(global_mean)
    
    # Map features to day 48 training set (using self stats as dummy/past stats)
    df_train_48 = train_48.copy()
    df_train_48['demand_day48'] = df_train_48['geohash'].map(train_48.groupby('geohash')['demand'].mean())
    df_train_48['shift_diff'] = 0.0
    df_train_48['shift_ratio'] = 1.0
    df_train_48 = df_train_48.merge(geohash_stats, on='geohash', how='left')
    
    # 7. Map features to test set using mapped_geohash to avoid NaNs for missing geohashes
    print("Mapping historical demand features to test...")
    df_test = test.merge(
        train_48[['geohash', 'timestamp', 'demand']], 
        left_on=['mapped_geohash', 'timestamp'], 
        right_on=['geohash', 'timestamp'], 
        how='left', 
        suffixes=('', '_day48_raw')
    )
    # Remove redundant column if present
    if 'geohash_day48_raw' in df_test.columns:
        df_test = df_test.drop(columns=['geohash_day48_raw'])
        
    df_test = df_test.merge(shifts, left_on='mapped_geohash', right_on='geohash', how='left', suffixes=('', '_shift_raw'))
    if 'geohash_shift_raw' in df_test.columns:
        df_test = df_test.drop(columns=['geohash_shift_raw'])
        
    df_test = df_test.merge(geohash_stats, left_on='mapped_geohash', right_on='geohash', how='left', suffixes=('', '_stats_raw'))
    if 'geohash_stats_raw' in df_test.columns:
        df_test = df_test.drop(columns=['geohash_stats_raw'])
        
    df_test['shift_diff'] = df_test['shift_diff'].fillna(0.0)
    df_test['shift_ratio'] = df_test['shift_ratio'].fillna(1.0)
    df_test['demand_day48'] = df_test['demand'].fillna(df_test['gh_mean']).fillna(global_mean)
    
    # Clean up df_test columns (rename target to what we want)
    if 'demand' in df_test.columns:
        df_test = df_test.drop(columns=['demand'])
        
    # Combine training sets
    combined_train = pd.concat([df_train_48, df_train_49], ignore_index=True)
    
    # 8. Encode categorical variables
    print("Encoding categorical variables...")
    cat_cols = ['RoadType', 'LargeVehicles', 'Landmarks', 'Weather']
    for col in cat_cols:
        combined_train[col] = combined_train[col].astype(str)
        df_train_49[col] = df_train_49[col].astype(str)
        df_test[col] = df_test[col].astype(str)
        
        le = LabelEncoder()
        combined_train[col + '_enc'] = le.fit_transform(combined_train[col])
        
        # Safe transform helper
        mapping = dict(zip(le.classes_, range(len(le.classes_))))
        df_train_49[col + '_enc'] = df_train_49[col].map(mapping).fillna(-1).astype(int)
        df_test[col + '_enc'] = df_test[col].map(mapping).fillna(-1).astype(int)
        
    # 9. Handle numeric missing values (median imputation)
    med_temp = combined_train['Temperature'].median()
    med_lanes = combined_train['NumberofLanes'].median()
    
    combined_train['Temperature'] = combined_train['Temperature'].fillna(med_temp)
    combined_train['NumberofLanes'] = combined_train['NumberofLanes'].fillna(med_lanes)
    
    df_train_49['Temperature'] = df_train_49['Temperature'].fillna(med_temp)
    df_train_49['NumberofLanes'] = df_train_49['NumberofLanes'].fillna(med_lanes)
    
    df_test['Temperature'] = df_test['Temperature'].fillna(med_temp)
    df_test['NumberofLanes'] = df_test['NumberofLanes'].fillna(med_lanes)
    
    # Features definition
    feature_cols = [
        'lat', 'lon', 'sin_time', 'cos_time', 'time_of_day',
        'demand_day48', 'shift_diff', 'shift_ratio', 'gh_mean',
        'NumberofLanes', 'Temperature'
    ] + [c + '_enc' for c in cat_cols]
    
    # 10. Training the Ensemble Models
    print("Training Ensemble Model 1: ExtraTrees (Day 49 only)...")
    et_day49 = ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    et_day49.fit(df_train_49[feature_cols], df_train_49['demand'])
    
    print("Training Ensemble Model 2: RandomForest (Day 49 only)...")
    rf_day49 = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf_day49.fit(df_train_49[feature_cols], df_train_49['demand'])
    
    print("Training Ensemble Model 3: ExtraTrees (Combined)...")
    et_comb = ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    et_comb.fit(combined_train[feature_cols], combined_train['demand'])
    
    print("Training Ensemble Model 4: RandomForest (Combined)...")
    rf_comb = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf_comb.fit(combined_train[feature_cols], combined_train['demand'])
    
    # 11. Predicting on test set
    print("Generating predictions...")
    preds_et_day49 = et_day49.predict(df_test[feature_cols])
    preds_rf_day49 = rf_day49.predict(df_test[feature_cols])
    preds_et_comb = et_comb.predict(df_test[feature_cols])
    preds_rf_comb = rf_comb.predict(df_test[feature_cols])
    
    # Weighted average ensemble focusing more on Combined models for daytime extrapolation
    final_preds = 0.20 * preds_et_day49 + 0.10 * preds_rf_day49 + 0.40 * preds_et_comb + 0.30 * preds_rf_comb
    final_preds = np.clip(final_preds, 0.0, 1.0)
    
    # 12. Create Submission File
    submission = pd.DataFrame({
        'Index': test_indices,
        'demand': final_preds
    })
    
    sub_path = r"c:\Users\KIIT\Desktop\flipkartgrid\submission.csv"
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved successfully to {sub_path}!")
    print(submission.head())
    print("Shape:", submission.shape)
    
if __name__ == '__main__':
    main()
