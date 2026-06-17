import pandas as pd
import numpy as np
import os
import zipfile
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
            print(f"Mapped {gh} -> {nearest_gh}")
            
        test['mapped_geohash'] = test['geohash'].map(lambda x: gh_mapping.get(x, x))
    else:
        test['mapped_geohash'] = test['geohash']
        
    # 3. Add temporal features (for sorting and grouping)
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
    
    # Merge overlap to compute shifts
    merged_ov = train_49_overlap.merge(
        train_48_overlap[['geohash', 'timestamp', 'demand']],
        on=['geohash', 'timestamp'],
        suffixes=('_49', '_48')
    )
    merged_ov['diff'] = merged_ov['demand_49'] - merged_ov['demand_48']
    
    # Compute multiple shift estimators
    # a. Mean shift
    mean_shift = merged_ov.groupby('geohash')['diff'].mean().reset_index().rename(columns={'diff': 'shift_diff_mean'})
    # b. Last timestamp shift (at 2:0)
    shift_2 = merged_ov[merged_ov['timestamp'] == '2:0'][['geohash', 'diff']].rename(columns={'diff': 'shift_diff_2_0'})
    # c. Weighted mean shift (exponential weighting towards 2:0)
    time_weights = {
        '0:0': 0.1, '0:15': 0.2, '0:30': 0.3, '0:45': 0.5,
        '1:0': 0.8, '1:15': 1.2, '1:30': 1.8, '1:45': 2.7, '2:0': 4.0
    }
    merged_ov['weight'] = merged_ov['timestamp'].map(time_weights)
    wmean_shift = merged_ov.groupby('geohash').apply(
        lambda g: np.average(g['diff'], weights=g['weight']),
        include_groups=False
    ).reset_index(name='shift_diff_wmean')
    
    # Merge shift estimators
    shifts_df = mean_shift.merge(shift_2, on='geohash', how='left')
    shifts_df = shifts_df.merge(wmean_shift, on='geohash', how='left')
    
    # Compute shift ratio
    overlap_48 = train_48_overlap.groupby('geohash')['demand'].mean().reset_index()
    overlap_48.columns = ['geohash', 'mean_overlap_48']
    overlap_49 = train_49_overlap.groupby('geohash')['demand'].mean().reset_index()
    overlap_49.columns = ['geohash', 'mean_overlap_49']
    
    ratios = overlap_48.merge(overlap_49, on='geohash', how='inner')
    ratios['shift_ratio'] = (ratios['mean_overlap_49'] + 1e-5) / (ratios['mean_overlap_48'] + 1e-5)
    
    shifts_df = shifts_df.merge(ratios[['geohash', 'shift_ratio']], on='geohash', how='left')
    
    # 5. Compute overall day 48 stats per geohash
    geohash_stats = train_48.groupby('geohash')['demand'].mean().reset_index()
    geohash_stats.columns = ['geohash', 'gh_mean']
    global_mean = train_48['demand'].mean()
    
    # 6. Map features to day 49 training set
    print("Mapping historical demand features to train_49...")
    df_train_49 = train_49.merge(train_48[['geohash', 'timestamp', 'demand']], on=['geohash', 'timestamp'], how='left', suffixes=('', '_day48'))
    df_train_49 = df_train_49.merge(shifts_df, on='geohash', how='left')
    df_train_49 = df_train_49.merge(geohash_stats, on='geohash', how='left')
    
    df_train_49['shift_diff_mean'] = df_train_49['shift_diff_mean'].fillna(0.0)
    df_train_49['shift_diff_2_0'] = df_train_49['shift_diff_2_0'].fillna(0.0)
    df_train_49['shift_diff_wmean'] = df_train_49['shift_diff_wmean'].fillna(0.0)
    df_train_49['shift_ratio'] = df_train_49['shift_ratio'].fillna(1.0)
    df_train_49['demand_day48'] = df_train_49['demand_day48'].fillna(df_train_49['gh_mean']).fillna(global_mean)
    
    # Compute residual target for Day 49
    df_train_49['residual'] = df_train_49['demand'] - df_train_49['demand_day48']
    
    # 7. Map features to test set using mapped_geohash to avoid NaNs for missing geohashes
    print("Mapping historical demand features to test...")
    df_test = test.merge(
        train_48[['geohash', 'timestamp', 'demand']], 
        left_on=['mapped_geohash', 'timestamp'], 
        right_on=['geohash', 'timestamp'], 
        how='left', 
        suffixes=('', '_day48_raw')
    )
    if 'geohash_day48_raw' in df_test.columns:
        df_test = df_test.drop(columns=['geohash_day48_raw'])
        
    df_test = df_test.merge(shifts_df, left_on='mapped_geohash', right_on='geohash', how='left', suffixes=('', '_shift_raw'))
    if 'geohash_shift_raw' in df_test.columns:
        df_test = df_test.drop(columns=['geohash_shift_raw'])
        
    df_test = df_test.merge(geohash_stats, left_on='mapped_geohash', right_on='geohash', how='left', suffixes=('', '_stats_raw'))
    if 'geohash_stats_raw' in df_test.columns:
        df_test = df_test.drop(columns=['geohash_stats_raw'])
        
    df_test['shift_diff_mean'] = df_test['shift_diff_mean'].fillna(0.0)
    df_test['shift_diff_2_0'] = df_test['shift_diff_2_0'].fillna(0.0)
    df_test['shift_diff_wmean'] = df_test['shift_diff_wmean'].fillna(0.0)
    df_test['shift_ratio'] = df_test['shift_ratio'].fillna(1.0)
    df_test['demand_day48'] = df_test['demand'].fillna(df_test['gh_mean']).fillna(global_mean)
    
    if 'demand' in df_test.columns:
        df_test = df_test.drop(columns=['demand'])
        
    # 8. Encode categorical variables
    print("Encoding categorical variables...")
    cat_cols = ['RoadType', 'LargeVehicles', 'Landmarks', 'Weather']
    for col in cat_cols:
        df_train_49[col] = df_train_49[col].astype(str)
        df_test[col] = df_test[col].astype(str)
        
        le = LabelEncoder()
        # Fit on all train+test categoricals to ensure safe mapping
        all_cats = pd.concat([train[col].astype(str), test[col].astype(str)], ignore_index=True)
        le.fit(all_cats)
        
        df_train_49[col + '_enc'] = le.transform(df_train_49[col])
        df_test[col + '_enc'] = le.transform(df_test[col])
        
    # 9. Handle numeric missing values (median imputation)
    med_temp = train_49['Temperature'].median()
    med_lanes = train_49['NumberofLanes'].median()
    
    df_train_49['Temperature'] = df_train_49['Temperature'].fillna(med_temp)
    df_train_49['NumberofLanes'] = df_train_49['NumberofLanes'].fillna(med_lanes)
    
    df_test['Temperature'] = df_test['Temperature'].fillna(med_temp)
    df_test['NumberofLanes'] = df_test['NumberofLanes'].fillna(med_lanes)
    
    # Residual Model Features
    feature_cols = [
        'lat', 'lon', 'demand_day48', 'shift_diff_mean', 'shift_diff_2_0', 'shift_diff_wmean', 'shift_ratio', 'gh_mean',
        'NumberofLanes', 'Temperature'
    ] + [c + '_enc' for c in cat_cols]
    
    # 10. Training the Residual Models
    print("Training Residual Model 1: ExtraTrees (Day 49)...")
    et_res = ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    et_res.fit(df_train_49[feature_cols], df_train_49['residual'])
    
    print("Training Residual Model 2: RandomForest (Day 49)...")
    rf_res = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf_res.fit(df_train_49[feature_cols], df_train_49['residual'])
    
    print("Training Residual Model 3: LightGBM (Day 49)...")
    lgb_res = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.05, max_depth=6, num_leaves=31, random_state=42, n_jobs=-1, verbosity=-1)
    lgb_res.fit(df_train_49[feature_cols], df_train_49['residual'])
    
    # 11. Predicting Residual on Test Set
    print("Generating predictions...")
    preds_et = et_res.predict(df_test[feature_cols])
    preds_rf = rf_res.predict(df_test[feature_cols])
    preds_lgb = lgb_res.predict(df_test[feature_cols])
    
    # Blended Residual (optimized weights: ET=0.05, RF=0.29, LGB=0.66)
    blend_res = 0.05 * preds_et + 0.29 * preds_rf + 0.66 * preds_lgb
    
    # Reconstruct final demand: demand_day48 + predicted_residual
    final_preds = df_test['demand_day48'] + blend_res
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
    
    # 13. Zip source files for portal submission
    zip_path = r"c:\Users\KIIT\Desktop\flipkartgrid\source_files.zip"
    print(f"Creating source package at {zip_path}...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(r"c:\Users\KIIT\Desktop\flipkartgrid\predict_v14.py", "predict_v14.py")
        zipf.write(r"c:\Users\KIIT\Desktop\flipkartgrid\README.txt", "README.txt")
        zipf.write(r"c:\Users\KIIT\Desktop\flipkartgrid\Traffic_Demand_Prediction.ipynb", "Traffic_Demand_Prediction.ipynb")
    print("Source code package zipped successfully!")

if __name__ == '__main__':
    main()
