import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import lightgbm as lgb
import xgboost as xgb

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
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    test_indices = test['Index'].copy()

    # SPATIAL COORDS
    def add_coords(df):
        lats, lons = [], []
        for g in df['geohash']:
            lat, lon = decode_geohash(g)
            lats.append(lat); lons.append(lon)
        df['lat'] = lats; df['lon'] = lons
        return df

    train = add_coords(train)
    test = add_coords(test)

    # Map missing geohashes
    geohashes_train = set(train['geohash'].unique())
    geohashes_test = set(test['geohash'].unique())
    missing = list(geohashes_test - geohashes_train)
    gh_mapping = {}
    if missing:
        tc = [(gh, *decode_geohash(gh)) for gh in geohashes_train]
        tdf = pd.DataFrame(tc, columns=['geohash', 'lat', 'lon'])
        for gh in missing:
            lat, lon = decode_geohash(gh)
            tdf['dist'] = np.sqrt((tdf['lat']-lat)**2 + (tdf['lon']-lon)**2)
            gh_mapping[gh] = tdf.sort_values('dist').iloc[0]['geohash']
    test['mapped_geohash'] = test['geohash'].map(lambda x: gh_mapping.get(x, x))

    # TEMPORAL
    def add_time_features(df):
        hours, minutes = [], []
        for t in df['timestamp']:
            h, m = map(int, t.split(':'))
            hours.append(h); minutes.append(m)
        df['hour'] = hours; df['minute'] = minutes
        df['time_of_day'] = df['hour'] + df['minute'] / 60.0
        df['sin_time'] = np.sin(2 * np.pi * df['time_of_day'] / 24.0)
        df['cos_time'] = np.cos(2 * np.pi * df['time_of_day'] / 24.0)
        df['quarter'] = (df['hour'] // 6).astype(int)
        df['is_peak'] = (((df['hour'] >= 7) & (df['hour'] <= 10)) | ((df['hour'] >= 17) & (df['hour'] <= 20))).astype(int)
        return df

    train = add_time_features(train)
    test = add_time_features(test)
    train_48 = train[train['day'] == 48].copy()
    train_49 = train[train['day'] == 49].copy()

    # OVERLAP SHIFTS
    overlap_times = ['0:0', '0:15', '0:30', '0:45', '1:0', '1:15', '1:30', '1:45', '2:0']
    t48_ovlp = train_48[train_48['timestamp'].isin(overlap_times)]
    t49_ovlp = train_49[train_49['timestamp'].isin(overlap_times)]

    ovlp48 = t48_ovlp.groupby('geohash')['demand'].agg(['mean','std','min','max']).reset_index()
    ovlp48.columns = ['geohash','mean_ovlp48','std_ovlp48','min_ovlp48','max_ovlp48']
    ovlp49 = t49_ovlp.groupby('geohash')['demand'].agg(['mean','std','min','max']).reset_index()
    ovlp49.columns = ['geohash','mean_ovlp49','std_ovlp49','min_ovlp49','max_ovlp49']

    shifts = ovlp48.merge(ovlp49, on='geohash', how='inner')
    shifts['shift_diff'] = shifts['mean_ovlp49'] - shifts['mean_ovlp48']
    shifts['shift_ratio'] = (shifts['mean_ovlp49'] + 1e-5) / (shifts['mean_ovlp48'] + 1e-5)
    shifts['shift_std_ratio'] = (shifts['std_ovlp49'] + 1e-5) / (shifts['std_ovlp48'] + 1e-5)

    # GEOHASH STATS
    gh_stats = train_48.groupby('geohash')['demand'].agg(['mean','std','median','min','max']).reset_index()
    gh_stats.columns = ['geohash','gh_mean','gh_std','gh_median','gh_min','gh_max']
    global_mean = train_48['demand'].mean()

    gh_hour_stats = train_48.groupby(['geohash','hour'])['demand'].mean().reset_index()
    gh_hour_stats.columns = ['geohash','hour','gh_hour_mean']

    shift_cols = ['geohash','shift_diff','shift_ratio','shift_std_ratio','mean_ovlp48','std_ovlp48','mean_ovlp49','std_ovlp49']

    def prep_dataset(df_target, geohash_col='geohash'):
        df = df_target.copy()
        
        # Merge day48 demand
        d48_demand = train_48[['geohash','timestamp','demand']].rename(columns={'demand':'demand_day48','geohash':'_gh'})
        df = df.merge(d48_demand, left_on=[geohash_col,'timestamp'], right_on=['_gh','timestamp'], how='left')
        df = df.drop(columns=['_gh'], errors='ignore')
        
        # Merge shifts
        df = df.merge(shifts[shift_cols].rename(columns={'geohash':'_gh'}), left_on=geohash_col, right_on='_gh', how='left')
        df = df.drop(columns=['_gh'], errors='ignore')
        
        # Merge gh stats
        df = df.merge(gh_stats.rename(columns={'geohash':'_gh'}), left_on=geohash_col, right_on='_gh', how='left')
        df = df.drop(columns=['_gh'], errors='ignore')
        
        # Merge gh hour stats
        df = df.merge(gh_hour_stats.rename(columns={'geohash':'_gh'}), left_on=[geohash_col,'hour'], right_on=['_gh','hour'], how='left')
        df = df.drop(columns=['_gh'], errors='ignore')
        
        df['shift_diff'] = df['shift_diff'].fillna(0.0)
        df['shift_ratio'] = df['shift_ratio'].fillna(1.0)
        df['shift_std_ratio'] = df['shift_std_ratio'].fillna(1.0)
        df['demand_day48'] = df['demand_day48'].fillna(df['gh_mean']).fillna(global_mean)
        df['gh_hour_mean'] = df['gh_hour_mean'].fillna(df['gh_mean']).fillna(global_mean)
        df['adj_pred_diff'] = np.clip(df['demand_day48'] + df['shift_diff'], 0, 1)
        df['adj_pred_ratio'] = np.clip(df['demand_day48'] * df['shift_ratio'], 0, 1)
        df['gh_hour_vs_mean'] = df['gh_hour_mean'] - df['gh_mean'].fillna(global_mean)
        return df

    print("Preparing datasets...")
    df_train_49 = prep_dataset(train_49, 'geohash')

    df_train_48 = train_48.copy()
    df_train_48['demand_day48'] = df_train_48['geohash'].map(train_48.groupby('geohash')['demand'].mean())
    df_train_48['shift_diff'] = 0.0; df_train_48['shift_ratio'] = 1.0; df_train_48['shift_std_ratio'] = 1.0
    df_train_48 = df_train_48.merge(gh_stats.rename(columns={'geohash':'_gh'}), left_on='geohash', right_on='_gh', how='left').drop(columns=['_gh'], errors='ignore')
    df_train_48 = df_train_48.merge(gh_hour_stats.rename(columns={'geohash':'_gh'}), left_on=['geohash','hour'], right_on=['_gh','hour'], how='left').drop(columns=['_gh'], errors='ignore')
    df_train_48['mean_ovlp48'] = df_train_48['gh_mean']
    df_train_48['std_ovlp48'] = df_train_48['gh_std']
    df_train_48['mean_ovlp49'] = df_train_48['gh_mean']
    df_train_48['std_ovlp49'] = df_train_48['gh_std']
    df_train_48['gh_hour_mean'] = df_train_48['gh_hour_mean'].fillna(df_train_48['gh_mean']).fillna(global_mean)
    df_train_48['adj_pred_diff'] = df_train_48['demand_day48'].fillna(global_mean)
    df_train_48['adj_pred_ratio'] = df_train_48['demand_day48'].fillna(global_mean)
    df_train_48['gh_hour_vs_mean'] = df_train_48['gh_hour_mean'] - df_train_48['gh_mean'].fillna(global_mean)

    df_test = prep_dataset(test, 'mapped_geohash')

    combined_train = pd.concat([df_train_48, df_train_49], ignore_index=True)

    # Encode categoricals
    cat_cols = ['RoadType', 'LargeVehicles', 'Landmarks', 'Weather']
    le_maps = {}
    for col in cat_cols:
        combined_train[col] = combined_train[col].astype(str)
        le = LabelEncoder()
        combined_train[col + '_enc'] = le.fit_transform(combined_train[col])
        le_maps[col] = dict(zip(le.classes_, range(len(le.classes_))))
        for df in [df_train_49, df_test]:
            df[col] = df[col].astype(str)
            df[col + '_enc'] = df[col].map(le_maps[col]).fillna(-1).astype(int)

    # Impute numeric
    med_temp = combined_train['Temperature'].median()
    med_lanes = combined_train['NumberofLanes'].median()
    for df in [combined_train, df_train_49, df_test]:
        df['Temperature'] = df['Temperature'].fillna(med_temp)
        df['NumberofLanes'] = df['NumberofLanes'].fillna(med_lanes)

    feature_cols = [
        'lat', 'lon', 'sin_time', 'cos_time', 'time_of_day', 'quarter', 'is_peak',
        'demand_day48', 'shift_diff', 'shift_ratio', 'shift_std_ratio',
        'gh_mean', 'gh_std', 'gh_median', 'gh_min', 'gh_max',
        'gh_hour_mean', 'gh_hour_vs_mean',
        'mean_ovlp48', 'std_ovlp48', 'mean_ovlp49', 'std_ovlp49',
        'adj_pred_diff', 'adj_pred_ratio',
        'NumberofLanes', 'Temperature'
    ] + [c + '_enc' for c in cat_cols]

    print(f"Training with {len(feature_cols)} features on combined data...")

    X_comb = combined_train[feature_cols]; y_comb = combined_train['demand']
    X_49 = df_train_49[feature_cols]; y_49 = df_train_49['demand']
    X_test = df_test[feature_cols]

    # Model 1: Extra Trees on Combined
    print("Training ExtraTrees (combined)...")
    et_comb = ExtraTreesRegressor(n_estimators=200, max_features=0.7, random_state=42, n_jobs=-1)
    et_comb.fit(X_comb, y_comb)

    # Model 2: Extra Trees on Day 49
    print("Training ExtraTrees (day49)...")
    et_49 = ExtraTreesRegressor(n_estimators=200, max_features=0.7, random_state=42, n_jobs=-1)
    et_49.fit(X_49, y_49)

    # Model 3: RandomForest on Combined
    print("Training RandomForest (combined)...")
    rf_comb = RandomForestRegressor(n_estimators=200, max_features=0.5, random_state=42, n_jobs=-1)
    rf_comb.fit(X_comb, y_comb)

    # Model 4: LightGBM on Combined
    print("Training LightGBM (combined)...")
    lgb_params = {
        'n_estimators': 500, 'learning_rate': 0.05,
        'max_depth': 8, 'num_leaves': 63,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'min_child_samples': 20, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
        'random_state': 42, 'n_jobs': -1, 'verbosity': -1
    }
    lgb_comb = lgb.LGBMRegressor(**lgb_params)
    lgb_comb.fit(X_comb, y_comb)

    # Model 5: LightGBM on Day 49
    print("Training LightGBM (day49)...")
    lgb_49 = lgb.LGBMRegressor(**lgb_params)
    lgb_49.fit(X_49, y_49)

    # Model 6: XGBoost on Combined
    print("Training XGBoost (combined)...")
    xgb_comb = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=7,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, n_jobs=-1, verbosity=0
    )
    xgb_comb.fit(X_comb, y_comb)

    # Weighted ensemble
    print("Generating predictions...")
    p1 = et_comb.predict(X_test)
    p2 = et_49.predict(X_test)
    p3 = rf_comb.predict(X_test)
    p4 = lgb_comb.predict(X_test)
    p5 = lgb_49.predict(X_test)
    p6 = xgb_comb.predict(X_test)

    final_preds = 0.20*p1 + 0.15*p2 + 0.10*p3 + 0.25*p4 + 0.20*p5 + 0.10*p6
    final_preds = np.clip(final_preds, 0.0, 1.0)

    submission = pd.DataFrame({'Index': test_indices, 'demand': final_preds})
    sub_path = r"c:\Users\KIIT\Desktop\flipkartgrid\submission.csv"
    submission.to_csv(sub_path, index=False)
    print(f"\nSubmission saved to {sub_path}")
    print("Shape:", submission.shape)
    print(submission.head())
    print("Min demand:", final_preds.min(), "Max demand:", final_preds.max())

if __name__ == '__main__':
    main()
