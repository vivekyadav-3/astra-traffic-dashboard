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
    
    # 1. Decode geohashes
    lats, lons = [], []
    for g in train['geohash']:
        lat, lon = decode_geohash(g)
        lats.append(lat)
        lons.append(lon)
    train['lat'] = lats
    train['lon'] = lons
    
    # 2. Temporal features
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
    
    # Create lag/lead features for Day 48
    train_48 = train[train['day'] == 48].copy()
    
    # Sort to create lag/lead properly
    # We want to get the demand at the previous and next timestamps for each geohash
    # A robust way is to pivot or use shift within groupby
    train_48 = train_48.sort_values(['geohash', 'time_of_day'])
    train_48['demand_day48_prev'] = train_48.groupby('geohash')['demand'].shift(1)
    train_48['demand_day48_next'] = train_48.groupby('geohash')['demand'].shift(-1)
    
    # Let's fill the boundary NaNs with the current demand or geohash mean
    gh_mean_48 = train_48.groupby('geohash')['demand'].mean().to_dict()
    train_48['gh_mean'] = train_48['geohash'].map(gh_mean_48)
    
    train_48['demand_day48_prev'] = train_48['demand_day48_prev'].fillna(train_48['gh_mean'])
    train_48['demand_day48_next'] = train_48['demand_day48_next'].fillna(train_48['gh_mean'])
    
    # Now for Day 49, we need to map:
    # - demand_day48 (exact)
    # - demand_day48_prev (demand on Day 48 at timestamp - 15m)
    # - demand_day48_next (demand on Day 48 at timestamp + 15m)
    
    # Let's create a lookup table for Day 48 demand by geohash and timestamp
    # But wait, we can just merge train_48 directly!
    train_49 = train[train['day'] == 49].copy()
    
    # We need to map the timestamp on Day 49 to the corresponding timestamps on Day 48
    # Let's define the timestamp mapping:
    # Let's write a function to add/subtract 15 minutes to timestamp string
    def shift_time_str(t_str, delta_mins):
        h, m = map(int, t_str.split(':'))
        total_mins = h * 60 + m + delta_mins
        total_mins = total_mins % (24 * 60) # wrap around
        new_h = total_mins // 60
        new_m = total_mins % 60
        return f"{new_h}:{new_m}"
        
    train_49['timestamp_prev'] = train_49['timestamp'].map(lambda x: shift_time_str(x, -15))
    train_49['timestamp_next'] = train_49['timestamp'].map(lambda x: shift_time_str(x, 15))
    
    # Now merge Day 48 demand at timestamp, timestamp_prev, and timestamp_next
    t48_lookup = train_48[['geohash', 'timestamp', 'demand']].rename(columns={'demand': 'demand_day48'})
    
    df_train_49 = train_49.merge(t48_lookup, on=['geohash', 'timestamp'], how='left')
    
    df_train_49 = df_train_49.merge(
        t48_lookup.rename(columns={'demand_day48': 'demand_day48_prev'}),
        left_on=['geohash', 'timestamp_prev'], right_on=['geohash', 'timestamp'], how='left'
    ).drop(columns=['timestamp_y']).rename(columns={'timestamp_x': 'timestamp'})
    
    df_train_49 = df_train_49.merge(
        t48_lookup.rename(columns={'demand_day48': 'demand_day48_next'}),
        left_on=['geohash', 'timestamp_next'], right_on=['geohash', 'timestamp'], how='left'
    ).drop(columns=['timestamp_y']).rename(columns={'timestamp_x': 'timestamp'})
    
    # Fill NAs
    df_train_49['gh_mean'] = df_train_49['geohash'].map(gh_mean_48)
    global_mean = train_48['demand'].mean()
    df_train_49['demand_day48'] = df_train_49['demand_day48'].fillna(df_train_49['gh_mean']).fillna(global_mean)
    df_train_49['demand_day48_prev'] = df_train_49['demand_day48_prev'].fillna(df_train_49['gh_mean']).fillna(global_mean)
    df_train_49['demand_day48_next'] = df_train_49['demand_day48_next'].fillna(df_train_49['gh_mean']).fillna(global_mean)
    
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
    
    df_train_49 = df_train_49.merge(shifts, on='geohash', how='left')
    df_train_49['shift_diff'] = df_train_49['shift_diff'].fillna(0.0)
    df_train_49['shift_ratio'] = df_train_49['shift_ratio'].fillna(1.0)
    
    # Encode categoricals
    cat_cols = ['RoadType', 'LargeVehicles', 'Landmarks', 'Weather']
    for col in cat_cols:
        df_train_49[col] = df_train_49[col].astype(str)
        le = LabelEncoder()
        df_train_49[col + '_enc'] = le.fit_transform(df_train_49[col])
        
    med_temp = df_train_49['Temperature'].median()
    med_lanes = df_train_49['NumberofLanes'].median()
    df_train_49['Temperature'] = df_train_49['Temperature'].fillna(med_temp)
    df_train_49['NumberofLanes'] = df_train_49['NumberofLanes'].fillna(med_lanes)
    
    # Let's compare feature sets using KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    base_feats = [
        'lat', 'lon', 'sin_time', 'cos_time', 'time_of_day',
        'demand_day48', 'shift_diff', 'shift_ratio', 'gh_mean',
        'NumberofLanes', 'Temperature'
    ] + [c + '_enc' for c in cat_cols]
    
    new_feats = base_feats + ['demand_day48_prev', 'demand_day48_next']
    
    print("\nRunning KFold CV with Base Features:")
    preds_base = np.zeros(len(df_train_49))
    et = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    for tr_idx, val_idx in kf.split(df_train_49):
        X_tr, y_tr = df_train_49[base_feats].iloc[tr_idx], df_train_49['demand'].iloc[tr_idx]
        X_val = df_train_49[base_feats].iloc[val_idx]
        et.fit(X_tr, y_tr)
        preds_base[val_idx] = et.predict(X_val)
    print(f"Base R2: {r2_score(df_train_49['demand'], preds_base):.5f}")
    
    print("\nRunning KFold CV with Trend Features:")
    preds_new = np.zeros(len(df_train_49))
    for tr_idx, val_idx in kf.split(df_train_49):
        X_tr, y_tr = df_train_49[new_feats].iloc[tr_idx], df_train_49['demand'].iloc[tr_idx]
        X_val = df_train_49[new_feats].iloc[val_idx]
        et.fit(X_tr, y_tr)
        preds_new[val_idx] = et.predict(X_val)
    print(f"Trend Features R2: {r2_score(df_train_49['demand'], preds_new):.5f}")

if __name__ == '__main__':
    main()
