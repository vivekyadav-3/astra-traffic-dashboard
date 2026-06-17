import pandas as pd
import numpy as np
import os

data_dir = r"c:\Users\KIIT\Desktop\flipkartgrid\dataset\dataset"
train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))

train_48 = train[train['day'] == 48].copy()

# Simple V1 mapping
geohashes_train = set(train['geohash'].unique())
geohashes_test = set(test['geohash'].unique())
missing_geohashes = list(geohashes_test - geohashes_train)

# Decode helper for coordinates
base32 = "0123456789bcdefghjkmnpqrstuvwxyz"
base32_map = {char: i for i, char in enumerate(base32)}
def decode(gh):
    lat_interval = [-90.0, 90.0]
    lon_interval = [-180.0, 180.0]
    is_even = True
    for char in gh:
        val = base32_map[char]
        for i in range(4, -1, -1):
            bit = (val >> i) & 1
            if is_even: lat_interval[0] = (lat_interval[0] + lat_interval[1]) / 2 if bit == 1 else lat_interval[0]
            else: lat_interval[0] = (lat_interval[0] + lat_interval[1]) / 2 if bit == 1 else lat_interval[0]
            is_even = not is_even
    return (lat_interval[0] + lat_interval[1]) / 2, (lon_interval[0] + lon_interval[1]) / 2

if len(missing_geohashes) > 0:
    train_coords = [(gh, *decode(gh)) for gh in geohashes_train]
    train_df = pd.DataFrame(train_coords, columns=['geohash', 'lat', 'lon'])
    gh_mapping = {}
    for gh in missing_geohashes:
        lat, lon = decode(gh)
        train_df['dist'] = np.sqrt((train_df['lat'] - lat)**2 + (train_df['lon'] - lon)**2)
        gh_mapping[gh] = train_df.sort_values('dist').iloc[0]['geohash']
    test['mapped_geohash'] = test['geohash'].map(lambda x: gh_mapping.get(x, x))
else:
    test['mapped_geohash'] = test['geohash']

# V1 merge
df_test_v1 = test.merge(
    train_48[['geohash', 'timestamp', 'demand']], 
    left_on=['mapped_geohash', 'timestamp'], 
    right_on=['geohash', 'timestamp'], 
    how='left', 
    suffixes=('', '_day48_raw')
)
geohash_stats = train_48.groupby('geohash')['demand'].mean().reset_index()
geohash_stats.columns = ['geohash', 'gh_mean']
global_mean = train_48['demand'].mean()

df_test_v1 = df_test_v1.merge(geohash_stats, left_on='mapped_geohash', right_on='geohash', how='left')
df_test_v1['demand_day48'] = df_test_v1['demand'].fillna(df_test_v1['gh_mean']).fillna(global_mean)

print("V1 demand_day48 stats:")
print(df_test_v1['demand_day48'].describe())

# Spatial-temporal mapping helper
def get_spatial_temporal_demand(target_df, source_df):
    mapped_demands = []
    unique_timestamps = target_df['timestamp'].unique()
    mapped_df_list = []
    
    # We decode geohashes to get coordinates
    target_df['lat_dec'] = target_df['geohash'].map(lambda x: decode(x)[0])
    target_df['lon_dec'] = target_df['geohash'].map(lambda x: decode(x)[1])
    source_df['lat_dec'] = source_df['geohash'].map(lambda x: decode(x)[0])
    source_df['lon_dec'] = source_df['geohash'].map(lambda x: decode(x)[1])
    
    for ts in unique_timestamps:
        target_ts = target_df[target_df['timestamp'] == ts].copy()
        source_ts = source_df[source_df['timestamp'] == ts].copy()
        
        candidates = source_ts[['geohash', 'lat_dec', 'lon_dec', 'demand']].drop_duplicates(subset=['geohash'])
        cand_lats = candidates['lat_dec'].values
        cand_lons = candidates['lon_dec'].values
        cand_ghs = candidates['geohash'].values
        cand_demands = candidates['demand'].values
        
        exact_map = dict(zip(cand_ghs, cand_demands))
        
        ts_demands = []
        for idx, row in target_ts.iterrows():
            gh = row['mapped_geohash']
            if gh in exact_map:
                ts_demands.append(exact_map[gh])
            else:
                dists = (cand_lats - row['lat_dec'])**2 + (cand_lons - row['lon_dec'])**2
                min_idx = np.argmin(dists)
                ts_demands.append(cand_demands[min_idx])
        target_ts['demand_day48'] = ts_demands
        mapped_df_list.append(target_ts)
    return pd.concat(mapped_df_list, ignore_index=True)

df_test_v15 = get_spatial_temporal_demand(test, train_48)
print("\nV15 demand_day48 stats:")
print(df_test_v15['demand_day48'].describe())

# Let's see the average absolute difference for the NaNs
nan_mask = df_test_v1['demand'].isna()
diffs = np.abs(df_test_v1.loc[nan_mask, 'demand_day48'] - df_test_v15.loc[nan_mask, 'demand_day48'])
print(f"\nFor the 4638 NaN rows:")
print(f"  Mean diff between V1 (gh_mean) and V15 (temporal neighbor): {diffs.mean():.5f}")
print(f"  V1 mean: {df_test_v1.loc[nan_mask, 'demand_day48'].mean():.5f}")
print(f"  V15 mean: {df_test_v15.loc[nan_mask, 'demand_day48'].mean():.5f}")
