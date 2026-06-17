import pandas as pd
import numpy as np
import os

data_dir = r"c:\Users\KIIT\Desktop\flipkartgrid\dataset\dataset"
train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))

train_48 = train[train['day'] == 48].copy()

# Decode/map geohashes to mapped_geohash in test
geohashes_train = set(train['geohash'].unique())
geohashes_test = set(test['geohash'].unique())
missing_geohashes = list(geohashes_test - geohashes_train)

if len(missing_geohashes) > 0:
    train_coords = []
    for gh in geohashes_train:
        base32 = "0123456789bcdefghjkmnpqrstuvwxyz"
        base32_map = {char: i for i, char in enumerate(base32)}
        lat_interval = [-90.0, 90.0]
        lon_interval = [-180.0, 180.0]
        is_even = True
        for char in gh:
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
        lat = (lat_interval[0] + lat_interval[1]) / 2
        lon = (lon_interval[0] + lon_interval[1]) / 2
        train_coords.append((gh, lat, lon))
    train_df = pd.DataFrame(train_coords, columns=['geohash', 'lat', 'lon'])
    
    gh_mapping = {}
    for gh in missing_geohashes:
        lat_interval = [-90.0, 90.0]
        lon_interval = [-180.0, 180.0]
        is_even = True
        for char in gh:
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
        lat = (lat_interval[0] + lat_interval[1]) / 2
        lon = (lon_interval[0] + lon_interval[1]) / 2
        train_df['dist'] = np.sqrt((train_df['lat'] - lat)**2 + (train_df['lon'] - lon)**2)
        gh_mapping[gh] = train_df.sort_values('dist').iloc[0]['geohash']
    test['mapped_geohash'] = test['geohash'].map(lambda x: gh_mapping.get(x, x))
else:
    test['mapped_geohash'] = test['geohash']

df_test = test.merge(
    train_48[['geohash', 'timestamp', 'demand']], 
    left_on=['mapped_geohash', 'timestamp'], 
    right_on=['geohash', 'timestamp'], 
    how='left', 
    suffixes=('', '_day48_raw')
)

print("Number of missing historical demands in test:", df_test['demand'].isna().sum())
print("Total rows in test:", len(test))
