import pandas as pd
import numpy as np
import os

data_dir = r"c:\Users\KIIT\Desktop\flipkartgrid\dataset\dataset"
train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))

# Mapping logic
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

train_48 = train[train['day'] == 48].copy()
geohashes_train_48 = set(train_48['geohash'].unique())

df_test = test.merge(
    train_48[['geohash', 'timestamp', 'demand']], 
    left_on=['mapped_geohash', 'timestamp'], 
    right_on=['geohash', 'timestamp'], 
    how='left', 
    suffixes=('', '_day48_raw')
)

nan_rows = df_test[df_test['demand'].isna()]
print("Number of NaNs:", len(nan_rows))

# Let's see how many distinct mapped_geohashes in these NaN rows are completely missing from train_48
nan_mapped_ghs = nan_rows['mapped_geohash'].unique()
missing_from_48 = [gh for gh in nan_mapped_ghs if gh not in geohashes_train_48]
print(f"Of {len(nan_mapped_ghs)} mapped geohashes in NaN rows, {len(missing_from_48)} are completely missing from Day 48!")

# What about the others? If they are in Day 48, do they just have missing timestamps?
in_48 = [gh for gh in nan_mapped_ghs if gh in geohashes_train_48]
print(f"And {len(in_48)} are present on Day 48 but have missing timestamps in test range!")
if len(in_48) > 0:
    print("Example in_48 mapped geohash row count in train_48:", len(train_48[train_48['geohash'] == in_48[0]]))
