import pandas as pd
import numpy as np
import os

data_dir = r"c:\Users\KIIT\Desktop\flipkartgrid\dataset\dataset"
train = pd.read_csv(os.path.join(data_dir, "train.csv"))

# Decode geohashes
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

train_48 = train[train['day'] == 48].copy()
print("Number of unique geohashes in train_48:", train_48['geohash'].nunique())

# Let's see how many geohashes have 96 timestamps
counts = train_48.groupby('geohash')['timestamp'].count()
print("Geohashes with complete 96 timestamps:", (counts == 96).sum())
print("Geohashes with < 96 timestamps:", (counts < 96).sum())
print("Minimum timestamp count for a geohash:", counts.min())
