import pandas as pd
import numpy as np
import os
import time

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
            if is_even: lon_interval[0] = (lon_interval[0] + lon_interval[1]) / 2 if bit == 1 else lon_interval[0]
            else: lat_interval[0] = (lat_interval[0] + lat_interval[1]) / 2 if bit == 1 else lat_interval[0]
            is_even = not is_even
    return (lat_interval[0] + lat_interval[1]) / 2, (lon_interval[0] + lon_interval[1]) / 2

def main():
    data_dir = r"c:\Users\KIIT\Desktop\flipkartgrid\dataset\dataset"
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    
    # 1. Decode geohashes
    print("Decoding geohashes...")
    def add_coords(df):
        lats, lons = [], []
        for g in df['geohash']:
            lat, lon = decode_geohash(g)
            lats.append(lat); lons.append(lon)
        df['lat'] = lats; df['lon'] = lons
        return df
        
    train = add_coords(train)
    test = add_coords(test)
    
    train_48 = train[train['day'] == 48].copy()
    train_49 = train[train['day'] == 49].copy()
    
    # Let's perform spatial-temporal mapping for Test Set
    print("\nPerforming spatial-temporal mapping for Test Set...")
    start_time = time.time()
    
    # We will build a mapping for each timestamp
    df_test_mapped = []
    
    unique_timestamps = test['timestamp'].unique()
    for ts in unique_timestamps:
        test_ts = test[test['timestamp'] == ts].copy()
        train_48_ts = train_48[train_48['timestamp'] == ts].copy()
        
        if len(train_48_ts) == 0:
            # Fallback to nearest globally if timestamp is completely missing on Day 48 (should not happen)
            print(f"Warning: timestamp {ts} is completely missing in train_48!")
            train_48_ts = train_48.copy()
            
        # Get candidate coordinates
        candidates = train_48_ts[['geohash', 'lat', 'lon', 'demand']].drop_duplicates(subset=['geohash'])
        
        # Build coordinates arrays for fast computation
        cand_lats = candidates['lat'].values
        cand_lons = candidates['lon'].values
        cand_ghs = candidates['geohash'].values
        cand_demands = candidates['demand'].values
        
        mapped_demands = []
        for idx, row in test_ts.iterrows():
            # Euclidean distance to all candidates
            dists = (cand_lats - row['lat'])**2 + (cand_lons - row['lon'])**2
            min_idx = np.argmin(dists)
            mapped_demands.append(cand_demands[min_idx])
            
        test_ts['demand_day48'] = mapped_demands
        df_test_mapped.append(test_ts)
        
    df_test_mapped = pd.concat(df_test_mapped, ignore_index=True)
    end_time = time.time()
    print(f"Mapped {len(test)} test rows in {end_time - start_time:.3f} seconds!")
    print("Missing demands in mapped test:", df_test_mapped['demand_day48'].isna().sum())
    
    # Compare stats
    print("\nMapped test demand stats:")
    print(df_test_mapped['demand_day48'].describe())

if __name__ == '__main__':
    main()
