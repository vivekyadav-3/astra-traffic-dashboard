import os
import json
import pandas as pd
import numpy as np

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

def time_key(t):
    h, m = map(int, t.split(':'))
    return h * 60 + m

def main():
    print("Loading datasets...")
    data_dir = r"c:\Users\KIIT\Desktop\flipkartgrid\dataset\dataset"
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    sub_path = r"c:\Users\KIIT\Desktop\flipkartgrid\submission.csv"
    
    if os.path.exists(sub_path):
        submission = pd.read_csv(sub_path)
        print("Loaded submission.csv with shape:", submission.shape)
    else:
        print("ERROR: submission.csv not found!")
        return

    # Add predictions to test
    test_preds = test.merge(submission, on="Index", how="left")
    
    # Unique geohashes in test set (these are the ones we need to display)
    test_geohashes = test_preds['geohash'].unique()
    print(f"Total unique geohashes in test set: {len(test_geohashes)}")
    
    # Define chronological timestamps
    # Day 48 has all 96 timestamps (0:0 to 23:45)
    day48_timestamps = sorted(list(train[train['day'] == 48]['timestamp'].unique()), key=time_key)
    
    # Day 49 has actual overlap (0:0 to 2:0) and test predictions (2:15 to 13:45)
    day49_actual_ts = sorted(list(train[train['day'] == 49]['timestamp'].unique()), key=time_key)
    day49_pred_ts = sorted(list(test['timestamp'].unique()), key=time_key)
    day49_timestamps = day49_actual_ts + day49_pred_ts
    
    print(f"Day 48 timestamps count: {len(day48_timestamps)}")
    print(f"Day 49 timestamps count: {len(day49_timestamps)}")
    
    # Aggregate data per geohash
    geohash_data = {}
    
    # Filter train sets
    train_48 = train[train['day'] == 48]
    train_49 = train[train['day'] == 49]
    
    # Map from geohash -> details
    # We will compute properties from the train set (or test set if missing in train)
    print("Building geohash profiles...")
    for gh in test_geohashes:
        lat, lon = decode_geohash(gh)
        
        # Get metadata
        gh_train = train_48[train_48['geohash'] == gh]
        if len(gh_train) > 0:
            lanes = int(gh_train['NumberofLanes'].dropna().iloc[0]) if not gh_train['NumberofLanes'].dropna().empty else 2
            road_type = str(gh_train['RoadType'].dropna().iloc[0]) if not gh_train['RoadType'].dropna().empty else "Unknown"
            landmarks = str(gh_train['Landmarks'].dropna().iloc[0]) if not gh_train['Landmarks'].dropna().empty else "None"
            weather = str(gh_train['Weather'].dropna().iloc[0]) if not gh_train['Weather'].dropna().empty else "Clear"
            temp = float(gh_train['Temperature'].dropna().iloc[0]) if not gh_train['Temperature'].dropna().empty else 25.0
        else:
            gh_test = test[test['geohash'] == gh]
            lanes = int(gh_test['NumberofLanes'].dropna().iloc[0]) if not gh_test['NumberofLanes'].dropna().empty else 2
            road_type = str(gh_test['RoadType'].dropna().iloc[0]) if not gh_test['RoadType'].dropna().empty else "Unknown"
            landmarks = str(gh_test['Landmarks'].dropna().iloc[0]) if not gh_test['Landmarks'].dropna().empty else "None"
            weather = str(gh_test['Weather'].dropna().iloc[0]) if not gh_test['Weather'].dropna().empty else "Clear"
            temp = float(gh_test['Temperature'].dropna().iloc[0]) if not gh_test['Temperature'].dropna().empty else 25.0
            
        # Compile Day 48 actuals
        # Build map for O(1) timestamp lookup
        gh48_map = dict(zip(gh_train['timestamp'], gh_train['demand']))
        day48_demand_list = []
        for ts in day48_timestamps:
            day48_demand_list.append(round(float(gh48_map.get(ts, 0.0)), 4))
            
        # Compile Day 49 profiles (Actual overlap + predictions)
        # Actual overlap (0:0 to 2:0)
        gh49_actual_map = dict(zip(train_49[train_49['geohash'] == gh]['timestamp'], train_49[train_49['geohash'] == gh]['demand']))
        # Predictions (2:15 to 13:45)
        gh_preds = test_preds[test_preds['geohash'] == gh]
        gh49_pred_map = dict(zip(gh_preds['timestamp'], gh_preds['demand']))
        
        day49_demand_list = []
        for ts in day49_timestamps:
            if ts in gh49_actual_map:
                day49_demand_list.append(round(float(gh49_actual_map[ts]), 4))
            elif ts in gh49_pred_map:
                day49_demand_list.append(round(float(gh49_pred_map[ts]), 4))
            else:
                # Fallback to last known value or 0.0
                day49_demand_list.append(0.0)
                
        # Mean demand for sorting/ranking
        mean_demand = round(float(np.mean(day49_demand_list)), 4)
        
        geohash_data[gh] = {
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "lanes": lanes,
            "road_type": road_type,
            "landmarks": landmarks,
            "weather": weather,
            "temp": temp,
            "mean_demand": mean_demand,
            "day48": day48_demand_list,
            "day49": day49_demand_list
        }
        
    # Global stats
    print("Calculating global statistics...")
    global_day48 = []
    global_day49 = []
    
    # Calculate global temporal averages across all geohashes
    for i in range(len(day48_timestamps)):
        vals = [geohash_data[gh]["day48"][i] for gh in test_geohashes]
        global_day48.append(round(float(np.mean(vals)), 4))
        
    for i in range(len(day49_timestamps)):
        vals = [geohash_data[gh]["day49"][i] for gh in test_geohashes]
        global_day49.append(round(float(np.mean(vals)), 4))
        
    # Calculate road type stats
    road_types = {}
    for gh in test_geohashes:
        rt = geohash_data[gh]["road_type"]
        if rt not in road_types:
            road_types[rt] = []
        road_types[rt].append(geohash_data[gh]["mean_demand"])
        
    road_type_stats = {rt: round(float(np.mean(vals)), 4) for rt, vals in road_types.items()}
    
    # Combine everything
    dashboard_data = {
        "day48_timestamps": day48_timestamps,
        "day49_timestamps": day49_timestamps,
        "global_day48": global_day48,
        "global_day49": global_day49,
        "road_type_stats": road_type_stats,
        "geohashes": geohash_data
    }
    
    # Save to file
    out_path = r"c:\Users\KIIT\Desktop\flipkartgrid\dashboard_data.json"
    with open(out_path, "w") as f:
        json.dump(dashboard_data, f)
        
    print(f"Successfully saved dashboard data to {out_path}!")
    print(f"File size: {os.path.getsize(out_path) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main()
