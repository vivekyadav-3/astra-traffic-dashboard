import os
import pandas as pd
import numpy as np

def main():
    data_dir = r"c:\Users\KIIT\Desktop\flipkartgrid\dataset\dataset"
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    
    train_48 = train[train['day'] == 48].copy()
    train_49 = train[train['day'] == 49].copy()
    
    # Let's merge Day 48 and Day 49 on the overlap period to analyze
    overlap_times = ['0:0', '0:15', '0:30', '0:45', '1:0', '1:15', '1:30', '1:45', '2:0']
    
    # Merge them
    merged = train_49[train_49['timestamp'].isin(overlap_times)].merge(
        train_48[['geohash', 'timestamp', 'demand']],
        on=['geohash', 'timestamp'],
        suffixes=('_49', '_48')
    )
    
    merged['diff'] = merged['demand_49'] - merged['demand_48']
    
    # Let's compute different shift estimators per geohash
    # 1. Simple Mean
    mean_shift = merged.groupby('geohash')['diff'].mean().reset_index().rename(columns={'diff': 'mean_shift'})
    
    # 2. Simple Median
    median_shift = merged.groupby('geohash')['diff'].median().reset_index().rename(columns={'diff': 'median_shift'})
    
    # 3. Weighted Mean (giving higher weight to later times)
    # Let's assign weights: 0:0 to 1:0 get weight 0.5, 1:15 to 2:0 get weight 1.5
    time_weights = {
        '0:0': 0.2, '0:15': 0.4, '0:30': 0.6, '0:45': 0.8,
        '1:0': 1.0, '1:15': 1.2, '1:30': 1.4, '1:45': 1.6, '2:0': 1.8
    }
    merged['weight'] = merged['timestamp'].map(time_weights)
    
    def weighted_mean(group):
        return np.average(group['diff'], weights=group['weight'])
        
    wmean_shift = merged.groupby('geohash').apply(weighted_mean).reset_index(name='wmean_shift')
    
    # 4. Last timestamp shift (at 2:0)
    last_shift = merged[merged['timestamp'] == '2:0'][['geohash', 'diff']].rename(columns={'diff': 'last_shift'})
    
    # Merge all estimators
    estimators = mean_shift.merge(median_shift, on='geohash', how='left')
    estimators = estimators.merge(wmean_shift, on='geohash', how='left')
    estimators = estimators.merge(last_shift, on='geohash', how='left')
    
    # Now let's see which estimator predicts the demand at 2:0 best (using demand_48 at 2:0 + estimator)
    # Wait, to predict 2:0, we should only use data up to 1:45 for the estimator to avoid leakage!
    # Let's calculate the estimators using only '0:0' to '1:45'
    merged_before_2 = merged[merged['timestamp'] != '2:0']
    
    mean_shift_prev = merged_before_2.groupby('geohash')['diff'].mean().reset_index().rename(columns={'diff': 'mean_shift_prev'})
    median_shift_prev = merged_before_2.groupby('geohash')['diff'].median().reset_index().rename(columns={'diff': 'median_shift_prev'})
    
    merged_before_2['weight_prev'] = merged_before_2['timestamp'].map(time_weights)
    wmean_shift_prev = merged_before_2.groupby('geohash').apply(
        lambda g: np.average(g['diff'], weights=g['weight_prev'])
    ).reset_index(name='wmean_shift_prev')
    
    last_shift_prev = merged[merged['timestamp'] == '1:45'][['geohash', 'diff']].rename(columns={'diff': 'last_shift_prev'})
    
    eval_df = merged[merged['timestamp'] == '2:0'][['geohash', 'demand_49', 'demand_48']].merge(
        mean_shift_prev, on='geohash', how='left'
    ).merge(
        median_shift_prev, on='geohash', how='left'
    ).merge(
        wmean_shift_prev, on='geohash', how='left'
    ).merge(
        last_shift_prev, on='geohash', how='left'
    )
    
    # Fill missing values
    eval_df = eval_df.fillna(0)
    
    from sklearn.metrics import r2_score
    print("Predicting demand at 2:0 using previous overlap timestamps:")
    for col in ['mean_shift_prev', 'median_shift_prev', 'wmean_shift_prev', 'last_shift_prev']:
        pred = np.clip(eval_df['demand_48'] + eval_df[col], 0, 1)
        r2 = r2_score(eval_df['demand_49'], pred)
        print(f"  Using {col:<20} -> R2: {r2:.5f}")

if __name__ == '__main__':
    main()
