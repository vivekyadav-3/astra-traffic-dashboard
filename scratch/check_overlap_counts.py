import pandas as pd
import os

data_dir = r"c:\Users\KIIT\Desktop\flipkartgrid\dataset\dataset"
train = pd.read_csv(os.path.join(data_dir, "train.csv"))

train_48 = train[train['day'] == 48].copy()
train_49 = train[train['day'] == 49].copy()

overlap_times = ['0:0', '0:15', '0:30', '0:45', '1:0', '1:15', '1:30', '1:45', '2:0']
train_48_overlap = train_48[train_48['timestamp'].isin(overlap_times)]
train_49_overlap = train_49[train_49['timestamp'].isin(overlap_times)]

counts_48 = train_48_overlap.groupby('geohash')['timestamp'].count()
counts_49 = train_49_overlap.groupby('geohash')['timestamp'].count()

print("Day 48 Overlap Count Stats:")
print(counts_48.describe())

print("\nDay 49 Overlap Count Stats:")
print(counts_49.describe())

# Check how many have less than 9 overlap timestamps
print(f"\nDay 48 geohashes with < 9 overlap records: {(counts_48 < 9).sum()} out of {len(counts_48)}")
print(f"Day 49 geohashes with < 9 overlap records: {(counts_49 < 9).sum()} out of {len(counts_49)}")
