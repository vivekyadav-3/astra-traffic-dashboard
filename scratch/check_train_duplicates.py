import pandas as pd
import os

data_dir = r"c:\Users\KIIT\Desktop\flipkartgrid\dataset\dataset"
train = pd.read_csv(os.path.join(data_dir, "train.csv"))

dups = train.duplicated(subset=['day', 'geohash', 'timestamp']).sum()
print("Number of duplicates in train (day, geohash, timestamp):", dups)

if dups > 0:
    print("\nExample duplicates:")
    dup_rows = train[train.duplicated(subset=['day', 'geohash', 'timestamp'], keep=False)]
    print(dup_rows.sort_values(['day', 'geohash', 'timestamp']).head(10))
