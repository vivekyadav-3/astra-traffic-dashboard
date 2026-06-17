import pandas as pd
import os

data_dir = r"c:\Users\KIIT\Desktop\flipkartgrid\dataset\dataset"
train = pd.read_csv(os.path.join(data_dir, "train.csv"))

overlap_times = ['0:0', '0:15', '0:30', '0:45', '1:0', '1:15', '1:30', '1:45', '2:0']
overlap_48 = train[(train['day'] == 48) & (train['timestamp'].isin(overlap_times))]
overlap_49 = train[(train['day'] == 49) & (train['timestamp'].isin(overlap_times))]

print("Overlap Day 48 demand stats:")
print(overlap_48['demand'].describe())

print("\nOverlap Day 49 demand stats:")
print(overlap_49['demand'].describe())

print("\nFull Day 48 demand stats:")
print(train[train['day'] == 48]['demand'].describe())
