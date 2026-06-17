import os
import pandas as pd
import numpy as np

print("Checking workspace files...")
files = ["predict.py", "predict_v2.py", "predict_v3.py", "predict_v4.py", "predict_v5.py", "submission.csv"]
for f in files:
    path = os.path.join(r"c:\Users\KIIT\Desktop\flipkartgrid", f)
    if os.path.exists(path):
        mtime = os.path.getmtime(path)
        import datetime
        dt = datetime.datetime.fromtimestamp(mtime)
        print(f"{f}: size={os.path.getsize(path)} bytes, last_modified={dt}")
    else:
        print(f"{f}: NOT FOUND")

print("\nReading train.csv info:")
train_path = r"c:\Users\KIIT\Desktop\flipkartgrid\dataset\dataset\train.csv"
if os.path.exists(train_path):
    df_train = pd.read_csv(train_path)
    print("Train shape:", df_train.shape)
    print("Days in train:", df_train['day'].unique())
    print("Day 49 timestamps in train:", df_train[df_train['day'] == 49]['timestamp'].unique())
    print("Day 48 timestamps in train count:", df_train[df_train['day'] == 48]['timestamp'].nunique())
    print("Total geohashes in train:", df_train['geohash'].nunique())
else:
    print("train.csv NOT FOUND")

print("\nReading test.csv info:")
test_path = r"c:\Users\KIIT\Desktop\flipkartgrid\dataset\dataset\test.csv"
if os.path.exists(test_path):
    df_test = pd.read_csv(test_path)
    print("Test shape:", df_test.shape)
    print("Days in test:", df_test['day'].unique())
    print("Timestamps in test count:", df_test['timestamp'].nunique())
    print("Timestamps in test:", df_test['timestamp'].unique()[:10], "... to ...", df_test['timestamp'].unique()[-10:])
    print("Total geohashes in test:", df_test['geohash'].nunique())
else:
    print("test.csv NOT FOUND")

print("\nReading current submission.csv:")
sub_path = r"c:\Users\KIIT\Desktop\flipkartgrid\submission.csv"
if os.path.exists(sub_path):
    df_sub = pd.read_csv(sub_path)
    print("Submission shape:", df_sub.shape)
    print("Submission head:\n", df_sub.head())
    print("Submission stats:\n", df_sub['demand'].describe())
    print("Any NaNs:", df_sub.isna().sum().sum())
else:
    print("submission.csv NOT FOUND")
