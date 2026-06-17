import pandas as pd
import os

sub_path = r"c:\Users\KIIT\Desktop\flipkartgrid\submission.csv"
if os.path.exists(sub_path):
    df_sub = pd.read_csv(sub_path)
    print("Submission shape:", df_sub.shape)
    print("Unique Index count in submission:", df_sub['Index'].nunique())
    print("Min Index:", df_sub['Index'].min(), "Max Index:", df_sub['Index'].max())
    print("Any NaNs in submission:", df_sub.isna().sum().to_dict())
    
    test_path = r"c:\Users\KIIT\Desktop\flipkartgrid\dataset\dataset\test.csv"
    df_test = pd.read_csv(test_path)
    print("Test shape:", df_test.shape)
    print("Unique Index count in test:", df_test['Index'].nunique())
    print("Is submission Index exactly equal to test Index?", df_sub['Index'].equals(df_test['Index']))
else:
    print("submission.csv NOT FOUND")
