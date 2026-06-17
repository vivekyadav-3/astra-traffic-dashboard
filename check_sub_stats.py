import pandas as pd

sub = pd.read_csv(r"c:\Users\KIIT\Desktop\flipkartgrid\submission.csv")
print("New submission shape:", sub.shape)
print("Stats:\n", sub['demand'].describe())
print("NaN count:", sub.isna().sum().sum())
print("Negative values count:", (sub['demand'] < 0).sum())
print("Values > 1 count:", (sub['demand'] > 1).sum())
