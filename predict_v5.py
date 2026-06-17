import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import lightgbm as lgb

def decode_geohash(geohash):
    base32 = "0123456789bcdefghjkmnpqrstuvwxyz"
    base32_map = {char: i for i, char in enumerate(base32)}
    lat_interval = [-90.0, 90.0]; lon_interval = [-180.0, 180.0]
    is_even = True
    for char in geohash:
        val = base32_map[char]
        for i in range(4, -1, -1):
            bit = (val >> i) & 1
            if is_even:
                mid = (lon_interval[0] + lon_interval[1]) / 2
                if bit == 1: lon_interval[0] = mid
                else: lon_interval[1] = mid
            else:
                mid = (lat_interval[0] + lat_interval[1]) / 2
                if bit == 1: lat_interval[0] = mid
                else: lat_interval[1] = mid
            is_even = not is_even
    return (lat_interval[0]+lat_interval[1])/2, (lon_interval[0]+lon_interval[1])/2

def main():
    print("Loading data...")
    data_dir = r"c:\Users\KIIT\Desktop\flipkartgrid\dataset\dataset"
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test  = pd.read_csv(os.path.join(data_dir, "test.csv"))
    test_indices = test['Index'].copy()

    # Spatial coords
    def add_coords(df):
        lats, lons = [], []
        for g in df['geohash']:
            lat, lon = decode_geohash(g); lats.append(lat); lons.append(lon)
        df['lat'] = lats; df['lon'] = lons
        return df
    train = add_coords(train); test = add_coords(test)

    # Map missing test geohashes to nearest train geohash
    gh_train = set(train['geohash'].unique())
    missing   = list(set(test['geohash'].unique()) - gh_train)
    gh_map    = {}
    if missing:
        tc  = [(g, *decode_geohash(g)) for g in gh_train]
        tdf = pd.DataFrame(tc, columns=['geohash','lat','lon'])
        for g in missing:
            lat, lon = decode_geohash(g)
            tdf['d'] = np.sqrt((tdf['lat']-lat)**2+(tdf['lon']-lon)**2)
            gh_map[g] = tdf.sort_values('d').iloc[0]['geohash']
    test['mapped_gh'] = test['geohash'].map(lambda x: gh_map.get(x, x))

    # Temporal features
    def add_time(df):
        hrs, mins = [], []
        for t in df['timestamp']:
            h, m = map(int, t.split(':')); hrs.append(h); mins.append(m)
        df['hour'] = hrs; df['minute'] = mins
        df['time_of_day'] = df['hour'] + df['minute']/60.0
        df['sin_time'] = np.sin(2*np.pi*df['time_of_day']/24.0)
        df['cos_time'] = np.cos(2*np.pi*df['time_of_day']/24.0)
        return df
    train = add_time(train); test = add_time(test)

    train_48 = train[train['day']==48].copy()
    train_49 = train[train['day']==49].copy()

    # Overlap shifts using FULL 0:00-2:00 window
    ovlp = ['0:0','0:15','0:30','0:45','1:0','1:15','1:30','1:45','2:0']
    o48  = train_48[train_48['timestamp'].isin(ovlp)].groupby('geohash')['demand'].mean().reset_index()
    o49  = train_49[train_49['timestamp'].isin(ovlp)].groupby('geohash')['demand'].mean().reset_index()
    o48.columns = ['geohash','mean_o48']; o49.columns = ['geohash','mean_o49']
    shifts = o48.merge(o49, on='geohash', how='inner')
    shifts['shift_diff']  = shifts['mean_o49'] - shifts['mean_o48']
    shifts['shift_ratio'] = (shifts['mean_o49']+1e-5)/(shifts['mean_o48']+1e-5)
    global_shift_diff  = shifts['shift_diff'].mean()
    global_shift_ratio = shifts['shift_ratio'].mean()

    # Geohash-level stats from day 48
    gh_stats = train_48.groupby('geohash')['demand'].agg(['mean','std']).reset_index()
    gh_stats.columns = ['geohash','gh_mean','gh_std']
    global_mean = train_48['demand'].mean()

    # *** KEY ADDITION: geohash x hour mean from day 48 ***
    # Captures intra-day demand patterns per location - works equally well at any time of day
    gh_hr = train_48.groupby(['geohash','hour'])['demand'].mean().reset_index()
    gh_hr.columns = ['geohash','hour','gh_hr_mean']

    # Feature engineering function
    def build(df, geohash_col):
        d = df.copy()
        gh = d[geohash_col]

        # Day 48 demand at exact same timestamp
        d48_demand = train_48[['geohash','timestamp','demand']].rename(
            columns={'demand':'demand_day48','geohash':'_g'})
        d = d.merge(d48_demand, left_on=[geohash_col,'timestamp'],
                    right_on=['_g','timestamp'], how='left').drop(columns=['_g'],errors='ignore')

        # Shifts
        d = d.merge(shifts[['geohash','shift_diff','shift_ratio']].rename(columns={'geohash':'_g'}),
                    left_on=geohash_col, right_on='_g', how='left').drop(columns=['_g'],errors='ignore')
        d['shift_diff']  = d['shift_diff'].fillna(global_shift_diff)
        d['shift_ratio'] = d['shift_ratio'].fillna(global_shift_ratio)

        # Geohash stats
        d = d.merge(gh_stats.rename(columns={'geohash':'_g'}),
                    left_on=geohash_col, right_on='_g', how='left').drop(columns=['_g'],errors='ignore')

        # *** Geohash x hour mean (the only new feature vs v1) ***
        d = d.merge(gh_hr.rename(columns={'geohash':'_g'}),
                    left_on=[geohash_col,'hour'], right_on=['_g','hour'],
                    how='left').drop(columns=['_g'],errors='ignore')
        d['gh_hr_mean'] = d['gh_hr_mean'].fillna(d['gh_mean']).fillna(global_mean)

        # Fill demand_day48 with geohash mean / global mean
        d['demand_day48'] = d['demand_day48'].fillna(d['gh_mean']).fillna(global_mean)

        return d

    print("Building features...")
    df49  = build(train_49, 'geohash')
    df_test = build(test, 'mapped_gh')

    # Day 48 training set (no past day data, use gh_mean as demand_day48)
    df48 = train_48.copy()
    df48['demand_day48'] = df48['geohash'].map(train_48.groupby('geohash')['demand'].mean())
    df48['shift_diff']   = 0.0; df48['shift_ratio'] = 1.0
    df48 = df48.merge(gh_stats.rename(columns={'geohash':'_g'}),
                      left_on='geohash', right_on='_g', how='left').drop(columns=['_g'],errors='ignore')
    df48 = df48.merge(gh_hr.rename(columns={'geohash':'_g'}),
                      left_on=['geohash','hour'], right_on=['_g','hour'],
                      how='left').drop(columns=['_g'],errors='ignore')
    df48['gh_hr_mean'] = df48['gh_hr_mean'].fillna(df48['gh_mean']).fillna(global_mean)

    combined = pd.concat([df48, df49], ignore_index=True)

    # Encode categoricals
    cat_cols = ['RoadType','LargeVehicles','Landmarks','Weather']
    le_maps  = {}
    for col in cat_cols:
        combined[col] = combined[col].astype(str)
        le = LabelEncoder()
        combined[col+'_enc'] = le.fit_transform(combined[col])
        le_maps[col] = dict(zip(le.classes_, range(len(le.classes_))))
        for df in [df49, df_test]:
            df[col] = df[col].astype(str)
            df[col+'_enc'] = df[col].map(le_maps[col]).fillna(-1).astype(int)

    med_t = combined['Temperature'].median()
    med_l = combined['NumberofLanes'].median()
    for df in [combined, df49, df_test]:
        df['Temperature']   = df['Temperature'].fillna(med_t)
        df['NumberofLanes'] = df['NumberofLanes'].fillna(med_l)

    # Feature set: same as v1 + gh_hr_mean + gh_std
    feats = [
        'lat','lon','sin_time','cos_time','time_of_day',
        'demand_day48','shift_diff','shift_ratio',
        'gh_mean','gh_std','gh_hr_mean',
        'NumberofLanes','Temperature'
    ] + [c+'_enc' for c in cat_cols]
    print(f"Feature count: {len(feats)}")

    X_comb = combined[feats]; y_comb = combined['demand']
    X_49   = df49[feats];     y_49   = df49['demand']
    X_test = df_test[feats]

    # --- Ensemble: ET + RF + LGB (all on combined + day49 flavors) ---
    print("Training models...")

    # ExtraTrees on Combined
    et_c = ExtraTreesRegressor(n_estimators=300, max_features=0.7, min_samples_leaf=1, random_state=42, n_jobs=-1)
    et_c.fit(X_comb, y_comb)
    print("  ET(Combined) done")

    # ExtraTrees on Day49 only
    et_49 = ExtraTreesRegressor(n_estimators=300, max_features=0.7, min_samples_leaf=1, random_state=42, n_jobs=-1)
    et_49.fit(X_49, y_49)
    print("  ET(Day49) done")

    # RandomForest on Combined
    rf_c = RandomForestRegressor(n_estimators=300, max_features=0.5, random_state=42, n_jobs=-1)
    rf_c.fit(X_comb, y_comb)
    print("  RF(Combined) done")

    # LightGBM on Combined (well-regularized)
    lgb_c = lgb.LGBMRegressor(
        n_estimators=1000, learning_rate=0.02, max_depth=6, num_leaves=40,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=30,
        reg_alpha=0.2, reg_lambda=0.5, random_state=42, n_jobs=-1, verbosity=-1
    )
    lgb_c.fit(X_comb, y_comb)
    print("  LGB(Combined) done")

    # LightGBM on Day49 only
    lgb_49 = lgb.LGBMRegressor(
        n_estimators=1000, learning_rate=0.02, max_depth=6, num_leaves=40,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=30,
        reg_alpha=0.2, reg_lambda=0.5, random_state=42, n_jobs=-1, verbosity=-1
    )
    lgb_49.fit(X_49, y_49)
    print("  LGB(Day49) done")

    # Generate predictions
    print("Generating predictions...")
    p_et_c   = et_c.predict(X_test)
    p_et_49  = et_49.predict(X_test)
    p_rf_c   = rf_c.predict(X_test)
    p_lgb_c  = lgb_c.predict(X_test)
    p_lgb_49 = lgb_49.predict(X_test)

    # Weighted blend (ET-focused, same spirit as v1 + LGB)
    final = (0.25*p_et_c + 0.15*p_et_49 +
             0.15*p_rf_c +
             0.30*p_lgb_c + 0.15*p_lgb_49)
    final = np.clip(final, 0.0, 1.0)

    sub = pd.DataFrame({'Index': test_indices, 'demand': final})
    out = r"c:\Users\KIIT\Desktop\flipkartgrid\submission.csv"
    sub.to_csv(out, index=False)
    print(f"Submission saved: {out}")
    print("Shape:", sub.shape)
    print(sub.head())

    # Show ET feature importance
    imp = pd.Series(et_c.feature_importances_, index=feats).sort_values(ascending=False)
    print("\nTop features (ET Combined):")
    print(imp.head(10))

if __name__ == '__main__':
    main()
