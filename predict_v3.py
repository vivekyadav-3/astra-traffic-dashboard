import pandas as pd
import numpy as np
import os
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import lightgbm as lgb

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
                if bit == 1: lon_interval[0] = mid
                else: lon_interval[1] = mid
            else:
                mid = (lat_interval[0] + lat_interval[1]) / 2
                if bit == 1: lat_interval[0] = mid
                else: lat_interval[1] = mid
            is_even = not is_even
    return (lat_interval[0] + lat_interval[1]) / 2, (lon_interval[0] + lon_interval[1]) / 2

def main():
    print("Loading data...")
    data_dir = r"c:\Users\KIIT\Desktop\flipkartgrid\dataset\dataset"
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test  = pd.read_csv(os.path.join(data_dir, "test.csv"))
    test_indices = test['Index'].copy()

    # --- Spatial coords ---
    def add_coords(df):
        lats, lons = [], []
        for g in df['geohash']:
            lat, lon = decode_geohash(g)
            lats.append(lat); lons.append(lon)
        df['lat'] = lats; df['lon'] = lons
        return df
    train = add_coords(train); test = add_coords(test)

    # --- Map missing test geohashes to nearest training geohash ---
    gh_train = set(train['geohash'].unique())
    gh_test  = set(test['geohash'].unique())
    missing  = list(gh_test - gh_train)
    gh_map   = {}
    if missing:
        tc  = [(g, *decode_geohash(g)) for g in gh_train]
        tdf = pd.DataFrame(tc, columns=['geohash','lat','lon'])
        for g in missing:
            lat, lon = decode_geohash(g)
            tdf['d'] = np.sqrt((tdf['lat']-lat)**2+(tdf['lon']-lon)**2)
            gh_map[g] = tdf.sort_values('d').iloc[0]['geohash']
    test['mapped_gh'] = test['geohash'].map(lambda x: gh_map.get(x, x))

    # --- Temporal features ---
    def add_time(df):
        hrs, mins = [], []
        for t in df['timestamp']:
            h, m = map(int, t.split(':'))
            hrs.append(h); mins.append(m)
        df['hour'] = hrs; df['minute'] = mins
        df['tod']  = df['hour'] + df['minute']/60.0
        df['sin_t'] = np.sin(2*np.pi*df['tod']/24.0)
        df['cos_t'] = np.cos(2*np.pi*df['tod']/24.0)
        return df
    train = add_time(train); test = add_time(test)

    train_48 = train[train['day']==48].copy()
    train_49 = train[train['day']==49].copy()

    # --- Overlap shifts (full 0:0–2:0 window) ---
    ovlp = ['0:0','0:15','0:30','0:45','1:0','1:15','1:30','1:45','2:0']
    o48 = train_48[train_48['timestamp'].isin(ovlp)].groupby('geohash')['demand'].mean().reset_index()
    o48.columns = ['geohash','mean48']
    o49 = train_49[train_49['timestamp'].isin(ovlp)].groupby('geohash')['demand'].mean().reset_index()
    o49.columns = ['geohash','mean49']
    shifts = o48.merge(o49, on='geohash', how='inner')
    shifts['shift_diff']  = shifts['mean49'] - shifts['mean48']
    shifts['shift_ratio'] = (shifts['mean49']+1e-5)/(shifts['mean48']+1e-5)

    # --- Geohash-level stats from day 48 ---
    gh_stats = train_48.groupby('geohash')['demand'].agg(['mean','std','median']).reset_index()
    gh_stats.columns = ['geohash','gh_mean','gh_std','gh_med']
    global_mean = train_48['demand'].mean()

    # --- Geohash x Hour stats from day 48 ---
    gh_hr = train_48.groupby(['geohash','hour'])['demand'].mean().reset_index()
    gh_hr.columns = ['geohash','hour','gh_hr_mean']

    # --- Feature builder ---
    def build(df, geohash_col):
        d  = df.copy()
        # lag demand from day 48
        lag = train_48[['geohash','timestamp','demand']].rename(
            columns={'demand':'d48','geohash':'_g'})
        d  = d.merge(lag, left_on=[geohash_col,'timestamp'], right_on=['_g','timestamp'], how='left').drop(columns=['_g'],errors='ignore')
        # shifts
        d  = d.merge(shifts.rename(columns={'geohash':'_g'}), left_on=geohash_col, right_on='_g', how='left').drop(columns=['_g'],errors='ignore')
        # gh stats
        d  = d.merge(gh_stats.rename(columns={'geohash':'_g'}), left_on=geohash_col, right_on='_g', how='left').drop(columns=['_g'],errors='ignore')
        # gh x hour
        d  = d.merge(gh_hr.rename(columns={'geohash':'_g'}), left_on=[geohash_col,'hour'], right_on=['_g','hour'], how='left').drop(columns=['_g'],errors='ignore')

        d['shift_diff']  = d['shift_diff'].fillna(0.0)
        d['shift_ratio'] = d['shift_ratio'].fillna(1.0)
        d['d48']         = d['d48'].fillna(d['gh_mean']).fillna(global_mean)
        d['gh_hr_mean']  = d['gh_hr_mean'].fillna(d['gh_mean']).fillna(global_mean)
        # Direct adjusted preds – these are very strong features
        d['adj_diff']    = np.clip(d['d48'] + d['shift_diff'], 0, 1)
        d['adj_ratio']   = np.clip(d['d48'] * d['shift_ratio'], 0, 1)
        d['adj_blend']   = 0.7*d['adj_diff'] + 0.3*d['adj_ratio']
        return d

    print("Building datasets...")
    df49 = build(train_49, 'geohash')

    df48 = train_48.copy()
    df48['d48']        = df48['geohash'].map(train_48.groupby('geohash')['demand'].mean())
    df48['shift_diff'] = 0.0;  df48['shift_ratio'] = 1.0
    df48 = df48.merge(gh_stats.rename(columns={'geohash':'_g'}), left_on='geohash', right_on='_g', how='left').drop(columns=['_g'],errors='ignore')
    df48 = df48.merge(gh_hr.rename(columns={'geohash':'_g'}), left_on=['geohash','hour'], right_on=['_g','hour'], how='left').drop(columns=['_g'],errors='ignore')
    df48['gh_hr_mean'] = df48['gh_hr_mean'].fillna(df48['gh_mean']).fillna(global_mean)
    df48['adj_diff']   = df48['d48'].fillna(global_mean)
    df48['adj_ratio']  = df48['d48'].fillna(global_mean)
    df48['adj_blend']  = df48['d48'].fillna(global_mean)
    df48['mean48']     = df48['gh_mean'];  df48['mean49'] = df48['gh_mean']

    df_test = build(test, 'mapped_gh')
    combined = pd.concat([df48, df49], ignore_index=True)

    # --- Encode categoricals ---
    cat_cols = ['RoadType','LargeVehicles','Landmarks','Weather']
    le_maps  = {}
    for col in cat_cols:
        combined[col] = combined[col].astype(str)
        le = LabelEncoder()
        combined[col+'_e'] = le.fit_transform(combined[col])
        le_maps[col] = dict(zip(le.classes_, range(len(le.classes_))))
        for df in [df49, df_test]:
            df[col] = df[col].astype(str)
            df[col+'_e'] = df[col].map(le_maps[col]).fillna(-1).astype(int)

    med_t = combined['Temperature'].median()
    med_l = combined['NumberofLanes'].median()
    for df in [combined, df49, df_test]:
        df['Temperature']   = df['Temperature'].fillna(med_t)
        df['NumberofLanes'] = df['NumberofLanes'].fillna(med_l)

    feats = [
        'lat','lon','sin_t','cos_t','tod',
        'd48','shift_diff','shift_ratio',
        'gh_mean','gh_std','gh_med','gh_hr_mean',
        'mean48','mean49',
        'adj_diff','adj_ratio','adj_blend',
        'NumberofLanes','Temperature'
    ] + [c+'_e' for c in cat_cols]

    # --- Spatial 5-Fold CV ---
    print("Running 5-fold spatial CV...")
    geohashes = df49['geohash'].unique()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_et, cv_lgb = [], []

    lgb_params = dict(
        n_estimators=800, learning_rate=0.03,
        max_depth=7, num_leaves=50,
        subsample=0.7, colsample_bytree=0.7,
        min_child_samples=30,
        reg_alpha=0.2, reg_lambda=0.5,
        random_state=42, n_jobs=-1, verbosity=-1
    )

    for fold, (ti, vi) in enumerate(kf.split(geohashes)):
        tr_gh = geohashes[ti]; va_gh = geohashes[vi]
        Xtr = combined[combined['geohash'].isin(tr_gh)][feats]
        ytr = combined[combined['geohash'].isin(tr_gh)]['demand']
        Xva = df49[df49['geohash'].isin(va_gh)][feats]
        yva = df49[df49['geohash'].isin(va_gh)]['demand']

        et  = ExtraTreesRegressor(n_estimators=200, max_features=0.6, random_state=42, n_jobs=-1)
        et.fit(Xtr, ytr)
        r2_et = r2_score(yva, et.predict(Xva))
        cv_et.append(r2_et)

        lm  = lgb.LGBMRegressor(**lgb_params)
        lm.fit(Xtr, ytr)
        r2_lgb = r2_score(yva, lm.predict(Xva))
        cv_lgb.append(r2_lgb)

        print(f"  Fold {fold}: ET={r2_et:.5f}  LGB={r2_lgb:.5f}")

    print(f"\nMean ET  CV R2: {np.mean(cv_et):.5f}")
    print(f"Mean LGB CV R2: {np.mean(cv_lgb):.5f}")

    # --- Final training on ALL data ---
    print("\nTraining final models on full data...")
    X_all = combined[feats]; y_all = combined['demand']
    X_49  = df49[feats];     y_49  = df49['demand']
    X_tst = df_test[feats]

    et_all = ExtraTreesRegressor(n_estimators=300, max_features=0.6, random_state=42, n_jobs=-1)
    et_all.fit(X_all, y_all)

    et_49 = ExtraTreesRegressor(n_estimators=300, max_features=0.6, random_state=42, n_jobs=-1)
    et_49.fit(X_49, y_49)

    lgb_all = lgb.LGBMRegressor(**lgb_params)
    lgb_all.fit(X_all, y_all)

    lgb_49 = lgb.LGBMRegressor(**lgb_params)
    lgb_49.fit(X_49, y_49)

    # Weighted blend: LGB is generally better for generalization
    p_et_all  = et_all.predict(X_tst)
    p_et_49   = et_49.predict(X_tst)
    p_lgb_all = lgb_all.predict(X_tst)
    p_lgb_49  = lgb_49.predict(X_tst)

    final = 0.30*p_et_all + 0.20*p_et_49 + 0.30*p_lgb_all + 0.20*p_lgb_49
    final = np.clip(final, 0.0, 1.0)

    sub = pd.DataFrame({'Index': test_indices, 'demand': final})
    out = r"c:\Users\KIIT\Desktop\flipkartgrid\submission.csv"
    sub.to_csv(out, index=False)
    print(f"\nSubmission saved → {out}")
    print("Shape:", sub.shape)
    print(sub.head())

if __name__ == '__main__':
    main()
