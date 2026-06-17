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

def ts_to_min(ts):
    h, m = map(int, ts.split(':'))
    return h * 60 + m

def min_to_ts(minutes):
    # Clamp to 0..1435 (23:45)
    minutes = max(0, min(1435, minutes))
    h = minutes // 60; m = minutes % 60
    # Snap to nearest 15-min slot
    m = round(m / 15) * 15
    if m == 60: h += 1; m = 0
    return f"{h}:{m}"

def main():
    print("Loading data...")
    data_dir = r"c:\Users\KIIT\Desktop\flipkartgrid\dataset\dataset"
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test  = pd.read_csv(os.path.join(data_dir, "test.csv"))
    test_indices = test['Index'].copy()

    # === Spatial coords ===
    def add_coords(df):
        lats, lons = [], []
        for g in df['geohash']:
            lat, lon = decode_geohash(g); lats.append(lat); lons.append(lon)
        df['lat'] = lats; df['lon'] = lons
        return df
    train = add_coords(train); test = add_coords(test)

    # === Map missing geohashes ===
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

    # === Temporal features ===
    def add_time(df):
        hrs, mins = [], []
        for t in df['timestamp']:
            h, m = map(int, t.split(':')); hrs.append(h); mins.append(m)
        df['hour'] = hrs; df['minute'] = mins
        df['t_min'] = df['hour']*60 + df['minute']
        df['tod']   = df['hour'] + df['minute']/60.0
        df['sin_t'] = np.sin(2*np.pi*df['tod']/24.0)
        df['cos_t'] = np.cos(2*np.pi*df['tod']/24.0)
        df['is_peak'] = (((df['hour']>=7)&(df['hour']<=10))|((df['hour']>=17)&(df['hour']<=20))).astype(int)
        return df
    train = add_time(train); test = add_time(test)

    train_48 = train[train['day']==48].copy()
    train_49 = train[train['day']==49].copy()

    # === Day 48 demand lookup: {(geohash, t_min) -> demand} ===
    print("Building Day 48 demand lookup...")
    d48_lookup = {}
    for _, row in train_48.iterrows():
        d48_lookup[(row['geohash'], row['t_min'])] = row['demand']

    def get_lag_demand(gh, t_min, offset_min, fallback):
        target_min = t_min + offset_min
        # Snap to nearest 15-min slot
        snapped = round(target_min / 15) * 15
        return d48_lookup.get((gh, snapped), fallback)

    # === Overlap shifts (time-weighted: more recent = higher weight) ===
    print("Computing shifts...")
    ovlp = ['0:0','0:15','0:30','0:45','1:0','1:15','1:30','1:45','2:0']
    ovlp_min = [ts_to_min(t) for t in ovlp]

    # Per-geohash, per-timestamp shift
    t49_ovlp = train_49[train_49['timestamp'].isin(ovlp)].copy()
    t48_ovlp = train_48[train_48['timestamp'].isin(ovlp)].copy()

    merged_ovlp = t49_ovlp[['geohash','t_min','demand']].merge(
        t48_ovlp[['geohash','t_min','demand']].rename(columns={'demand':'d48_o'}),
        on=['geohash','t_min'], how='inner'
    )
    merged_ovlp['ts_diff'] = merged_ovlp['demand'] - merged_ovlp['d48_o']
    # Time-weighted: weight = t_min / max_t_min (later timestamps weighted more)
    merged_ovlp['weight'] = (merged_ovlp['t_min'] + 1) / (merged_ovlp['t_min'].max() + 1)

    # Weighted shift per geohash
    def weighted_shift(grp):
        w = grp['weight']
        diff  = (grp['ts_diff'] * w).sum() / w.sum()
        ratio = ((grp['demand']+1e-5)/(grp['d48_o']+1e-5) * w).sum() / w.sum()
        return pd.Series({'shift_diff': diff, 'shift_ratio': ratio,
                          'mean49': grp['demand'].mean(), 'mean48': grp['d48_o'].mean()})

    shifts = merged_ovlp.groupby('geohash').apply(weighted_shift).reset_index()
    global_shift_diff  = shifts['shift_diff'].mean()
    global_shift_ratio = shifts['shift_ratio'].mean()

    # === Geohash stats from day 48 ===
    gh_stats = train_48.groupby('geohash')['demand'].agg(['mean','std','median','min','max']).reset_index()
    gh_stats.columns = ['geohash','gh_mean','gh_std','gh_med','gh_min','gh_max']
    global_mean = train_48['demand'].mean()

    gh_hr = train_48.groupby(['geohash','hour'])['demand'].mean().reset_index()
    gh_hr.columns = ['geohash','hour','gh_hr_mean']

    # === Neighbor geohash features (top-5 nearest) ===
    print("Computing neighbor features for day 48...")
    gh_list = list(gh_train)
    gh_coord = {g: decode_geohash(g) for g in gh_list}
    gh_coord_df = pd.DataFrame([(g, *gh_coord[g]) for g in gh_list], columns=['geohash','lat','lon'])

    K = 5
    gh_neighbors = {}
    for g in gh_list:
        lat, lon = gh_coord[g]
        gh_coord_df['d'] = np.sqrt((gh_coord_df['lat']-lat)**2+(gh_coord_df['lon']-lon)**2)
        top = gh_coord_df.sort_values('d').iloc[1:K+1]['geohash'].tolist()
        gh_neighbors[g] = top

    # Build neighbor demand lookup: for each (geohash, t_min) -> mean of neighbor demands from day 48
    print("Building neighbor lookup (this may take a moment)...")
    # Precompute per-geohash per-tmin demand mean from neighbors
    t_mins_all = train_48['t_min'].unique()
    # Group day48 by geohash and t_min
    d48_by_gh_t = train_48.groupby(['geohash','t_min'])['demand'].mean().to_dict()

    def get_neighbor_mean(gh, t_min):
        neighbors = gh_neighbors.get(gh, [])
        vals = [d48_by_gh_t.get((n, t_min), np.nan) for n in neighbors]
        vals = [v for v in vals if not np.isnan(v)]
        return np.mean(vals) if vals else global_mean

    # Precompute for speed: build per (geohash, t_min)
    neighbor_lookup = {}
    for g in gh_list:
        neighbors = gh_neighbors.get(g, [])
        for t in t_mins_all:
            vals = [d48_by_gh_t.get((n, t), np.nan) for n in neighbors]
            vals = [v for v in vals if not np.isnan(v)]
            neighbor_lookup[(g, t)] = np.mean(vals) if vals else global_mean

    print("Neighbor lookup built.")

    # === Feature builder ===
    def build(df, geohash_col):
        d = df.copy()
        # Map geohash for merging historical features
        d['_gh'] = d[geohash_col]

        # Merge shifts
        d = d.merge(shifts[['geohash','shift_diff','shift_ratio','mean49','mean48']].rename(columns={'geohash':'_g'}),
                    left_on='_gh', right_on='_g', how='left').drop(columns=['_g'],errors='ignore')
        d['shift_diff']  = d['shift_diff'].fillna(global_shift_diff)
        d['shift_ratio'] = d['shift_ratio'].fillna(global_shift_ratio)

        # Merge gh stats
        d = d.merge(gh_stats.rename(columns={'geohash':'_g'}), left_on='_gh', right_on='_g', how='left').drop(columns=['_g'],errors='ignore')
        d = d.merge(gh_hr.rename(columns={'geohash':'_g'}), left_on=['_gh','hour'], right_on=['_g','hour'], how='left').drop(columns=['_g'],errors='ignore')
        d['gh_hr_mean'] = d['gh_hr_mean'].fillna(d['gh_mean']).fillna(global_mean)

        # Exact lag d48
        d['d48'] = d.apply(lambda r: d48_lookup.get((r['_gh'], r['t_min']), np.nan), axis=1)
        d['d48'] = d['d48'].fillna(d['gh_mean']).fillna(global_mean)

        # Rolling lag features from day 48
        for offset in [-60, -45, -30, -15, 15, 30, 45, 60]:
            col = f'd48_lag_{offset}'
            d[col] = d.apply(lambda r: get_lag_demand(r['_gh'], r['t_min'], offset, np.nan), axis=1)
            d[col] = d[col].fillna(d['d48'])

        # Rolling mean of -60 to +60 from day 48
        lag_cols = [f'd48_lag_{o}' for o in [-60,-45,-30,-15,15,30,45,60]]
        d['d48_rolling_mean'] = d[lag_cols + ['d48']].mean(axis=1)
        d['d48_rolling_std']  = d[lag_cols + ['d48']].std(axis=1).fillna(0)

        # Neighbor mean from day 48
        d['neighbor_mean'] = d.apply(lambda r: neighbor_lookup.get((r['_gh'], r['t_min']),
                                     neighbor_lookup.get((r['_gh'], t_mins_all[0]), global_mean)), axis=1)

        # Adjusted predictions
        d['adj_diff']  = np.clip(d['d48'] + d['shift_diff'], 0, 1)
        d['adj_ratio'] = np.clip(d['d48'] * d['shift_ratio'], 0, 1)
        d['adj_blend'] = 0.6*d['adj_diff'] + 0.4*d['adj_ratio']
        d['adj_hr']    = np.clip(d['gh_hr_mean'] + d['shift_diff'], 0, 1)

        d.drop(columns=['_gh'], inplace=True, errors='ignore')
        return d

    print("Building train/test features...")
    df49 = build(train_49, 'geohash')

    df48 = build(train_48, 'geohash')
    df48['shift_diff']  = 0.0; df48['shift_ratio'] = 1.0
    df48['d48']         = df48['demand']
    for offset in [-60,-45,-30,-15,15,30,45,60]:
        df48[f'd48_lag_{offset}'] = df48.apply(
            lambda r: get_lag_demand(r['geohash'], r['t_min'], offset, r['demand']), axis=1)
    lag_cols = [f'd48_lag_{o}' for o in [-60,-45,-30,-15,15,30,45,60]]
    df48['d48_rolling_mean'] = df48[lag_cols+['d48']].mean(axis=1)
    df48['d48_rolling_std']  = df48[lag_cols+['d48']].std(axis=1).fillna(0)
    df48['neighbor_mean']    = df48.apply(lambda r: neighbor_lookup.get((r['geohash'],r['t_min']),global_mean), axis=1)
    df48['adj_diff']   = df48['d48']
    df48['adj_ratio']  = df48['d48']
    df48['adj_blend']  = df48['d48']
    df48['adj_hr']     = df48['gh_hr_mean'].fillna(global_mean)
    df48['mean49']     = df48['gh_mean']
    df48['mean48']     = df48['gh_mean']

    df_test = build(test, 'mapped_gh')
    combined = pd.concat([df48, df49], ignore_index=True)

    # Encode categoricals
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
        'lat','lon','sin_t','cos_t','tod','is_peak',
        'd48','shift_diff','shift_ratio',
        'gh_mean','gh_std','gh_med','gh_min','gh_max',
        'gh_hr_mean','mean48','mean49',
        'adj_diff','adj_ratio','adj_blend','adj_hr',
        'd48_rolling_mean','d48_rolling_std','neighbor_mean',
        'd48_lag_-60','d48_lag_-45','d48_lag_-30','d48_lag_-15',
        'd48_lag_15','d48_lag_30','d48_lag_45','d48_lag_60',
        'NumberofLanes','Temperature'
    ] + [c+'_e' for c in cat_cols]
    print(f"Total features: {len(feats)}")

    # Spatial 5-fold CV
    print("Running Spatial 5-Fold CV...")
    geohashes = df49['geohash'].unique()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    lgb_params = dict(
        n_estimators=1000, learning_rate=0.02,
        max_depth=8, num_leaves=63,
        subsample=0.7, colsample_bytree=0.7,
        min_child_samples=20,
        reg_alpha=0.1, reg_lambda=0.3,
        random_state=42, n_jobs=-1, verbosity=-1
    )
    for fold, (ti, vi) in enumerate(kf.split(geohashes)):
        tr_gh = geohashes[ti]; va_gh = geohashes[vi]
        Xtr = combined[combined['geohash'].isin(tr_gh)][feats]
        ytr = combined[combined['geohash'].isin(tr_gh)]['demand']
        Xva = df49[df49['geohash'].isin(va_gh)][feats]
        yva = df49[df49['geohash'].isin(va_gh)]['demand']
        lm  = lgb.LGBMRegressor(**lgb_params)
        lm.fit(Xtr, ytr)
        r2 = r2_score(yva, lm.predict(Xva))
        cv_scores.append(r2)
        print(f"  Fold {fold}: LGB R2 = {r2:.5f}")
    print(f"\nMean CV R2: {np.mean(cv_scores):.5f}")

    # Final training
    print("\nTraining final models on all data...")
    lgb_all = lgb.LGBMRegressor(**lgb_params)
    lgb_all.fit(combined[feats], combined['demand'])

    lgb_49 = lgb.LGBMRegressor(**lgb_params)
    lgb_49.fit(df49[feats], df49['demand'])

    et_all = ExtraTreesRegressor(n_estimators=200, max_features=0.6, random_state=42, n_jobs=-1)
    et_all.fit(combined[feats], combined['demand'])

    p1 = lgb_all.predict(df_test[feats])
    p2 = lgb_49.predict(df_test[feats])
    p3 = et_all.predict(df_test[feats])

    final = 0.40*p1 + 0.35*p2 + 0.25*p3
    final = np.clip(final, 0.0, 1.0)

    # Feature importance
    imp = pd.Series(lgb_all.feature_importances_, index=feats).sort_values(ascending=False)
    print("\nTop 15 Features (LGB):")
    print(imp.head(15))

    sub = pd.DataFrame({'Index': test_indices, 'demand': final})
    out = r"c:\Users\KIIT\Desktop\flipkartgrid\submission.csv"
    sub.to_csv(out, index=False)
    print(f"\nDone! Submission saved to {out}")
    print("Shape:", sub.shape)
    print(sub.head())

if __name__ == '__main__':
    main()
