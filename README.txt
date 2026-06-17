Flipkart Gridlock Hackathon 2.0 - Traffic Demand Prediction
===========================================================

Overview of the Ultimate Approach (v16)
--------------------------------------
Our approach treats this as a time-series spatial regression task. The goal is to forecast future traffic demand (Day 49 from 2:15 to 13:45) at specific geohashes based on historical traffic demand (Day 48 and early Day 49).

In this final production version (predict.py), we implement an optimized hybrid ensemble:
1. **Bayesian Overlap Shrinkage (k = 2.0)**: Over 50% of geohashes are sparse. We apply Bayesian m-estimate shrinkage (with a smoothing parameter of k = 2.0) to smooth morning overlap (00:00 to 02:00) shift differences between Day 48 and Day 49, pulling local estimates toward the global average. We also compute an exponentially-weighted overlap shift feature (giving higher weights to time periods closer to 02:00).
2. **Robust Historical Imputation**: We map unseen geohashes in the test set to their spatially nearest train geohash. Missing Day 48 demand values are filled using a spatial fallback search or the geohash's diurnal average.
3. **Advanced Time-Lag Features**: We engineer ±15-minute time-lag features (`demand_day48_prev` and `demand_day48_next`) to capture temporal autocorrelation.
4. **Three-Model Regressor Ensemble**:
   We train an ensemble of ExtraTrees, RandomForest, and LightGBM regressors on the combined Day 48 & Day 49 feature set.
5. **Out-of-Fold (OOF) Grid Search Weights**:
   We run a 5-fold cross-validation grid search to optimize the blend weights:
   - ExtraTrees (ET) Weight: 0.80
   - LightGBM (LGB) Weight: 0.20
   - RandomForest (RF) Weight: 0.00
   - Baseline Blend Weight: 0.00
   This achieves a state-of-the-art local validation R² of 0.96400.

Feature Space (20 Dimensions)
----------------------------
1. Spatial: `lat`, `lon` (decoded from geohash).
2. Temporal: `sin_time`, `cos_time`, `time_of_day`.
3. Historical: `demand_day48`, `demand_day48_prev`, `demand_day48_next`.
4. Overlap & Shift: `shift_diff`, `shift_ratio`, `shift_diff_wmean`.
5. Geohash Stats: `gh_mean`, `gh_std` (volatility proxy).
6. Baseline: `baseline_pred`.
7. Contextual: `RoadType`, `LargeVehicles`, `Landmarks`, `Weather`, `Temperature`, `NumberofLanes`.

How to Run
----------
1. Ensure dependencies are installed:
   `pip install pandas numpy scikit-learn lightgbm`
2. Execute the prediction script:
   `python predict.py`
3. Output file `submission.csv` will be generated in the current directory.
