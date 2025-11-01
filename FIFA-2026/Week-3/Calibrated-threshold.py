from pathlib import Path
import numpy as np
import pandas as pd
import importlib.util
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score

# load train_models helpers
train_path = Path(__file__).resolve().parent / 'train_models.py'
spec = importlib.util.spec_from_file_location('tm', str(train_path))
tm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tm)

OUT_DIR = Path(__file__).resolve().parent.parent / 'reports' / 'model_eval_full'
QUAL_OUT = Path(__file__).resolve().parent.parent / 'data' / 'cleaned' / 'qualifiers_predictions_2026.csv'

# Load data and features
print('Loading data...')
df_stats, df_top4, df_quals = tm.load_data()
team_placement_feats = tm.compute_team_placement_features(df_stats, df_top4)
if not team_placement_feats.empty and 'Team' in df_stats.columns:
    df_stats = df_stats.merge(team_placement_feats, how='left', on='Team')
# aggregate history
matches_path = Path(__file__).resolve().parent.parent / 'data' / 'cleaned' / 'matches_clean.csv'
team_year_feats, team_recent_overall = tm.aggregate_team_history(matches_path, recent_n=10)
if not team_year_feats.empty and 'Year' in df_stats.columns and 'Team' in df_stats.columns:
    df_stats = df_stats.merge(team_year_feats, how='left', left_on=['Team', 'Year'], right_on=['Team', 'Year'])
if not team_recent_overall.empty:
    rename_map = {c: c.replace('_overall', '') for c in team_recent_overall.columns if c.endswith('_overall')}
    team_recent = team_recent_overall.rename(columns=rename_map)
    if 'Team' in df_stats.columns:
        df_stats = df_stats.merge(team_recent, how='left', on='Team')

# labels and features
df_stats = tm.make_label_finalist(df_stats, df_top4)
if 'Year' in df_stats.columns:
    df_stats = df_stats.dropna(subset=['Year'])
if 'Team' in df_stats.columns:
    df_stats = df_stats.dropna(subset=['Team'])

X, y, feature_cols = tm.select_features(df_stats)
print('Samples:', len(X), 'positives:', int(y.sum()))

# repeated CV to select thresholds
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
thresholds = []
fold = 0
for train_idx, val_idx in rskf.split(X, y):
    fold += 1
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # fit calibrated RF on training fold
    base = RandomForestClassifier(random_state=42, n_estimators=200, class_weight='balanced')
    # use simple imputer pipeline for robustness
    pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('clf', base)])
    # calibrate with internal cv=3
    calib = CalibratedClassifierCV(pipe, cv=3, method='sigmoid')
    try:
        calib.fit(X_train, y_train)
    except Exception as e:
        print('Calibration failed on fold', fold, e)
        continue

    probs = calib.predict_proba(X_val)[:, 1]
    # threshold sweep
    thr_values = np.linspace(0, 1, 101)
    best_t = 0.0
    best_f1 = -1.0
    for t in thr_values:
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    print(f'Fold {fold}: best threshold {best_t:.3f} with F1 {best_f1:.3f}')
    thresholds.append(best_t)

if not thresholds:
    print('No thresholds collected; aborting')
    raise SystemExit

# aggregate threshold (median recommended)
med_t = float(np.median(thresholds))
mean_t = float(np.mean(thresholds))
print('Thresholds collected (count=', len(thresholds), '); median=', med_t, 'mean=', mean_t)

# Retrain calibrated RF on full data
base = RandomForestClassifier(random_state=42, n_estimators=200, class_weight='balanced')
pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('clf', base)])
calib_full = CalibratedClassifierCV(pipe, cv=3, method='sigmoid')
calib_full.fit(X, y)
joblib.dump(calib_full, OUT_DIR / 'RandomForest_calibrated_retrained.joblib')
print('Saved retrained calibrated RF')

# Apply to qualifiers if available
if df_quals.empty:
    print('No qualifiers data; done')
else:
    # build Xq as in train_models.predict_qualifiers
    def normalize(name: str) -> str:
        return ''.join(ch.lower() for ch in str(name) if ch.isalnum())
    alias_map = {
        'Avg_International_Caps': 'Avg_Experience_Caps',
        'Win_Rate_%': 'Win_Rate_Percent',
        'Win_Rate': 'Win_Rate_Percent',
        'Avg_Squad_Age': 'Avg_Squad_Age'
    }
    quals_cols_norm = {normalize(c): c for c in df_quals.columns}
    Xq = pd.DataFrame()
    for feat in feature_cols:
        if feat in df_quals.columns:
            Xq[feat] = pd.to_numeric(df_quals[feat], errors='coerce')
            continue
        mapped = None
        for k, v in alias_map.items():
            if v == feat and k in df_quals.columns:
                mapped = k
                break
        if mapped:
            Xq[feat] = pd.to_numeric(df_quals[mapped], errors='coerce')
            continue
        nfeat = normalize(feat)
        if nfeat in quals_cols_norm:
            Xq[feat] = pd.to_numeric(df_quals[quals_cols_norm[nfeat]], errors='coerce')
            continue
        Xq[feat] = np.nan

    # predict probabilities
    probs_q = calib_full.predict_proba(Xq)[:, 1]
    # choose threshold: use median threshold from CV
    chosen_t = med_t
    preds_q = (probs_q >= chosen_t).astype(int)
    out = pd.DataFrame({'Team': df_quals.get('Team', df_quals.index.astype(str)), 'Finalist_Prob': probs_q, 'Finalist_Pred': preds_q})
    QUAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(QUAL_OUT, index=False)
    print('Saved qualifiers predictions to', QUAL_OUT, 'using threshold', chosen_t)

# save thresholds summary
thr_df = pd.DataFrame({'fold_threshold': thresholds})
thr_df.to_csv(OUT_DIR / 'thresholds_cv_folds.csv', index=False)
print('Saved thresholds per fold to', OUT_DIR / 'thresholds_cv_folds.csv')
print('Done')
