from pathlib import Path
import importlib.util
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# Load train_models module
train_path = Path(__file__).resolve().parent / 'train_models.py'
spec = importlib.util.spec_from_file_location('train_models_mod', str(train_path))
train_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mod)

OUT_DIR = Path(__file__).resolve().parent.parent / 'reports' / 'model_eval_full'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data and prepare features
print('Loading data...')
df_stats, df_top4, df_quals = train_mod.load_data()
team_placement_feats = train_mod.compute_team_placement_features(df_stats, df_top4)
if not team_placement_feats.empty and 'Team' in df_stats.columns:
    df_stats = df_stats.merge(team_placement_feats, how='left', on='Team')
# aggregate history
matches_path = Path(__file__).resolve().parent.parent / 'data' / 'cleaned' / 'matches_clean.csv'
team_year_feats, team_recent_overall = train_mod.aggregate_team_history(matches_path, recent_n=10)
if not team_year_feats.empty and 'Year' in df_stats.columns and 'Team' in df_stats.columns:
    df_stats = df_stats.merge(team_year_feats, how='left', left_on=['Team', 'Year'], right_on=['Team', 'Year'])
if not team_recent_overall.empty:
    rename_map = {}
    for col in team_recent_overall.columns:
        if col.endswith('_overall'):
            rename_map[col] = col.replace('_overall', '')
    team_recent = team_recent_overall.rename(columns=rename_map)
    if 'Team' in df_stats.columns:
        df_stats = df_stats.merge(team_recent, how='left', on='Team')

# make label
df_stats = train_mod.make_label_finalist(df_stats, df_top4)
if 'Year' in df_stats.columns:
    df_stats = df_stats.dropna(subset=['Year'])
if 'Team' in df_stats.columns:
    df_stats = df_stats.dropna(subset=['Team'])

X, y, feature_cols = train_mod.select_features(df_stats)
print(f'Total samples: {len(X)}, positives: {int(y.sum())}')

# Recreate holdout
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
print('Holdout: n_test=', len(X_test), 'positives in test=', int(y_test.sum()))

# Load existing RandomForest model if present
rf_path = OUT_DIR / 'RandomForest.joblib'
if not rf_path.exists():
    print('RandomForest.joblib not found in', rf_path, '-- training fresh small RF')
    rf_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', RandomForestClassifier(random_state=42, n_estimators=200, class_weight='balanced'))
    ])
    rf_pipe.fit(X_train, y_train)
    rf = rf_pipe
else:
    rf = joblib.load(rf_path)
    print('Loaded RandomForest from', rf_path)

# Calibrate RF probabilities
print('Calibrating RandomForest probabilities with CalibratedClassifierCV (sigmoid, cv=3)')
calibrator = CalibratedClassifierCV(rf, cv=3, method='sigmoid')
try:
    calibrator.fit(X_train, y_train)
except Exception as e:
    print('Calibration failed:', e)
    calibrator = None

if calibrator is not None:
    # Save calibrated model
    cal_path = OUT_DIR / 'RandomForest_calibrated.joblib'
    joblib.dump(calibrator, cal_path)
    print('Saved calibrated RF to', cal_path)

    # Evaluate on holdout
    probs_cal = calibrator.predict_proba(X_test)[:, 1]
    auc_cal = roc_auc_score(y_test, probs_cal)
    print('Calibrated RF AUC on test:', auc_cal)

    # Threshold sweep
    threshs = np.linspace(0, 1, 101)
    rows = []
    for t in threshs:
        p = (probs_cal >= t).astype(int)
        rows.append({'threshold': float(t), 'precision': float(precision_score(y_test, p, zero_division=0)), 'recall': float(recall_score(y_test, p, zero_division=0)), 'f1': float(f1_score(y_test, p, zero_division=0))})
    thr_df = pd.DataFrame(rows)
    thr_out = OUT_DIR / 'threshold_sweep_RandomForest_calibrated.csv'
    thr_df.to_csv(thr_out, index=False)
    best_row = thr_df.loc[thr_df['f1'].idxmax()]
    best_t = float(best_row['threshold'])
    print('Best threshold (calibrated) by F1:', best_t, best_row.to_dict())

    # Apply best threshold and compute metrics
    preds_best = (probs_cal >= best_t).astype(int)
    prec = precision_score(y_test, preds_best, zero_division=0)
    rec = recall_score(y_test, preds_best, zero_division=0)
    f1 = f1_score(y_test, preds_best, zero_division=0)
    acc = accuracy_score(y_test, preds_best)
    aucv = roc_auc_score(y_test, probs_cal)

    # Save metrics table append
    metrics_path = OUT_DIR / 'metrics_table.csv'
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
    else:
        metrics_df = pd.DataFrame()
    new_row = {'model': 'RandomForest_calibrated', 'roc_auc': float(aucv), 'accuracy': float(acc), 'precision': float(prec), 'recall': float(rec), 'f1': float(f1)}
    metrics_df = pd.concat([metrics_df, pd.DataFrame([new_row])], ignore_index=True)
    metrics_df.to_csv(metrics_path, index=False)
    print('Updated metrics_table.csv with calibrated RF metrics at best threshold')

    # Generate diagnostics using existing helper
    try:
        train_mod.generate_diagnostics(calibrator, 'RandomForest_calibrated', X_test, y_test, OUT_DIR)
    except Exception as e:
        print('generate_diagnostics failed for calibrated RF:', e)

# Repeated Stratified CV summaries for LR and RF pipelines
print('\nRunning repeated stratified CV to report mean/std for ROC, recall, F1 (this may take a moment)')
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

# define pipelines similar to training
lr_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=2000, solver='liblinear', class_weight='balanced'))
])
rf_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('clf', RandomForestClassifier(random_state=42, n_estimators=200, class_weight='balanced'))
])

scoring = {'roc_auc': 'roc_auc', 'recall': 'recall', 'f1': 'f1'}

try:
    lr_cv = cross_validate(lr_pipe, X, y, cv=cv, scoring=scoring, return_train_score=False, n_jobs=-1)
    rf_cv = cross_validate(rf_pipe, X, y, cv=cv, scoring=scoring, return_train_score=False, n_jobs=-1)

    def summarize(cv_res):
        return {k: (float(np.mean(v)), float(np.std(v))) for k, v in cv_res.items() if k.startswith('test_')}

    summary_lr = summarize(lr_cv)
    summary_rf = summarize(rf_cv)
    cv_summary = pd.DataFrame([{'model': 'LogisticRegression', 'roc_auc_mean': summary_lr['test_roc_auc'][0], 'roc_auc_std': summary_lr['test_roc_auc'][1], 'recall_mean': summary_lr['test_recall'][0], 'recall_std': summary_lr['test_recall'][1], 'f1_mean': summary_lr['test_f1'][0], 'f1_std': summary_lr['test_f1'][1]},
                               {'model': 'RandomForest', 'roc_auc_mean': summary_rf['test_roc_auc'][0], 'roc_auc_std': summary_rf['test_roc_auc'][1], 'recall_mean': summary_rf['test_recall'][0], 'recall_std': summary_rf['test_recall'][1], 'f1_mean': summary_rf['test_f1'][0], 'f1_std': summary_rf['test_f1'][1]}])
    cv_summary.to_csv(OUT_DIR / 'cv_summary_repeatedstratified.csv', index=False)
    print('Saved CV summary to', OUT_DIR / 'cv_summary_repeatedstratified.csv')
except Exception as e:
    print('Repeated CV failed or interrupted:', e)

print('\nImprovement script finished')
