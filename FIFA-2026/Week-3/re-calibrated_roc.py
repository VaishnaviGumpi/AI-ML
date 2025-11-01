from pathlib import Path
import importlib.util
import joblib

# Load train_models module
train_path = Path(__file__).resolve().parent / 'train_models.py'
spec = importlib.util.spec_from_file_location('train_models_mod', str(train_path))
train_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mod)

BASE = Path(__file__).resolve().parent.parent / 'reports' / 'model_eval_full'

def main():
    # Recreate X_test, y_test using helper flow from train_models
    df_stats, df_top4, df_quals = train_mod.load_data()
    team_placement_feats = train_mod.compute_team_placement_features(df_stats, df_top4)
    if not team_placement_feats.empty and 'Team' in df_stats.columns:
        df_stats = df_stats.merge(team_placement_feats, how='left', on='Team')
    matches_path = Path(__file__).resolve().parent.parent / 'data' / 'cleaned' / 'matches_clean.csv'
    team_year_feats, team_recent_overall = train_mod.aggregate_team_history(matches_path, recent_n=10)
    if not team_year_feats.empty and 'Year' in df_stats.columns and 'Team' in df_stats.columns:
        df_stats = df_stats.merge(team_year_feats, how='left', left_on=['Team', 'Year'], right_on=['Team', 'Year'])
    if not team_recent_overall.empty:
        rename_map = {c: c.replace('_overall','') for c in team_recent_overall.columns if c.endswith('_overall')}
        team_recent = team_recent_overall.rename(columns=rename_map)
        if 'Team' in df_stats.columns:
            df_stats = df_stats.merge(team_recent, how='left', on='Team')

    df_stats = train_mod.make_label_finalist(df_stats, df_top4)
    if 'Year' in df_stats.columns:
        df_stats = df_stats.dropna(subset=['Year'])
    if 'Team' in df_stats.columns:
        df_stats = df_stats.dropna(subset=['Team'])

    X, y, feature_cols = train_mod.select_features(df_stats)
    # recreate same split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    cal_path = BASE / 'RandomForest_calibrated.joblib'
    if not cal_path.exists():
        print('Calibrated model not found at', cal_path)
        return
    model = joblib.load(cal_path)
    print('Loaded calibrated model from', cal_path)

    # Use train_models helper to save single-model eval (creates confusion and ROC)
    train_mod.save_single_model_eval(model, 'RandomForest_calibrated', X_test, y_test, BASE)
    print('Regenerated ROC and confusion for RandomForest_calibrated')

if __name__ == '__main__':
    main()
