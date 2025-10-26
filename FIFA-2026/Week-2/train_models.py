"""Train simple classification models to predict World Cup finalists.

This script builds two baseline classifiers (Logistic Regression and
Random Forest) using historical per-team tournament features and attempts
to predict whether a team is a finalist (winner or runner-up) in a given
tournament year. It also loads a 2026 qualifiers dataset (if present),
maps features, and writes predicted finalist probabilities for 2026 teams.

Usage:
    python3 fifa_2026/wc2026-finalists/scripts/train_models.py

Outputs:
    - prints CV results, best params, and test evaluation
    - saves `qualifiers_predictions_2026.csv` to the cleaned data folder

Notes:
    - The script prefers `world_cup_all_teams_complete_analysis.csv` at the
      repository root (per-team features for all teams). It falls back to
      `world_cup_complete_analysis_all_positions.csv` if the all-teams file
      is not present (limited to top-4 teams only).
    - The script is written to be readable and well-commented for teaching
      / assignment use; feel free to adapt feature selection and tuning.
"""

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.calibration import calibration_curve


def find_stats_file():
    """Return path to the preferred per-team stats file (all-teams preferred)."""
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    all_teams = repo_root / 'world_cup_all_teams_complete_analysis.csv'
    top4 = repo_root / 'world_cup_complete_analysis_all_positions.csv'
    if all_teams.exists():
        return all_teams
    if top4.exists():
        return top4
    raise FileNotFoundError(f"Neither {all_teams} nor {top4} found")


def load_data():
    """Load per-team stats and qualifiers.

    Returns:
        df_stats: DataFrame with per-team per-year features (training data)
        df_quals: DataFrame with 2026 qualifiers features (may be empty)
    """
    stats_path = find_stats_file()
    print(f"Loading per-team stats from: {stats_path}")
    df_stats = pd.read_csv(stats_path)
    df_stats.columns = df_stats.columns.str.strip()

    # basic team name trimming
    if 'Team' in df_stats.columns:
        df_stats['Team'] = df_stats['Team'].astype(str).str.strip()

    # try to load the top-4 placements file (used to derive Finalist labels)
    top4_path = Path(__file__).resolve().parent.parent.parent.parent / 'world_cup_complete_analysis_all_positions.csv'
    if top4_path.exists():
        df_top4 = pd.read_csv(top4_path)
        df_top4.columns = df_top4.columns.str.strip()
        if 'Team' in df_top4.columns:
            df_top4['Team'] = df_top4['Team'].astype(str).str.strip()
        print(f"Loaded top-4 placements from: {top4_path} (rows: {len(df_top4)})")
    else:
        df_top4 = pd.DataFrame()

    # load qualifiers (if present)
    quals_path = Path(__file__).resolve().parent.parent / 'data' / 'raw' / 'fifa_2026_qualifiers_player_analysis.csv'
    if quals_path.exists():
        df_quals = pd.read_csv(quals_path)
        df_quals.columns = df_quals.columns.str.strip()
        # trim team names if present
        if 'Team' in df_quals.columns:
            df_quals['Team'] = df_quals['Team'].astype(str).str.strip()
        print(f"Loaded qualifiers data: {quals_path} (rows: {len(df_quals)})")
    else:
        df_quals = pd.DataFrame()
        print(f"No qualifiers file found at {quals_path}; continuing without qualifiers predictions")

    return df_stats, df_top4, df_quals


def aggregate_team_history(matches_path: Path, recent_n: int = 10):
    """Aggregate per-team historical features from matches_clean.csv.

    Returns two DataFrames:
      - team_year_feats: per (Team, Year) aggregated features (tournament-level)
      - team_recent_feats: per Team aggregated recent-N-match features up to last year

    The function creates match-level rows for both home and away teams and
    computes metrics such as goals_per_game and win rate for the year and
    recent N matches (chronological by Year).
    """
    if not matches_path.exists():
        print(f"Matches file not found: {matches_path} — skipping history aggregation")
        return pd.DataFrame(), pd.DataFrame()

    m = pd.read_csv(matches_path)
    # normalize column names
    m.columns = m.columns.str.strip()

    # Build long-form match rows (one row per team per match)
    home_cols = ['Year', 'home_team', 'home_score', 'away_score', 'home_xg']
    away_cols = ['Year', 'away_team', 'away_score', 'home_score', 'away_xg']
    home_df = m[[c for c in home_cols if c in m.columns]].copy()
    home_df = home_df.rename(columns={
        'home_team': 'Team', 'home_score': 'goals_for', 'away_score': 'goals_against', 'home_xg': 'xg'
    })
    away_df = m[[c for c in away_cols if c in m.columns]].copy()
    away_df = away_df.rename(columns={
        'away_team': 'Team', 'away_score': 'goals_for', 'home_score': 'goals_against', 'away_xg': 'xg'
    })

    long = pd.concat([home_df, away_df], ignore_index=True, sort=False)
    # result: win=1 if goals_for>goals_against, draw=0.5, loss=0
    long['goals_for'] = pd.to_numeric(long['goals_for'], errors='coerce')
    long['goals_against'] = pd.to_numeric(long['goals_against'], errors='coerce')
    long['xg'] = pd.to_numeric(long.get('xg', pd.Series()), errors='coerce')
    long = long.dropna(subset=['Team'])
    long['Team'] = long['Team'].astype(str).str.strip()

    def result_win(gf, ga):
        if pd.isna(gf) or pd.isna(ga):
            return np.nan
        if gf > ga:
            return 1
        if gf == ga:
            return 0
        return 0

    long['is_win'] = long.apply(lambda r: result_win(r['goals_for'], r['goals_against']), axis=1)

    # tournament (year) aggregates per Team-Year
    team_year = long.groupby(['Team', 'Year']).agg(
        year_matches=('goals_for', 'count'),
        year_goals_for=('goals_for', 'sum'),
        year_goals_against=('goals_against', 'sum'),
        year_wins=('is_win', 'sum')
    ).reset_index()
    team_year['year_goals_for_pg'] = team_year['year_goals_for'] / team_year['year_matches']
    team_year['year_goals_against_pg'] = team_year['year_goals_against'] / team_year['year_matches']
    team_year['year_win_rate'] = team_year['year_wins'] / team_year['year_matches']

    # recent-N matches per Team up to each Year (exclude matches in the Year itself)
    teams = long['Team'].unique()
    recent_list = []
    for team in teams:
        tdf = long[long['Team'] == team].sort_values(['Year']).reset_index(drop=True)
        years = sorted(tdf['Year'].dropna().unique())
        for Y in years:
            prev = tdf[tdf['Year'] < Y]
            tail = prev.tail(recent_n)
            recent_matches = len(tail)
            if recent_matches == 0:
                rec_goals_for_pg = np.nan
                rec_goals_against_pg = np.nan
                rec_win_rate = np.nan
                rec_avg_xg = np.nan
            else:
                rec_goals_for_pg = tail['goals_for'].sum() / recent_matches
                rec_goals_against_pg = tail['goals_against'].sum() / recent_matches
                rec_win_rate = tail['is_win'].sum() / recent_matches
                rec_avg_xg = tail['xg'].mean()
            recent_list.append({
                'Team': team, 'Year': Y,
                'recent_n': recent_matches,
                'recent_goals_for_pg': rec_goals_for_pg,
                'recent_goals_against_pg': rec_goals_against_pg,
                'recent_win_rate': rec_win_rate,
                'recent_avg_xg': rec_avg_xg
            })

    recent_df = pd.DataFrame(recent_list)

    # Also compute overall recent features up to the latest Year (useful for qualifiers)
    latest_year = int(long['Year'].dropna().max())
    overall_recent = []
    for team in teams:
        tdf = long[(long['Team'] == team) & (long['Year'] <= latest_year)].sort_values(['Year']).reset_index(drop=True)
        tail = tdf.tail(recent_n)
        recent_matches = len(tail)
        if recent_matches == 0:
            rec_goals_for_pg = np.nan
            rec_goals_against_pg = np.nan
            rec_win_rate = np.nan
            rec_avg_xg = np.nan
        else:
            rec_goals_for_pg = tail['goals_for'].sum() / recent_matches
            rec_goals_against_pg = tail['goals_against'].sum() / recent_matches
            rec_win_rate = tail['is_win'].sum() / recent_matches
            rec_avg_xg = tail['xg'].mean()
        overall_recent.append({'Team': team,
                               'recent_n_overall': recent_matches,
                               'recent_goals_for_pg_overall': rec_goals_for_pg,
                               'recent_goals_against_pg_overall': rec_goals_against_pg,
                               'recent_win_rate_overall': rec_win_rate,
                               'recent_avg_xg_overall': rec_avg_xg})

    team_recent_overall = pd.DataFrame(overall_recent)

    # return team_year (per Team-Year) and team_recent_overall (per Team)
    return team_year, team_recent_overall


def make_label_finalist(df_stats, df_top4=pd.DataFrame()):
    """Create binary label `Finalist` indicating winner or runner-up.

    The function handles both numeric and string `Position` columns.
    """
    def is_final(pos):
        try:
            if pd.isna(pos):
                return False
            p = int(pos)
            return p <= 2
        except Exception:
            s = str(pos).lower()
            return ('winner' in s) or ('runner' in s) or ('runner-up' in s) or (s.strip() in ['1', '2'])

    # If we have a top-4 placements table, use it to label finalists
    if not df_top4.empty and 'Year' in df_top4.columns and 'Team' in df_top4.columns:
        # derive finalists from top4 file
        df_top4['Finalist'] = df_top4.get('Position', df_top4.get('position', pd.NA)).apply(is_final)
        finalists = df_top4[df_top4['Finalist'] == True][['Year', 'Team']].drop_duplicates()
        # left join to stats; default Finalist False
        df_stats = df_stats.merge(finalists.assign(Finalist=True), how='left', on=['Year', 'Team'])
        df_stats['Finalist'] = df_stats['Finalist'].fillna(False)
        return df_stats

    # fallback: if stats contain a Position column, use it
    if 'Position' in df_stats.columns:
        df_stats['Finalist'] = df_stats['Position'].apply(is_final)
        return df_stats

    # otherwise no reliable label available
    df_stats['Finalist'] = False
    return df_stats


def compute_team_placement_features(df_stats, df_top4=pd.DataFrame(), alpha: float = 0.12):
    """Compute placement-based aggregated features per Team.

    Returns a DataFrame indexed by Team with columns:
      - tournaments_played
      - participation_freq (relative to number of tournament years)
      - top4_count, top4_rate
      - top4_time_weighted (recent top-4s weighted by recency)
      - avg_finish (mean numeric Position when available)
      - stage_R16_count, stage_QF_count, stage_SF_count, stage_F_count
      - stage_R16_rate, ... (counts divided by tournaments_played)

    alpha controls time decay for time-weighted top4 (higher -> faster decay).
    """
    teams = pd.Series(df_stats['Team'].dropna().unique())
    if teams.empty:
        return pd.DataFrame()

    # Total distinct tournament years available
    total_years = int(df_stats['Year'].nunique()) if 'Year' in df_stats.columns else 0

    # tournaments played
    tournaments_played = df_stats.dropna(subset=['Year']).groupby('Team')['Year'].nunique().rename('tournaments_played')

    # top4 count / rate
    if not df_top4.empty and 'Team' in df_top4.columns:
        top4_counts = df_top4.groupby('Team').size().rename('top4_count')
    else:
        # try to derive from Position in df_stats if available
        if 'Position' in df_stats.columns:
            try:
                pos_numeric = pd.to_numeric(df_stats['Position'], errors='coerce')
                top4_counts = df_stats[pos_numeric <= 4].groupby('Team').size().rename('top4_count')
            except Exception:
                top4_counts = pd.Series(dtype=int, name='top4_count')
        else:
            top4_counts = pd.Series(dtype=int, name='top4_count')

    # time-weighted top4 score (uses df_top4 Year when available)
    if not df_top4.empty and 'Year' in df_top4.columns:
        max_year = int(df_top4['Year'].max())
        df_top4 = df_top4.dropna(subset=['Year'])
        df_top4['Year'] = pd.to_numeric(df_top4['Year'], errors='coerce')
        df_top4 = df_top4.dropna(subset=['Year'])
        df_top4['decay_w'] = (df_top4['Year'].apply(lambda y: np.exp(-alpha * (max_year - int(y)))))
        top4_time = df_top4.groupby('Team')['decay_w'].sum().rename('top4_time_weighted')
    else:
        top4_time = pd.Series(dtype=float, name='top4_time_weighted')

    # avg finish and stage counts if Position exists
    if 'Position' in df_stats.columns:
        pos = pd.to_numeric(df_stats['Position'], errors='coerce')
        pos_df = df_stats.assign(_pos=pos)

        avg_finish = pos_df.groupby('Team')['_pos'].mean().rename('avg_finish')

        def pos_to_stage(p):
            if pd.isna(p):
                return None
            p = int(p)
            if p <= 2:
                return 'F'
            if p <= 4:
                return 'SF'
            if p <= 8:
                return 'QF'
            if p <= 16:
                return 'R16'
            return 'GS'

        pos_df['stage'] = pos_df['_pos'].apply(pos_to_stage)
        stage_counts = pos_df.groupby(['Team', 'stage']).size().unstack(fill_value=0)
        # ensure columns for expected stages
        for st in ['R16', 'QF', 'SF', 'F', 'GS']:
            if st not in stage_counts.columns:
                stage_counts[st] = 0
        # rename columns
        stage_counts = stage_counts.rename(columns={
            'R16': 'stage_R16_count', 'QF': 'stage_QF_count', 'SF': 'stage_SF_count', 'F': 'stage_F_count', 'GS': 'stage_GS_count'
        })
    else:
        avg_finish = pd.Series(dtype=float, name='avg_finish')
        stage_counts = pd.DataFrame()

    # assemble features
    df_list = [tournaments_played]
    if not top4_counts.empty:
        df_list.append(top4_counts)
    if not top4_time.empty:
        df_list.append(top4_time)
    if not avg_finish.empty:
        df_list.append(avg_finish)
    if not stage_counts.empty:
        df_list.append(stage_counts)

    team_feats = pd.concat(df_list, axis=1)
    team_feats = team_feats.fillna(0)

    # derived rates
    if total_years > 0:
        team_feats['participation_freq'] = team_feats['tournaments_played'] / total_years
    else:
        team_feats['participation_freq'] = 0

    if 'top4_count' in team_feats.columns:
        team_feats['top4_rate'] = team_feats['top4_count'] / team_feats['tournaments_played'].replace(0, np.nan)
        team_feats['top4_rate'] = team_feats['top4_rate'].fillna(0)

    # stage rates
    for st in ['stage_R16_count', 'stage_QF_count', 'stage_SF_count', 'stage_F_count', 'stage_GS_count']:
        if st in team_feats.columns:
            team_feats[st.replace('_count', '_rate')] = team_feats[st] / team_feats['tournaments_played'].replace(0, np.nan)
            team_feats[st.replace('_count', '_rate')] = team_feats[st.replace('_count', '_rate')].fillna(0)

    team_feats = team_feats.reset_index().rename(columns={'index': 'Team'}) if 'Team' not in team_feats.columns else team_feats.reset_index()
    # ensure Team column exists as string
    if 'Team' not in team_feats.columns:
        team_feats = team_feats.reset_index().rename(columns={'index': 'Team'})
    team_feats['Team'] = team_feats['Team'].astype(str).str.strip()
    return team_feats


def select_features(df):
    """Choose a set of numeric features commonly useful for team performance.

    Returns feature matrix X and label y.
    """
    candidates = [
        'Matches', 'Wins', 'Draws', 'Losses',
        'Goals_For', 'Goals_Against', 'Goal_Difference',
        'Win_Rate_Percent', 'Avg_Squad_Age', 'Avg_Experience_Caps'
    ]
    # placement-based engineered features
    placement_feats = [
        'tournaments_played', 'participation_freq', 'top4_count', 'top4_rate', 'top4_time_weighted', 'avg_finish',
        'stage_R16_count', 'stage_QF_count', 'stage_SF_count', 'stage_F_count',
        'stage_R16_rate', 'stage_QF_rate', 'stage_SF_rate', 'stage_F_rate'
    ]
    candidates += placement_feats
    # also include aggregated history features if present
    extra_candidates = [c for c in df.columns if str(c).startswith('recent_') or str(c).startswith('year_')]
    all_candidates = candidates + extra_candidates
    existing = [c for c in all_candidates if c in df.columns]
    if not existing:
        raise ValueError('No expected feature columns found in stats file; inspect file and update feature list')

    X = df[existing].copy()
    y = df['Finalist'].astype(int)
    return X, y, existing


def train_and_evaluate(X, y):
    """Train two classifiers with simple hyperparameter tuning and evaluate.

    Returns fitted best estimators and a small summary dict.
    """
    # split holdout test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # pipeline for logistic regression: impute + scale + LR
    lr_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000, solver='liblinear', class_weight='balanced'))
    ])

    lr_params = {
        'clf__C': [0.01, 0.1, 1, 10]
    }

    rf_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])

    rf_params = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 5, 10]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print('\nStarting GridSearchCV for Logistic Regression...')
    lr_search = GridSearchCV(lr_pipe, lr_params, cv=cv, scoring='roc_auc', n_jobs=-1)
    lr_search.fit(X_train, y_train)
    print('LR best params:', lr_search.best_params_)
    print('LR best CV ROC-AUC:', lr_search.best_score_)

    print('\nStarting GridSearchCV for Random Forest...')
    rf_search = GridSearchCV(rf_pipe, rf_params, cv=cv, scoring='roc_auc', n_jobs=-1)
    rf_search.fit(X_train, y_train)
    print('RF best params:', rf_search.best_params_)
    print('RF best CV ROC-AUC:', rf_search.best_score_)

    # Evaluate on holdout test set
    lr_best = lr_search.best_estimator_
    rf_best = rf_search.best_estimator_

    def evaluate(model, X_test, y_test, name):
        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)
        auc = roc_auc_score(y_test, probs)
        print(f"\n=== Evaluation: {name} ===")
        print('ROC-AUC on test:', round(auc, 4))
        print('Classification report:')
        print(classification_report(y_test, preds, zero_division=0))
        print('Confusion matrix:')
        print(confusion_matrix(y_test, preds))

    evaluate(lr_best, X_test, y_test, 'LogisticRegression')
    evaluate(rf_best, X_test, y_test, 'RandomForest')

    # compute and return test split and models for further evaluation
    return lr_best, rf_best, X_test, X_train, y_test, y_train


def predict_qualifiers(model, feature_cols, df_quals, out_path):
    """Predict finalist probabilities for the qualifiers DataFrame and save results.

    The function will attempt to find matching feature columns in the qualifiers
    data; missing columns are filled with training medians by the model pipeline.
    """
    if df_quals.empty:
        print('No qualifiers data to predict.')
        return

    # try to locate a team-name column
    team_col_candidates = [c for c in ['Team', 'team', 'team_name'] if c in df_quals.columns]
    team_col = team_col_candidates[0] if team_col_candidates else None

    # Attempt to map qualifier columns to expected feature names.
    # Common variants:
    alias_map = {
        'Avg_International_Caps': 'Avg_Experience_Caps',
        'Win_Rate_%': 'Win_Rate_Percent',
        'Win_Rate': 'Win_Rate_Percent',
        'Avg_Squad_Age': 'Avg_Squad_Age'
    }

    # build normalized lookup for df_quals columns
    def normalize(name: str) -> str:
        return ''.join(ch.lower() for ch in str(name) if ch.isalnum())

    quals_cols_norm = {normalize(c): c for c in df_quals.columns}

    Xq = pd.DataFrame()
    for feat in feature_cols:
        # 1) exact match
        if feat in df_quals.columns:
            Xq[feat] = pd.to_numeric(df_quals[feat], errors='coerce')
            continue
        # 2) alias map
        mapped = None
        for k, v in alias_map.items():
            if v == feat and k in df_quals.columns:
                mapped = k
                break
        if mapped:
            Xq[feat] = pd.to_numeric(df_quals[mapped], errors='coerce')
            print(f"Mapped qualifier column '{mapped}' -> feature '{feat}'")
            continue
        # 3) normalized fuzzy match
        nfeat = normalize(feat)
        if nfeat in quals_cols_norm:
            colname = quals_cols_norm[nfeat]
            Xq[feat] = pd.to_numeric(df_quals[colname], errors='coerce')
            print(f"Normalized match: qualifier column '{colname}' -> feature '{feat}'")
            continue
        # 4) not found: fill NaN (imputer will handle)
        Xq[feat] = np.nan

    # debug: show how many non-missing values are present per feature
    non_null_counts = {c: int(Xq[c].notna().sum()) for c in Xq.columns}
    print('Qualifier features non-null counts:', non_null_counts)

    # use model to predict probabilities
    probs = model.predict_proba(Xq)[:, 1]
    # debug: print raw probability values (first 10) to inspect distribution
    try:
        import numpy as _np
        _np.set_printoptions(precision=6, suppress=True)
        print('Sample predicted finalist probabilities:', probs[:10])
    except Exception:
        print('Predicted probabilities (first 10):', probs[:10])
    results = pd.DataFrame({
        'Team': df_quals[team_col] if team_col is not None else df_quals.index.astype(str),
        'Finalist_Prob': probs
    })
    results = results.sort_values('Finalist_Prob', ascending=False).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f'Qualifiers predictions saved to: {out_path}')


def save_evaluation_and_report(lr_model, rf_model, X_test, y_test, feature_cols, out_dir: Path):
    """Save metrics, confusion matrices, ROC curves and a short markdown report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = out_dir / 'figures'
    figs_dir.mkdir(parents=True, exist_ok=True)

    eval_rows = []
    models = [('LogisticRegression', lr_model), ('RandomForest', rf_model)]
    for name, model in models:
        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)
        auc_score = roc_auc_score(y_test, probs)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        eval_rows.append({'model': name, 'roc_auc': float(auc_score), 'accuracy': float(acc), 'precision': float(prec), 'recall': float(rec), 'f1': float(f1)})

        # confusion matrix plot
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        cm_path = figs_dir / f'confusion_{name}.png'
        plt.savefig(cm_path)
        plt.close()

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend(loc='lower right')
        roc_path = figs_dir / f'roc_{name}.png'
        plt.tight_layout()
        plt.savefig(roc_path)
        plt.close()

        # save model
        model_path = out_dir / f'{name}.joblib'
        joblib.dump(model, model_path)

    # save metrics table
    metrics_df = pd.DataFrame(eval_rows)
    metrics_df.to_csv(out_dir / 'metrics_table.csv', index=False)

    # write a short markdown report
    report_lines = []
    report_lines.append('# Model evaluation report')
    report_lines.append('')
    report_lines.append('## Summary metrics')
    report_lines.append('')
    # render metrics_df as a simple markdown table (avoid optional dependency on tabulate)
    cols = metrics_df.columns.tolist()
    header = '| ' + ' | '.join(cols) + ' |'
    sep = '| ' + ' | '.join(['---'] * len(cols)) + ' |'
    report_lines.append(header)
    report_lines.append(sep)
    for _, r in metrics_df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                vals.append(f'{v:.4f}')
            else:
                vals.append(str(v))
        report_lines.append('| ' + ' | '.join(vals) + ' |')
    report_lines.append('')
    report_lines.append('## Notes')
    report_lines.append('')
    report_lines.append('- Models evaluated: Logistic Regression and Random Forest. Metrics shown above include accuracy, precision, recall, F1 and ROC-AUC on the holdout test set.')
    report_lines.append('- Confusion matrix and ROC curve PNGs are saved in the `figures/` directory.')
    report_lines.append('')
    report_lines.append('## Practical comparison and recommendation')
    report_lines.append('')
    report_lines.append('Logistic Regression tends to be well-calibrated and interpretable; Random Forest can capture non-linear interactions and may give higher AUC but worse recall for the positive class in this small dataset. Given the cost of false negatives (missed finalists), consider choosing the model with higher recall or calibrating thresholds to favor recall if that aligns with the project objectives.')
    report_lines.append('')
    (out_dir / 'model_evaluation.md').write_text('\n'.join(report_lines))
    print(f'Model evaluation artifacts saved to: {out_dir}')
    # generate diagnostics for both models
    try:
        generate_diagnostics(lr_model, 'LogisticRegression', X_test, y_test, out_dir)
    except Exception as e:
        print('Could not generate diagnostics for LogisticRegression:', e)
    try:
        generate_diagnostics(rf_model, 'RandomForest', X_test, y_test, out_dir)
    except Exception as e:
        print('Could not generate diagnostics for RandomForest:', e)


def generate_diagnostics(model, name, X_test, y_test, out_dir: Path):
    """Create probability histograms, calibration plots and threshold sweep CSV for a model."""
    out_dir.mkdir(parents=True, exist_ok=True)
    figs = out_dir / 'figures'
    figs.mkdir(parents=True, exist_ok=True)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    # Probability histogram (positives vs negatives)
    dfp = pd.DataFrame({'prob': probs, 'y': y_test.values})
    plt.figure(figsize=(6, 4))
    sns.histplot(dfp[dfp['y'] == 0]['prob'], color='C0', label='neg', kde=False, stat='density', bins=20, alpha=0.6)
    sns.histplot(dfp[dfp['y'] == 1]['prob'], color='C1', label='pos', kde=False, stat='density', bins=20, alpha=0.6)
    plt.legend()
    plt.xlabel('Predicted probability')
    plt.title(f'Probability histogram - {name}')
    plt.tight_layout()
    plt.savefig(figs / f'prob_hist_{name}.png')
    plt.close()

    # Calibration / reliability plot
    try:
        frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)
        plt.figure(figsize=(5, 5))
        plt.plot(mean_pred, frac_pos, 's-', label='calibration')
        plt.plot([0, 1], [0, 1], 'k--', label='perfect')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title(f'Calibration plot - {name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(figs / f'calibration_{name}.png')
        plt.close()
    except Exception as e:
        print('Calibration plot failed for', name, e)

    # Threshold sweep for precision/recall/f1
    threshs = np.linspace(0, 1, 101)
    rows = []
    for t in threshs:
        p = (probs >= t).astype(int)
        rows.append({'threshold': float(t), 'precision': float(precision_score(y_test, p, zero_division=0)), 'recall': float(recall_score(y_test, p, zero_division=0)), 'f1': float(f1_score(y_test, p, zero_division=0))})
    thr_df = pd.DataFrame(rows)
    thr_df.to_csv(out_dir / f'threshold_sweep_{name}.csv', index=False)

    print(f'Diagnostics saved for {name} in {out_dir}')


def save_single_model_eval(model, name: str, X_test, y_test, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = out_dir / 'figures'
    figs_dir.mkdir(parents=True, exist_ok=True)

    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    auc_score = roc_auc_score(y_test, probs)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    # confusion matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    cm_path = figs_dir / f'confusion_{name}.png'
    plt.savefig(cm_path)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
    plt.legend(loc='lower right')
    roc_path = figs_dir / f'roc_{name}.png'
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()

    # save metrics to CSV
    metrics_df = pd.DataFrame([{
        'model': name,
        'roc_auc': float(auc_score),
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1)
    }])
    metrics_df.to_csv(out_dir / f'metrics_{name}.csv', index=False)
    # save model
    joblib.dump(model, out_dir / f'{name}.joblib')
    print(f'Saved single-model evaluation for {name} to {out_dir}')
    # generate additional diagnostics
    try:
        generate_diagnostics(model, name, X_test, y_test, out_dir)
    except Exception as e:
        print('Could not generate diagnostics for', name, e)


def main():
    # Load stats, top4 (if present), and qualifiers
    df_stats, df_top4, df_quals = load_data()

    # Compute and merge placement-based team features (top4 freq, avg finish, stage counts)
    team_placement_feats = compute_team_placement_features(df_stats, df_top4)
    if not team_placement_feats.empty:
        # merge into df_stats
        if 'Team' in df_stats.columns:
            df_stats = df_stats.merge(team_placement_feats, how='left', on='Team')
        # also merge into qualifiers so qualifiers inherit placement features when possible
        if not df_quals.empty and 'Team' in df_quals.columns:
            df_quals = df_quals.merge(team_placement_feats, how='left', on='Team')

    # Aggregate history from matches_clean.csv and merge into stats + qualifiers
    matches_path = Path(__file__).resolve().parent.parent / 'data' / 'cleaned' / 'matches_clean.csv'
    team_year_feats, team_recent_overall = aggregate_team_history(matches_path, recent_n=10)
    if not team_year_feats.empty:
        # merge tournament-year features into df_stats on Team & Year
        if 'Year' in df_stats.columns and 'Team' in df_stats.columns:
            df_stats = df_stats.merge(team_year_feats, how='left', left_on=['Team', 'Year'], right_on=['Team', 'Year'])
        # merge overall recent features into qualifiers
    if not team_recent_overall.empty:
        # rename overall recent columns to match training recent_ names
        rename_map = {}
        for col in team_recent_overall.columns:
            if col.endswith('_overall'):
                rename_map[col] = col.replace('_overall', '')
        team_recent = team_recent_overall.rename(columns=rename_map)
        # merge recent (per-team overall) features into training stats as well
        if 'Team' in df_stats.columns:
            df_stats = df_stats.merge(team_recent, how='left', on='Team')
        # also merge into qualifiers (if present) so qualifiers get the same recent features
        if not df_quals.empty:
            df_quals = df_quals.merge(team_recent, how='left', on='Team')

    # Prepare label (use top-4 placements to derive finalists when available)
    df_stats = make_label_finalist(df_stats, df_top4)

    # drop rows missing Year or Team if present
    if 'Year' in df_stats.columns:
        df_stats = df_stats.dropna(subset=['Year'])

    # remove rows without a Team value
    if 'Team' in df_stats.columns:
        df_stats = df_stats.dropna(subset=['Team'])

    # choose features and labels
    X, y, feature_cols = select_features(df_stats)
    print(f"Training samples: {len(X)}, positive (finalists): {y.sum()}")

    # train and evaluate
    lr_model, rf_model, X_test, X_train, y_test, y_train = train_and_evaluate(X, y)

    # Predict qualifiers using the Random Forest (often better for tabular data)
    out_pred = Path(__file__).resolve().parent.parent / 'data' / 'cleaned' / 'qualifiers_predictions_2026.csv'
    # Determine which features are actually present in the qualifiers file
    if not df_quals.empty:
        # alias map same as used in predict_qualifiers
        alias_map = {
            'Avg_International_Caps': 'Avg_Experience_Caps',
            'Win_Rate_%': 'Win_Rate_Percent',
            'Win_Rate': 'Win_Rate_Percent',
            'Avg_Squad_Age': 'Avg_Squad_Age'
        }
        def normalize(name: str) -> str:
            return ''.join(ch.lower() for ch in str(name) if ch.isalnum())
        quals_cols_norm = {normalize(c): c for c in df_quals.columns}

        shared_features = []
        for feat in feature_cols:
            if feat in df_quals.columns:
                shared_features.append(feat)
                continue
            # alias
            mapped = None
            for k, v in alias_map.items():
                if v == feat and k in df_quals.columns:
                    mapped = k
                    break
            if mapped:
                shared_features.append(feat)
                continue
            # normalized match
            if normalize(feat) in quals_cols_norm:
                shared_features.append(feat)
                continue

        print('Shared features between training and qualifiers:', shared_features)

        # If there are fewer shared features than original, retrain a small RF on the shared features
        if shared_features and len(shared_features) < len(feature_cols):
            print('Retraining a RandomForest on shared features for qualifiers prediction...')
            # prepare X_shared and y
            X_shared = df_stats[shared_features].copy()
            # simple imputation of medians for any missing training values
            X_shared = X_shared.fillna(X_shared.median())
            rf_shared = RandomForestClassifier(random_state=42, n_estimators=200, class_weight='balanced')
            rf_shared.fit(X_shared, df_stats['Finalist'].astype(int))
            # use rf_shared for predicting qualifiers
            predict_qualifiers(rf_shared, shared_features, df_quals, out_pred)
            # save evaluation artifacts using the shared-RF and the held-out test
            # evaluate only the retrained RF using the reduced feature set
            save_single_model_eval(rf_shared, 'RandomForest_shared', X_test[shared_features].fillna(0), y_test, Path(__file__).resolve().parent.parent / 'reports' / 'model_eval')
            # also save a full evaluation report for the original LR and RF (using the full test set)
            try:
                save_evaluation_and_report(lr_model, rf_model, X_test, y_test, feature_cols, Path(__file__).resolve().parent.parent / 'reports' / 'model_eval_full')
            except Exception as e:
                print('Warning: could not save full evaluation report:', e)
            return

    # otherwise use the original RF trained on all features
    predict_qualifiers(rf_model, feature_cols, df_quals, out_pred)
    # Save evaluation artifacts for the final trained models
    save_evaluation_and_report(lr_model, rf_model, X_test, y_test, feature_cols, Path(__file__).resolve().parent.parent / 'reports' / 'model_eval')


if __name__ == '__main__':
    main()
