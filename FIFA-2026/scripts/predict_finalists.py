#!/usr/bin/env python3
"""
predict_finalists.py

Reproducible script to predict World Cup finalists using historical match data and
current team metrics. Implements nested cross-validation, three classifiers
(Logistic Regression, Random Forest, XGBoost) and outputs top predicted finalists.

This script intentionally REMOVES target-derived features (finals/semis/quarters)
to avoid leakage when predicting future tournaments.

Usage:
    python scripts/predict_finalists.py

Outputs:
    outputs/finalists_top9.csv
    outputs/model_scores.json

"""
import os
import json
import warnings
from typing import List
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_data(hist_path: str, qualifiers_path: str = None) -> (pd.DataFrame, pd.DataFrame):
    hist = pd.read_csv(hist_path)
    qualifiers = None
    if qualifiers_path and os.path.exists(qualifiers_path):
        qualifiers = pd.read_csv(qualifiers_path)
    return hist, qualifiers


def build_team_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-team historical metrics from match-level data."""
    teams = {}
    for _, row in df.iterrows():
        for side in ['home', 'away']:
            team = row.get(f'{side}_team')
            if pd.isna(team):
                continue
            if team not in teams:
                teams[team] = {'team': team, 'matches': 0, 'wins': 0, 'draws': 0, 'losses': 0,
                               'goals_for': 0, 'goals_against': 0, 'finals': 0, 'semis': 0, 'quarters': 0}
            teams[team]['matches'] += 1
            if side == 'home':
                gf = row.get('home_score', 0)
                ga = row.get('away_score', 0)
            else:
                gf = row.get('away_score', 0)
                ga = row.get('home_score', 0)
            teams[team]['goals_for'] += (gf if pd.notnull(gf) else 0)
            teams[team]['goals_against'] += (ga if pd.notnull(ga) else 0)
            if pd.notnull(gf) and pd.notnull(ga):
                if gf > ga:
                    teams[team]['wins'] += 1
                elif gf == ga:
                    teams[team]['draws'] += 1
                else:
                    teams[team]['losses'] += 1
            r = str(row.get('Round', ''))
            if 'Final' in r:
                teams[team]['finals'] += 1
            if 'Semi' in r:
                teams[team]['semis'] += 1
            if 'Quarter' in r:
                teams[team]['quarters'] += 1
    profs = []
    for t, vals in teams.items():
        matches = vals['matches'] if vals['matches'] > 0 else 1
        profs.append({
            'team': t,
            'matches': matches,
            'win_rate': vals['wins'] / matches,
            'goals_for_per_match': vals['goals_for'] / matches,
            'goals_against_per_match': vals['goals_against'] / matches,
            'goal_diff_per_match': (vals['goals_for'] - vals['goals_against']) / matches,
            'finals': vals['finals'],
            'semis': vals['semis'],
            'quarters': vals['quarters']
        })
    return pd.DataFrame(profs).set_index('team')


def prepare_features(team_profiles: pd.DataFrame, age_df: pd.DataFrame = None, drop_leakage: bool = True) -> pd.DataFrame:
    df = team_profiles.copy()
    if age_df is not None:
        age_clean = age_df.rename(columns={'Team': 'team', 'Avg_Squad_Age': 'avg_squad_age'})
        age_latest = age_clean.sort_values('Year').groupby('team').last()
        df = df.join(age_latest[['avg_squad_age']], how='left')
    # create target
    df['is_finalist'] = ((df.get('finals', 0) > 0) | (df.get('semis', 0) > 0)).astype(int)
    if drop_leakage:
        # Remove target-derived columns to avoid leakage
        for col in ['finals', 'semis', 'quarters']:
            if col in df.columns:
                df = df.drop(columns=[col])
    return df.reset_index().rename(columns={'index': 'team'})


def select_and_scale(X: pd.DataFrame, y: pd.Series, k_best: int = 15):
    # Remove low variance
    vt = VarianceThreshold(threshold=0.0)
    X_var = vt.fit_transform(X)
    selected_cols = [c for i, c in enumerate(X.columns) if vt.get_support()[i]]
    # SelectKBest
    k = min(k_best, X_var.shape[1])
    skb = SelectKBest(mutual_info_classif, k=k)
    X_sel = skb.fit_transform(X[selected_cols], y)
    final_cols = [selected_cols[i] for i in skb.get_support(indices=True)]
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(pd.DataFrame(X_sel, columns=final_cols))
    return pd.DataFrame(X_scaled, columns=final_cols), final_cols, scaler


def nested_cv_models(X: pd.DataFrame, y: pd.Series):
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr_grid = {'C': [0.01, 0.1, 1.0], 'penalty': ['l2'], 'solver': ['liblinear', 'lbfgs']}
    gs_lr = GridSearchCV(lr, lr_grid, cv=inner, scoring='roc_auc', n_jobs=-1)
    scores_lr = cross_val_score(gs_lr, X, y, cv=outer, scoring='roc_auc', n_jobs=-1)
    gs_lr.fit(X, y)
    results['lr'] = {'model': gs_lr.best_estimator_, 'cv_scores': scores_lr.tolist()}

    # Random Forest
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    rf_grid = {'n_estimators': [100, 200], 'max_depth': [4, 6], 'min_samples_leaf': [2, 4]}
    gs_rf = GridSearchCV(rf, rf_grid, cv=inner, scoring='roc_auc', n_jobs=-1)
    scores_rf = cross_val_score(gs_rf, X, y, cv=outer, scoring='roc_auc', n_jobs=-1)
    gs_rf.fit(X, y)
    results['rf'] = {'model': gs_rf.best_estimator_, 'cv_scores': scores_rf.tolist()}

    # XGBoost if available
    if XGBClassifier is not None:
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE)
        xgb_grid = {'n_estimators': [100], 'max_depth': [3, 4], 'learning_rate': [0.01, 0.05]}
        gs_xgb = GridSearchCV(xgb, xgb_grid, cv=inner, scoring='roc_auc', n_jobs=-1)
        scores_xgb = cross_val_score(gs_xgb, X, y, cv=outer, scoring='roc_auc', n_jobs=-1)
        gs_xgb.fit(X, y)
        results['xgb'] = {'model': gs_xgb.best_estimator_, 'cv_scores': scores_xgb.tolist()}
    else:
        results['xgb'] = None

    return results


def ensemble_predict(team_df: pd.DataFrame, feature_cols: List[str], models: dict, scaler: StandardScaler, selector_cols: List[str]):
    # Prepare feature matrix for prediction
    Xp = team_df[feature_cols].fillna(0).copy()
    # Ensure columns match selector_cols
    Xp = Xp[selector_cols]
    Xp_scaled = scaler.transform(Xp)
    probs = {}
    probs['lr'] = models['lr']['model'].predict_proba(Xp_scaled)[:, 1]
    probs['rf'] = models['rf']['model'].predict_proba(Xp_scaled)[:, 1]
    if models.get('xgb'):
        probs['xgb'] = models['xgb']['model'].predict_proba(Xp_scaled)[:, 1]
    # build ensemble
    prob_arrays = [v for v in probs.values()]
    ensemble = np.mean(prob_arrays, axis=0)
    df = team_df[['team']].copy()
    df['prob_lr'] = probs['lr']
    df['prob_rf'] = probs['rf']
    df['ensemble_prob'] = ensemble
    return df


def main():
    # Use paths_util to discover files/dirs after user rearranged workspace
    try:
        import paths_util
    except Exception:
        paths_util = None

    os.makedirs('outputs', exist_ok=True)
    # discover historical cleaned dataset
    hist_candidate = paths_util.find_file('cleaned_data-set.csv') if paths_util else None
    hist_path = str(hist_candidate) if hist_candidate else 'FIFA-2026/Week-1/Data/cleaned_data-set.csv'
    age_candidate = paths_util.find_file('world_cup_2002_2022_squad_avg_age.csv') if paths_util else None
    age_path = str(age_candidate) if age_candidate else 'world_cup_2002_2022_squad_avg_age.csv'
    qualifiers_candidate = paths_util.find_file('fifa_2026_qualifiers.csv') if paths_util else None
    qualifiers_path = str(qualifiers_candidate) if qualifiers_candidate else 'FIFA-2026/Week-1/fifa_2026_qualifiers.csv'

    hist, qualifiers = load_data(hist_path, qualifiers_path)
    age_df = pd.read_csv(age_path) if os.path.exists(age_path) else None
    team_profiles = build_team_profiles(hist)
    # Prepare features without leakage
    features = prepare_features(team_profiles, age_df=age_df, drop_leakage=True)

    # We'll model using teams present in features
    # Exclude teams with very few matches (optional)
    features = features[features['matches'] >= 3].reset_index(drop=True)

    y = features['is_finalist']
    feature_cols = [c for c in features.columns if c not in ['team', 'is_finalist']]
    X = features[feature_cols].fillna(0)

    # Select and scale
    X_scaled_df, selected_cols, scaler = select_and_scale(X, y, k_best=12)

    # Nested CV models
    models = nested_cv_models(X_scaled_df, y)

    # Save model CV scores
    scores = {k: (v['cv_scores'] if v else None) for k, v in models.items()}
    with open('outputs/model_scores.json', 'w') as f:
        json.dump(scores, f, indent=2)

    # Build final ensemble predictions on the same teams (for demonstration)
    # In practice, predict on candidate teams only (confirmed excluded)
    preds = ensemble_predict(features[['team'] + selected_cols], feature_cols, models, scaler, selected_cols)
    preds_sorted = preds.sort_values('ensemble_prob', ascending=False)
    # Save finalists outputs to a dedicated folder to keep them separate from qualification outputs
    # choose outputs directory via paths_util if available
    if paths_util:
        out_dir = paths_util.find_outputs_dir()
    else:
        out_dir = Path('outputs') / 'finalists'
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_sorted.to_csv(out_dir / 'finalists_top_all.csv', index=False)

    # Top 9 finalists
    top9 = preds_sorted.head(9)
    top9.to_csv(out_dir / 'finalists_top9.csv', index=False)
    print(f'Saved {out_dir / "finalists_top9.csv"} and {out_dir / "finalists_top_all.csv"}')


if __name__ == '__main__':
    main()
