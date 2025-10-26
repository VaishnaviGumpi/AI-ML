This document summarizes the modeling pipeline implemented for Task 2: predicting World
Cup finalists.
Data sources
- Per-team stats: `world_cup_all_teams_complete_analysis.csv` (preferred) or fallback
`world_cup_complete_analysis_all_positions.csv`.
- Top 4 placements (for label derivation):
`world_cup_complete_analysis_all_positions.csv`.
- Matches history (cleaned): `fifa_2026/wc2026-finalists/data/cleaned/matches_clean.csv`.
- Qualifiers input:
`fifa_2026/wc2026-finalists/data/raw/fifa_2026_qualifiers_player_analysis.csv`.
Label
- Binary label `Finalist` indicates Winner or Runner-up (derived by merging top 4
placements into per-team stats).
Feature engineering
- Base numeric features: Matches, Wins, Draws, Losses, Goals_For, Goals_Against,
Goal_Difference, Win_Rate_Percent, Avg_Squad_Age, Avg_Experience_Caps.
- Placement features: tournaments_played, participation_freq, top4_count, top4_rate,
top4_time_weighted, avg_finish, stage counts and rates (R16/QF/SF/F).
- Match-history features: per Team-Year aggregates (year_matches, year_goals_for_pg,
year_win_rate) and recent-N-match aggregates (recent_goals_for_pg, recent_win_rate,
recent_avg_xg).
Preprocessing
- Median imputation for missing numeric values (SimpleImputer).
- Standard scaling applied for Logistic Regression.
- RandomForest uses imputed values directly (tree-based models do not need scaling).
Models and tuning
- Logistic Regression: `liblinear`, `class_weight='balanced'`. Grid search over C=[0.01,
0.1, 1, 10].
- Random Forest: `class_weight='balanced'`, `n_estimators` in [100,200], `max_depth` in
[None,5,10].
- Cross-validation: StratifiedKFold(n_splits=5) optimizing ROC-AUC.
- Holdout: stratified 20% test split for final evaluation
- Validation and diagnostics
- Metrics recorded: accuracy, precision, recall, F1, ROC-AUC.
- Visuals produced: confusion matrices, ROC curves, probability histograms, calibration
plots, and threshold-sweep CSVs.
Notes & caveats
- Severe class imbalance and very few positive examples in the holdout cause unstable
precision/recall. ROC-AUC can be high while classifier predicts no positives at threshold
0.5.
- Qualifiers input often lacks many training features; the pipeline retrains a
RandomForest on shared features for prediction when needed.
- Team-name mismatches can reduce merge coverage; canonical alias mapping is recommended
to improve feature availability.
Next recommended steps
1. Calibrate probabilities (Platt / isotonic) and pick an operating threshold favoring
recall if false negatives are costly.
2. Implement time-aware validation (train on earlier years, test on later) for realistic
forecasting.
3. Add Elo/ranking time-series, head-to-head aggregates, and canonical team-name mapping
to increase signal.
Files produced
- `fifa_2026/wc2026-finalists/reports/model_eval_full/` metrics, figures, models, PDF
report.
- `fifa_2026/wc2026-finalists/data/cleaned/qualifiers_predictions_2026.csv` predicted
finalist probabilities for 2026 qualifiers
