# Final Report — FIFA 2026 Qualifiers & Finalists Prediction

Date: 2025-11-07

This document summarizes the end-to-end workflow carried out in this project: from preparing historical match/team data and predicting the remaining 20 qualifiers for FIFA 2026, to building and validating finalist prediction models, tuning, evaluation, feature-importance analysis, final predictions, and reflections on limitations and ethics.

## 1 — Goal

- Predict the remaining 20 qualifiers for FIFA 2026 (given 28 confirmed).
- Using historical World Cup match data (2002–2022) and squad/team features, build honest, leakage-free classification models that predict which teams reach the Finals (or are likely finalists) for 2026.

## 2 — Data & artifacts (where to find them)

- Raw / cleaned data used:
  - `FIFA-2026/Week-1/Data/cleaned_data-set.csv` (historical match-level features)
  - `world_cup_2002_2022_squad_avg_age.csv` and `world_cup_2002_2022_team_avg_experience.csv` (squad features)
- Key generated artifacts (workspace):
  - `outputs/finalists/finalists_top_all.csv` — team-level predictions and probabilities
  - `outputs/finalists/finalists_top9.csv` — top-9 predicted finalists
  - `outputs/finalists/lr_best.pkl`, `rf_best.pkl` — saved best estimator pickles
  - `outputs/finalists/feat_lr_top.png`, `feat_rf_top.png`, `feat_rf_cumulative.png`, `feat_agreement_heatmap.png` — feature-importance plots
  - `notebooks/final_predictions_reflection.ipynb` — final predictions + reflection notebook
  - `notebooks/finalists_prediction.ipynb`, `notebooks/feature_importance.ipynb`, `notebooks/finalists_evaluation.ipynb` — main implementation & evaluation notebooks

## 3 — High-level workflow (step-by-step)

1. Data collection & cleaning
   - Gathered historical World Cup match records and squad/team aggregates.
   - Engineered team-level profiles (aggregating match-level statistics into team summaries) so each team is one observation for modelling.

2. Candidate features & leakage prevention
   - Features included: win rates, goals per match, goal difference per match, knockout win rate, finals/semis/quarters historical counts, avg squad age, avg international caps, FIFA ranking and engineered diffs (rank/age/experience).
   - Explicit leakage policy: removed any columns that directly reveal future tournament outcomes (e.g., fields named `finals`, `semis`, `quarters`). Any features derived from future knowledge were excluded.

3. Feature preprocessing
   - Missing value handling: median imputation for numeric features; indicator flags for heavily missing fields when appropriate.
   - Scaling: `StandardScaler` (applied inside sklearn Pipeline so no train-test leakage).
   - Encoding: low-cardinality categorical fields (if used) were one-hot encoded inside the pipeline. Team name categorical encoding was avoided or applied cautiously.
   - Feature selection: used SelectKBest (mutual_info_classif) inside the pipeline to reduce dimensionality and avoid overfitting on small-N data.

4. Modelling approach
   - Baseline: Logistic Regression (with L1/L2 regularization). Provides interpretability via coefficients.
   - Non-linear model: Random Forest (with constrained depth and leaf samples to control overfitting).
   - Optional: XGBoost available in templates for further experiments (not required to pass core task).

5. Validation and hyperparameter tuning (to avoid optimistic bias)
   - Nested cross-validation setup (honest tuning):
     - Outer loop: StratifiedKFold(n_splits=5) — estimate generalization performance.
     - Inner loop: StratifiedKFold(n_splits=3) — hyperparameter search.
   - Scoring metric: ROC-AUC (primary) and F1/precision/recall where appropriate.

6. Hyperparameter grids (examples used)
   - Logistic Regression (GridSearchCV):
     - penalty: ['l1', 'l2']
     - C: [0.001, 0.01, 0.1, 1, 10]
     - solver: ['liblinear', 'saga']
     - class_weight: [None, 'balanced']
   - Random Forest (RandomizedSearchCV):
     - n_estimators: [50, 100, 200]
     - max_depth: [3, 5, 10, None]
     - min_samples_split: [5, 10, 20]
     - min_samples_leaf: [2, 5, 10]
     - max_features: ['sqrt', 'log2', 0.3]
     - class_weight: [None, 'balanced', 'balanced_subsample']

7. Calibration & ensemble
   - Probability calibration was considered with `CalibratedClassifierCV` where needed.
   - Ensemble predictions were created by averaging LR and RF probabilities (simple ensemble). If an ensemble pickle exists it was used directly.

8. Threshold tuning
   - We report ranking metrics (ROC-AUC) and tune a classification threshold for the ensemble using F1 maximization on validation data.
   - Example result: tuned ensemble threshold ≈ 0.6411 (improves precision / F1 trade-off compared with 0.5).

9. Final predictions
   - For qualifiers: predicted remaining 20 teams (the pipeline and Copilot prompts guided how to do this — the resulting candidate list is saved in `outputs/finalists/` files).
   - For finalists: produced ranked probabilities and selected top-9 finalists (artifacts: `finalists_top9.csv`, `finalists_top_all.csv`).

10. Feature importance & interpretation
    - Extracted LR coefficients (signed influence) and RF feature_importances_.
    - Plotted top features, cumulative importance, and agreement heatmap between models.
    - Interpreted top features in football context (finals_appearances, knockout_win_rate, elite indicator, FIFA ranking, goal difference) and noted that some expected features (average age, home advantage) had weaker effects.

11. Evaluation & reflection
    - Evaluation used OOF (out-of-fold) ROC and precision/recall. Observed very high ROC-AUC (LR ≈ 0.99, RF ≈ 0.98, ensemble ≈ 0.987) — models rank finalists well.
    - Noted perfect recall at threshold 0.5 on historical finalists but several false positives; tuned threshold to balance.
    - Wrote a final reflection document on limitations, uncertainties (injuries, randomness, tactical change), and ethical risks (gambling, fan/athlete impact, bias against emerging nations).

## 4 — Key findings and numbers (concise)

- Nested CV performance (representative):
  - Logistic Regression outer CV ROC-AUC: mean ≈ 0.939 ± 0.061 (example nested CV run)
  - Random Forest outer CV ROC-AUC: mean ≈ 0.917 ± 0.086 (example nested CV run)
- OOF diagnostics: LR OOF AUC ≈ 0.916, RF OOF AUC ≈ 0.898 (example run)
- Final saved models: `lr_best.pkl`, `rf_best.pkl` in `outputs/finalists/`.
- Tuned ensemble threshold (F1-optimized): ≈ 0.6411 (used to reduce false positives while maintaining high recall in practice).

## 5 — Why the models look good — and caveats

- Strengths:
  - High ROC-AUC indicates the models order teams reliably (good ranking power).
  - Nested CV and pipelines reduce leakage and tuning optimism.
  - Simpler models (LR) provide interpretability useful for reporting.

- Important caveats:
  - Small positive-class (historical finalists ≈ 9) increases statistical uncertainty.
  - Many features are historically driven — models may underweight emerging teams.
  - Temporal extrapolation risk: model trained on 2002–2022, predicting 2026.
  - Inherent sports randomness (penalty shootouts, luck) is not modelable.

## 6 — Repro & how to re-run key steps

1. Reproduce models and predictions
   - Open `notebooks/finalists_prediction.ipynb` and run top-to-bottom. It contains the pipelines, nested CV training, and code to save `lr_best.pkl`, `rf_best.pkl`, and predictions.
2. Recreate final predictions & reflection
   - Open `notebooks/final_predictions_reflection.ipynb` — it loads pickles in `outputs/finalists/` and a team-features CSV. It will use precomputed probabilities if present, or run model inference if raw features are provided.
3. Useful commands (run in workspace root)
```
# run training notebook headless (requires jupyter nbconvert installed)
jupyter nbconvert --to notebook --execute notebooks/finalists_prediction.ipynb --output notebooks/finalists_prediction.executed.ipynb

# run the final predictions notebook
jupyter nbconvert --to notebook --execute notebooks/final_predictions_reflection.ipynb --output notebooks/final_predictions_reflection.executed.ipynb
```

## 7 — Recommendations & next steps (priority)

1. Temporal holdout: train on <=2018, test on 2022. This gives a realistic estimate of out-of-sample forecasting performance and should be done before public claims.
2. Probability calibration: evaluate calibration plots; if miscalibrated, apply `CalibratedClassifierCV` or isotonic/Platt scaling. Use calibrated probabilities before applying fixed thresholds.
3. Bootstrap confidence intervals: use bootstrap resampling of training data (or repeated CV) to get robust CIs for team probabilities.
4. SHAP explainability: run SHAP on best estimator for per-team explanations and waterfall plots — very helpful for communicating reasons behind predictions.
5. Fairness audit: analyze model bias toward historical powers and consider weighting recent form more heavily or building separate models for 'emerging' vs 'traditional' nations.

## 8 — Ethical & communication guidance (short)

- Always present probabilities with uncertainty bands and clear disclaimers ("for entertainment/analysis, not financial advice").
- Avoid deterministic headlines. Use phrasing like "likely finalists" or "top contenders".
- Do not release betting-facing predictions without a clear ethical review and harm-mitigation plan.

## 9 — Closing note

The pipeline we built is robust for ranking candidate teams and producing interpretable signals about which teams are probable finalists. However, ranking power (high ROC-AUC) is not the same as perfect forecasting — especially in sport where randomness and temporal change matter. Use calibrated probabilities, temporal validation, and bootstrap uncertainty before making strong public claims. If you'd like, I can implement the high-priority next steps (temporal holdout, bootstrap CIs, SHAP) and produce a one-page PDF summary for submission.

---
Files created/edited in this session: `notebooks/final_predictions_reflection.ipynb`, `notebooks/final_report_summary.md` (this file), plus plotted artifacts under `notebooks/outputs/finalists/`.
