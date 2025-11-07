Model Evaluation Report — FIFA 2026 Finalists Prediction

Executive Summary
-----------------
Task: Predict which teams historically reached World Cup finals (2002–2022) and evaluate three models: Logistic Regression (LR), Random Forest (RF), and an Ensemble (average probabilities). Input predictions file: `outputs/finalists_top_all.csv` (75 teams). Ground truth: 9 historical finalists (Argentina, Brazil, Croatia, France, Germany, Italy, Netherlands, Spain, West Germany).

Main findings (threshold = 0.5):
- All models achieve perfect recall (identify all 9 historical finalists).
- Precision at 0.5: LR = 0.75, RF = 0.6429, Ensemble = 0.60.
- ROC-AUC: LR = 0.993, RF = 0.978, Ensemble = 0.987 — models rank finalists well.
- Trade-off: 0.5 threshold maximizes sensitivity but produces several false positives (LR: 3 FP; RF: 5 FP; Ensemble: 6 FP).

Recommendation: Use ensemble probabilities for ranking. Choose threshold depending on business needs:
- If recall is paramount (don’t miss finalists): use threshold = 0.5.
- If you want a balanced F1 (reduce false positives): use optimized threshold ≈ 0.6411 for the ensemble (see tuning section).

Methodology
-----------
- Ground truth: teams labeled 1 if historically reached a Final (2002–2022), else 0.
- Metrics computed: confusion matrix (TP, FP, FN, TN), Accuracy, Precision, Recall (TPR), F1-score, ROC-AUC, Balanced Accuracy, Specificity (TNR), Cohen's Kappa.
- Evaluation used the model output probabilities from `outputs/finalists_top_all.csv`.
- Threshold tuning: ensemble probabilities were swept (0.0–1.0) and threshold maximizing F1 was selected.

Results — Threshold = 0.5
-------------------------
Metrics (threshold 0.5):

Logistic Regression
- TP=9, FP=3, FN=0, TN=63
- Accuracy=0.96, Precision=0.75, Recall=1.00, F1=0.857, ROC-AUC=0.993, Balanced Accuracy=0.977, Cohen's Kappa≈0.834, Specificity=0.9545

Random Forest
- TP=9, FP=5, FN=0, TN=61
- Accuracy=0.9333, Precision=0.6429, Recall=1.00, F1=0.7826, ROC-AUC=0.978, Balanced Accuracy=0.9621, Cohen's Kappa≈0.745, Specificity=0.9242

Ensemble (LR + RF)
- TP=9, FP=6, FN=0, TN=60
- Accuracy=0.92, Precision=0.60, Recall=1.00, F1=0.75, ROC-AUC=0.9865, Balanced Accuracy=0.9545, Cohen's Kappa≈0.706, Specificity=0.9091

Visual artifacts (saved under `outputs/finalists/`):
- `eval_confusion_matrices.png` — confusion matrices for each model (threshold=0.5)
- `eval_roc_curves.png` — ROC curves with AUC annotations
- `eval_metrics_bar.png` — grouped bar chart (Accuracy, Precision, Recall, F1, ROC-AUC, Balanced Acc)
- `eval_prob_dist.png` — histogram of ensemble probabilities (finalists vs non-finalists)
- `eval_metrics_table.csv` — saved metrics table for the three models

Interpretation
--------------
- High ROC-AUC indicates good ranking ability; the models place true finalists near top probability scores. This justifies using probabilities for ranking (not just hard labels).
- Perfect recall at 0.5 means no historical finalists are missed, but precision varies — LR produces the fewest false positives at this threshold.
- Because the dataset is imbalanced (9 vs 66), accuracy alone is misleading; prefer ROC-AUC, Balanced Accuracy, Precision/Recall, and F1.

Threshold tuning (Ensemble)
--------------------------
- We computed precision, recall and F1 across thresholds and selected the threshold that maximizes F1 for the ensemble.
- Optimized threshold (max F1): ≈ 0.6411.

Rerun metrics at optimized threshold (≈0.6411)
---------------------------------------------
Files saved with `_thr` suffix under `outputs/finalists/`:
- `eval_metrics_table_thr.csv` — metrics at optimized threshold
- `eval_confusion_matrices_thr.png` — confusion matrices at optimized threshold
- `eval_metrics_bar_thr.png` — metrics bar chart at optimized threshold

Key observations after tuning:
- The optimized threshold reduces false positives compared with threshold=0.5, increasing precision and improving F1 in a balanced trade-off.
- Example confusion matrices (at thr ≈0.6411):
  - Logistic (thr=0.641): TP=7, FP=1, FN=2, TN=65
  - RandomForest (thr=0.641): TP=9, FP=4, FN=0, TN=62
  - Ensemble (thr=0.641): TP=9, FP=3, FN=0, TN=63
  (See `eval_confusion_matrices_thr.png` for exact counts and per-model layout.)

Recommendation after tuning
---------------------------
- Use the ensemble probabilities for ranking teams and apply threshold ≈0.64 if you want a balanced F1 and fewer false positives while keeping high recall.
- If missing a finalist is unacceptable, stick to 0.5 or a lower threshold to bias toward recall (accept more false positives).
- For deployment, present probability tiers rather than hard labels: >0.80 = High confidence finalist; 0.60–0.80 = Likely; 0.40–0.60 = Possible; <0.40 = Unlikely.

Limitations
-----------
- Evaluation uses historical finalists (2002–2022). Models trained on historical features may not capture tactical or structural changes by 2026.
- Class imbalance (9 positives) limits some statistical certainty — small changes can flip metrics.
- If any engineered features leak post-target information, ROC-AUC and high recall/precision may be over-optimistic. The pipeline removed direct `finals/semis/quarters` indicators; however, verify that no other target-derived proxies remain.

Next steps (suggested)
----------------------
1. Produce a 1–2 page PDF by combining the saved figures and this report (I can generate that from the notebook if you want). 
2. Run a temporal holdout experiment (train on ≤2018, test on 2022) to better estimate forecasting performance. 
3. Add SHAP explanations for the ensemble components to explain why specific non-finalists received high probabilities.

Artifacts & where to find them
------------------------------
- Notebook (evaluation): `notebooks/finalists_evaluation.ipynb` (contains code and inserted report cells)
- Report (this file): `notebooks/finalists_evaluation_report.md`
- Generated figures and CSVs: `outputs/finalists/` (files listed above)

Prepared by: evaluation runner
Date: 2025-11-07
