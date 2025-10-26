World Cup Data Project — Data collection, cleaning, feature engineering and scraper documentation

Executive summary

This document summarizes the end-to-end data work performed for the World Cup matches project covering the years 2002–2022. The primary match dataset was obtained from Kaggle ("jahaidulislam/fifa-world-cup-1930-2022-all-match-dataset"). The raw match data were trimmed to the target years, then cleaned and normalized. Additional per-team statistics were merged back into the match-level data to add team-level features such as average player experience (caps), win rate, goal difference and squad average age.

This processed dataset (saved as `fifa_2026/wc2026-finalists/data/cleaned/matches_clean.csv`) contains 552 match rows and includes prefixed team-stat columns (home_*/away_*) that are populated when the matching per-team statistics are available for the same tournament year.

1) Data collection

Source and method
- Dataset: Kaggle — jahaidulislam/fifa-world-cup-1930-2022-all-match-dataset
- Acquisition approach used during the project (example):
  path = kagglehub.dataset_download("jahaidulislam/fifa-world-cup-1930-2022-all-match-dataset")

Process
- Download the Kaggle dataset and place the relevant CSV(s) under `fifa_2026/wc2026-finalists/data/raw/`.
- The working file used in this project is `matches_2002-2022.csv` (raw matches filtered to the 2002–2022 tournaments).

Notes on provenance and reproducibility
- For full reproducibility prefer the Kaggle CLI or API (see scraper README). Store any downloaded files or checksums in a raw-data directory and track their provenance.

2) Data trimming and cleaning

What was trimmed
- The original dataset spans 1930–2022. The project focuses on the modern tournaments 2002–2022. Rows outside this range were removed before cleaning.

Cleaning steps performed (implemented in `fifa_2026/wc2026-finalists/data/cleaned/clean_wc_data.py`)
- Path handling: Paths resolve relative to the script so the pipeline is robust to CWD.
- Load: All CSV files in `data/raw/` are concatenated safely (`pd.read_csv(..., low_memory=False)`).
- Column normalization: column names trimmed of whitespace and checked for expected columns. Missing expected columns are created with NA so the pipeline stays consistent across versions of the raw file.
- Type coercion: numeric columns coerced with `pd.to_numeric(errors='coerce')`. Year and score columns are cast to integers after removing rows with missing team names or scores. xG columns are floats and penalty columns use pandas nullable integer dtype (`Int64`) to allow NaN values.
- Rows dropped: Entire rows with missing home/away team names or missing scores were dropped (these are incomplete matches).
- Duplicate removal: exact duplicates are removed.
- Round normalization: whitespace trimmed; values preserved for grouping (group stage, round of 16, quarter-finals, etc.).

Validation
- The cleaning script was executed and produced `matches_clean.csv` (552 rows). Post-clean checks confirm expected columns and types.

3) Feature engineering and top‑4 team augmentation

Objective
- Enrich each match row with per-team statistics gathered for the tournament top‑4 teams (Winner, Runner‑up, Third place, Fourth place). These features are intended to capture squad-level experience and performance indicators that may be predictive or explanatory in downstream analysis.

Per-team stats sources
- The script will prefer an "all teams" stats table when present: `world_cup_all_teams_complete_analysis.csv` (placed at the repository root). That file contains one row per team per tournament and provides broad coverage for merging team-level features onto matches.
- If the all-teams file is not present, the script falls back to the smaller top‑4 table `world_cup_complete_analysis_all_positions.csv` (also at repository root), which contains one row per top‑4 placement per tournament. When the top‑4 table is used, merged features will be populated only for those four teams per tournament.
- Historically, a user-supplied list of 24 `Avg_Squad_Age` values (4 per tournament) was used to augment the top‑4 table; if you prefer reproducibility, convert that list into a small CSV (e.g., `data/top4_avg_squad_age.csv`) and place it in the repository so the script can read it explicitly.

Merged features added to matches
- For each match, the script performs a left-join on (Year, Team) to bring per-team statistics into the match table and creates prefixed columns for both teams (e.g., `home_Avg_Experience_Caps`, `away_Avg_Experience_Caps`, `home_Win_Rate_Percent`, `away_Win_Rate_Percent`, `home_Avg_Squad_Age`, etc.).
- Coverage depends on which stats file is available. When `world_cup_all_teams_complete_analysis.csv` is present, the coverage is much higher because that file contains every team for each tournament. When falling back to the top‑4 file, only the four teams per tournament will have non-null values.

Design notes
- Merging by exact `(Year, Team)` keeps the logic simple and auditable. If team names have variant spellings (accents, abbreviations), the join may fail for those rows; a normalization step would increase merge coverage.

4) Scraper and reproducibility (how to re-run)

Minimal reproducible commands (Kaggle)
1) Install and configure Kaggle CLI:
   - `pip install kaggle`
   - Set environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY` (or place credentials in `~/.kaggle/kaggle.json`).
2) Download dataset and unzip to `data/raw/`:
   - `kaggle datasets download jahaidulislam/fifa-world-cup-1930-2022-all-match-dataset -p fifa_2026/wc2026-finalists/data/raw/ --unzip`
3) Trim and run cleaning script:
   - `python3 fifa_2026/wc2026-finalists/data/cleaned/clean_wc_data.py`

Notes
- The script expects `world_cup_complete_analysis_all_positions.csv` to be in the repository root for merging top‑4 stats. If you prefer, convert the supplied Avg_Squad_Age list into a small CSV (e.g., `data/top4_avg_squad_age.csv`) and update the script to read it instead of embedding ages.

5) Recommendations and next steps
- Normalize team names before merging (strip accents, map known variants) to increase merge coverage.
- Add boolean flags `home_is_top4` / `away_is_top4` for easy model input.
- Optionally remove granular breakdown columns (Matches, Wins, Draws, Losses, Goals_For, Goals_Against) if you only need Avg_Experience_Caps, Win_Rate_Percent, and Avg_Squad_Age.
- Add unit tests that verify the pipeline's outputs and non-null counts for merged fields.

Appendix — input/output contract
- Inputs: `data/raw/*.csv` (raw matches), and optionally one of:
   - `world_cup_all_teams_complete_analysis.csv` (preferred — per-team stats for all teams), or
   - `world_cup_complete_analysis_all_positions.csv` (fallback — top‑4 only)
- Output: `data/cleaned/matches_clean.csv` containing cleaned matches and merged per-team stats (home_/away_ prefixed).

Notes
- If team name mismatches prevent merges, normalize team names in both the raw matches and the per-team stats file (strip whitespace, unify accents/aliases). The cleaning script strips whitespace by default; consider adding a canonical mapping for known variants to increase merge coverage.

End of document

End of document
