Scraper and data acquisition README

Purpose
This README explains how to reproducibly download the World Cup match dataset from Kaggle and where to place supporting per-team stats so the project's cleaning script can run and enrich match rows.

Prerequisites
- Python 3.8+
- pip
- Kaggle account (username and API key)

Setup Kaggle CLI
1) Install the Kaggle CLI:
   pip install kaggle
2) Configure credentials:
   - Option A (recommended): place a JSON file at `~/.kaggle/kaggle.json` with:
     {
       "username": "YOUR_KAGGLE_USERNAME",
       "key": "YOUR_KAGGLE_KEY"
     }
   - Option B: set environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY`.

Download the dataset (example)
From the repository root, run:

  kaggle datasets download jahaidulislam/fifa-world-cup-1930-2022-all-match-dataset -p fifa_2026/wc2026-finalists/data/raw/ --unzip

This will place the dataset CSV(s) into `fifa_2026/wc2026-finalists/data/raw/`. Ensure the raw file used by the pipeline is named or moved to `matches_2002-2022.csv`.

Per-team stats files (optional but recommended)
- To enable merging of broad per-team statistics for every team, place `world_cup_all_teams_complete_analysis.csv` at the repository root. The cleaning pipeline will prefer this file and merge its columns (prefixed with `home_`/`away_`) into the match table when available.
- If the all-teams file is not available, the pipeline will fall back to `world_cup_complete_analysis_all_positions.csv` (top‑4 only), which will only populate merged columns for the four top-placed teams per tournament.

Run the cleaning pipeline
After raw files are in place, run:

  python3 fifa_2026/wc2026-finalists/data/cleaned/clean_wc_data.py

What the script does
- concatenates CSVs in `data/raw/` (if multiple),
- cleans types and missing values,
- attempts to locate and merge per-team stats (prefers `world_cup_all_teams_complete_analysis.csv`, falls back to the top‑4 file), and
- writes `matches_clean.csv` to `fifa_2026/wc2026-finalists/data/cleaned/`.

Notes and troubleshooting
- If the Kaggle download fails, verify `kaggle` is installed and credentials are correct.
- If team name mismatches prevent merges, consider normalizing team names in both the raw matches and the per-team stats file. The script trims whitespace by default; canonical mappings for known variants will further improve coverage.

Optional: small download script
Create `scraper/get_kaggle_dataset.sh` with the following contents and run it (after configuring credentials):

  #!/usr/bin/env bash
  set -euo pipefail
  mkdir -p fifa_2026/wc2026-finalists/data/raw
  kaggle datasets download jahaidulislam/fifa-world-cup-1930-2022-all-match-dataset -p fifa_2026/wc2026-finalists/data/raw/ --unzip

Make it executable: `chmod +x scraper/get_kaggle_dataset.sh`

End of README
