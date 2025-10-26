import pandas as pd
from pathlib import Path
from typing import List

# ---------- Paths (resolve relative to this script so script works from any CWD) ----------
BASE_DATA_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DATA_DIR / "raw"
OUT = Path(__file__).resolve().parent / "matches_clean.csv"


def find_raw_csvs(raw_dir: Path) -> List[Path]:
    """Return a list of CSV paths under raw_dir."""
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")
    return sorted(raw_dir.glob("*.csv"))


def load_and_concat(paths: List[Path]) -> pd.DataFrame:
    """Load CSVs and concatenate them into one DataFrame.

    Uses low_memory=False to avoid dtype inference issues and
    preserves original columns where possible.
    """
    dfs = []
    for p in paths:
        print(f"Loading: {p}")
        df = pd.read_csv(p, low_memory=False)
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # normalize column names (strip and keep original case for human-readability)
    df.columns = df.columns.str.strip()

    # expected columns and ensure they exist
    expected_cols = [
        'Year', 'home_team', 'away_team', 'home_score', 'home_xg',
        'home_penalty', 'away_score', 'away_xg', 'away_penalty', 'Round'
    ]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = pd.NA

    # trim whitespace in string columns
    for col in ['home_team', 'away_team', 'Round']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace({'nan': pd.NA})

    # coerce numeric columns safely
    num_cols = ['Year', 'home_score', 'away_score', 'home_xg', 'away_xg', 'home_penalty', 'away_penalty']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # drop rows without teams or without scores
    df = df.dropna(subset=['home_team', 'away_team', 'home_score', 'away_score'])

    # cast Year and scores to integers where possible
    df['Year'] = df['Year'].astype(int)
    df['home_score'] = df['home_score'].astype(int)
    df['away_score'] = df['away_score'].astype(int)

    # xG columns: keep as float (NaN allowed)
    for col in ['home_xg', 'away_xg']:
        if col in df.columns:
            df[col] = df[col].astype(float)

    # penalties: may be NaN; keep as Int where possible (use pandas nullable Int64)
    for col in ['home_penalty', 'away_penalty']:
        if col in df.columns:
            df[col] = df[col].astype('Int64')

    # remove exact duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # basic cleanup for Round values
    if 'Round' in df.columns:
        df['Round'] = df['Round'].astype(str).str.strip().replace({'nan': pd.NA})

    return df


def merge_top4_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Merge top-4 team stats (avg exp, win rate, etc.) from the repository CSV into matches.

    The source file is expected at the repository root named
    `world_cup_complete_analysis_all_positions.csv`. For each match row we add
    prefixed columns for the home and away teams (e.g. `home_Avg_Experience_Caps`).
    """
    # try to locate the repository root by walking upwards and looking for
    # the expected extra-stats files. This is more robust across slightly
    # different repository layouts than relying on a fixed `parents[4]`.
    repo_root = Path(__file__).resolve().parent
    for _ in range(6):
        if (repo_root / 'world_cup_all_teams_complete_analysis.csv').exists() or (repo_root / 'world_cup_complete_analysis_all_positions.csv').exists():
            break
        if repo_root.parent == repo_root:
            break
        repo_root = repo_root.parent
    # if neither file was found in the upward walk, fall back to the original heuristic
    if not (repo_root / 'world_cup_all_teams_complete_analysis.csv').exists() and not (repo_root / 'world_cup_complete_analysis_all_positions.csv').exists():
        repo_root = Path(__file__).resolve().parents[4]

    # Prefer the full all-teams stats file if present; otherwise fall back to the
    # top-4 per-year file used earlier.
    all_teams_path = repo_root / 'world_cup_all_teams_complete_analysis.csv'
    top4_path = repo_root / 'world_cup_complete_analysis_all_positions.csv'

    if all_teams_path.exists():
        extra_path = all_teams_path
        print(f"Using all-teams stats file: {extra_path}")
    elif top4_path.exists():
        extra_path = top4_path
        print(f"Using top-4 stats file: {extra_path}")
    else:
        print(f"No extra stats file found at {all_teams_path} or {top4_path} — skipping merge")
        return df

    extra = pd.read_csv(extra_path)
    extra.columns = extra.columns.str.strip()

    # Basic team-name trimming to improve join coverage (strip extra whitespace)
    if 'Team' in extra.columns:
        extra['Team'] = extra['Team'].astype(str).str.strip()
    if 'home_team' in df.columns:
        df['home_team'] = df['home_team'].astype(str).str.strip()
    if 'away_team' in df.columns:
        df['away_team'] = df['away_team'].astype(str).str.strip()

    # Ensure Year and Team columns exist
    if 'Year' not in extra.columns or 'Team' not in extra.columns:
        print(f"Extra stats file {extra_path} missing required columns 'Year' and 'Team' — skipping merge")
        return df

    # Build home-prefixed extra and merge
    home_extra = extra.copy()
    # rename all columns except Year and Team to have home_ prefix
    rename_home = {c: f'home_{c}' for c in home_extra.columns if c not in ['Year', 'Team']}
    home_extra = home_extra.rename(columns=rename_home)
    # keep Year and Team (Team will be used as join key then dropped)
    home_cols = ['Year', 'Team'] + [c for c in home_extra.columns if c.startswith('home_')]
    home_extra = home_extra.loc[:, home_cols]

    df = df.merge(home_extra, left_on=['Year', 'home_team'], right_on=['Year', 'Team'], how='left')
    if 'Team' in df.columns:
        df.drop(columns=['Team'], inplace=True)

    # Build away-prefixed extra and merge
    away_extra = extra.copy()
    rename_away = {c: f'away_{c}' for c in away_extra.columns if c not in ['Year', 'Team']}
    away_extra = away_extra.rename(columns=rename_away)
    away_cols = ['Year', 'Team'] + [c for c in away_extra.columns if c.startswith('away_')]
    away_extra = away_extra.loc[:, away_cols]

    df = df.merge(away_extra, left_on=['Year', 'away_team'], right_on=['Year', 'Team'], how='left')
    if 'Team' in df.columns:
        df.drop(columns=['Team'], inplace=True)

    return df


def main():
    csvs = find_raw_csvs(RAW_DIR)
    if not csvs:
        print(f"No CSV files found in {RAW_DIR}")
        return

    df = load_and_concat(csvs)
    print("Initial shape:", df.shape)
    print("Columns:", df.columns.tolist())

    df_clean = clean(df)

    # merge extra top-4 team stats (if available)
    df_clean = merge_top4_stats(df_clean)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(OUT, index=False)
    print(f"✅ Cleaned CSV saved → {OUT} ({len(df_clean)} rows)")


if __name__ == '__main__':
    main()
