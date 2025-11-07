"""Merge team-level scraped features into matches dataset.

Reads a matches CSV and the following team-level CSVs if present in workspace root:
 - world_cup_2002_2022_squad_avg_age.csv
 - world_cup_2002_2022_team_avg_experience.csv
 - fifa_rankings_wc_years_2002_2022_clean.csv

Produces an output file next to the input called <matches_input_basename>_with_team_features.csv

Usage:
    python merge_match_team_features.py --matches matches_clean_Data.csv

The script normalizes simple whitespace and parentheses in team names and attempts case-insensitive
matching. It reports unmatched teams and the number of enriched matches.
"""

from __future__ import annotations
import argparse
import pandas as pd
import re
from pathlib import Path
import difflib


def normalize_name(s: str) -> str:
    if pd.isna(s):
        return ''
    s = str(s)
    s = s.replace('\n', ' ')
    s = re.sub(r"\s+", ' ', s)
    s = re.sub(r"\s*\(.*?\)", '', s)  # remove parenthetical notes
    s = s.strip()
    return s


# small built-in alias map to improve joins between sources. If you need more
# mappings, create a CSV file 'team_aliases.csv' with columns 'alias,canonical'
# in the workspace root and it will be loaded instead (alias in left col).
DEFAULT_ALIAS_MAP = {
    'Korea Republic': 'South Korea',
    'IR Iran': 'Iran',
    'United States': 'USA',
    'USA': 'United States',
    "Côte d'Ivoire": "Cote d'Ivoire",
    'Korea DPR': 'North Korea',
    'Republic of Ireland': 'Ireland',
    'Türkiye': 'Turkey',
    'Bosnia and Herzegovina': 'Bosnia & Herzegovina',
    'Czech Republic': 'Czechia',
}


def load_alias_map(root: Path) -> dict:
    csv_fp = root / 'team_aliases.csv'
    if csv_fp.exists():
        try:
            df = pd.read_csv(csv_fp)
            m = {}
            # expect columns alias,canonical (case-insensitive)
            cols = {c.lower(): c for c in df.columns}
            acol = cols.get('alias') or cols.get('from') or list(df.columns)[0]
            ccol = cols.get('canonical') or cols.get('to') or (list(df.columns)[1] if len(df.columns) > 1 else acol)
            for _, r in df.iterrows():
                if pd.isna(r[acol]):
                    continue
                m[str(r[acol]).strip()] = str(r[ccol]).strip()
            return m
        except Exception:
            return DEFAULT_ALIAS_MAP
    return DEFAULT_ALIAS_MAP


def load_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def main(matches_fp: Path):
    matches = pd.read_csv(matches_fp)
    print('Loaded matches:', len(matches))

    # detect common team-level files in workspace root
    root = Path('.').resolve()
    age_fp = root / 'world_cup_2002_2022_squad_avg_age.csv'
    exp_fp = root / 'world_cup_2002_2022_team_avg_experience.csv'
    rank_fp = root / 'fifa_rankings_wc_years_2002_2022_clean.csv'

    df_age = load_csv_if_exists(age_fp)
    df_exp = load_csv_if_exists(exp_fp)
    df_rank = load_csv_if_exists(rank_fp)

    print('Found age rows:', len(df_age), 'exp rows:', len(df_exp), 'rank rows:', len(df_rank))

    # normalize team column names in team tables
    for df in (df_age, df_exp, df_rank):
        if df.empty:
            continue
        if 'Team' not in df.columns and 'team' in df.columns:
            df.rename(columns={c: c.title() for c in df.columns}, inplace=True)

    # load alias map (can be overridden by a team_aliases.csv file)
    alias_map = load_alias_map(root)

    def canonicalize(team: str) -> str:
        if pd.isna(team):
            return ''
        n = normalize_name(team)
        return alias_map.get(n, n)

    # prepare lookup dicts keyed by (Year, Team)
    def make_lookup(df: pd.DataFrame, value_col: str, team_col: str = 'Team'):
        if df.empty:
            return {}
        l = {}
        # ensure Year present
        if 'Year' not in df.columns:
            # if only one Year present, broadcast
            years = df['Year'] if 'Year' in df.columns else [None]
        for _, r in df.iterrows():
            yr = int(r['Year']) if 'Year' in r and not pd.isna(r['Year']) else None
            t = canonicalize(r[team_col])
            key = (yr, t)
            l[key] = r.get(value_col)
        return l

    age_lookup = make_lookup(df_age, 'Avg_Squad_Age', team_col='Team')
    exp_lookup = make_lookup(df_exp, 'Avg_Experience_Caps', team_col='Team')
    # ranking: we want Points and Rank
    # keep only ranking (we no longer use/emit FIFA points)
    rank_lookup_rank = {}
    if not df_rank.empty:
        for _, r in df_rank.iterrows():
            yr = int(r['Year']) if 'Year' in r and not pd.isna(r['Year']) else None
            t = canonicalize(r['Team'])
            key = (yr, t)
            rank_lookup_rank[key] = r.get('Rank')

    # augment matches with both home/away features
    out = matches.copy()

    # Normalize team name columns used in matches: try common names
    team_cols = [c for c in out.columns if c.lower() in ('home_team', 'away_team', 'team1', 'team2', 'home', 'away')]
    if not team_cols:
        # fallback: check for 'HomeTeam'/'AwayTeam'
        for name in ('HomeTeam','AwayTeam','home_team','away_team'):
            if name in out.columns:
                team_cols = [name]
                break

    # prefer detecting separate home/away columns
    home_col = None
    away_col = None
    for c in out.columns:
        if c.lower() in ('home_team','hometeam','home'):
            home_col = c
        if c.lower() in ('away_team','awayteam','away'):
            away_col = c
    # fallback to common names
    if not home_col or not away_col:
        # try 'Team1'/'Team2'
        for c in out.columns:
            if c.lower() in ('team1','team_1') and not home_col:
                home_col = c
            if c.lower() in ('team2','team_2') and not away_col:
                away_col = c

    if not home_col or not away_col:
        raise SystemExit('Could not detect home/away team columns in matches file. Columns: ' + ','.join(out.columns))

    # add normalized name columns
    out['_home_team_norm'] = out[home_col].apply(canonicalize)
    out['_away_team_norm'] = out[away_col].apply(canonicalize)

    # helper to lookup with fallbacks (exact year, then any-year)
    def lookup_with_fallback(lookup_dict, year, team):
        # exact (use canonical form of input team)
        team_can = canonicalize(team)
        key = (int(year) if pd.notna(year) else None, team_can)
        if key in lookup_dict:
            return lookup_dict[key]
        # try any-year matches
        for (y, t), v in lookup_dict.items():
            if t == team_can:
                return v
        # try case-insensitive
        for (y, t), v in lookup_dict.items():
            if t.casefold() == team_can.casefold():
                return v
        # fuzzy
        choices = [t for (_, t) in lookup_dict.keys()]
        match = difflib.get_close_matches(team_can, choices, n=1, cutoff=0.85)
        if match:
            # find first matching key
            for (y, t), v in lookup_dict.items():
                if t == match[0]:
                    return v
        return None

    # perform merges
    home_age = []
    away_age = []
    home_exp = []
    away_exp = []
    home_rank = []
    away_rank = []

    for _, r in out.iterrows():
        yr = r.get('Year') if 'Year' in r else None
        ht = r['_home_team_norm']
        at = r['_away_team_norm']
        home_age.append(lookup_with_fallback(age_lookup, yr, ht))
        away_age.append(lookup_with_fallback(age_lookup, yr, at))
        home_exp.append(lookup_with_fallback(exp_lookup, yr, ht))
        away_exp.append(lookup_with_fallback(exp_lookup, yr, at))
        home_rank.append(lookup_with_fallback(rank_lookup_rank, yr, ht))
        away_rank.append(lookup_with_fallback(rank_lookup_rank, yr, at))

    out['Home_Avg_Squad_Age'] = home_age
    out['Away_Avg_Squad_Age'] = away_age
    out['Home_Avg_Experience_Caps'] = home_exp
    out['Away_Avg_Experience_Caps'] = away_exp
    out['Home_FIFA_Rank'] = home_rank
    out['Away_FIFA_Rank'] = away_rank

    # compute rank difference (away - home) so positive means away ranked worse (higher number)
    out['FIFA_Rank_Diff'] = pd.to_numeric(out['Away_FIFA_Rank'], errors='coerce') - pd.to_numeric(out['Home_FIFA_Rank'], errors='coerce')
    out['Age_Diff'] = pd.to_numeric(out['Home_Avg_Squad_Age'], errors='coerce') - pd.to_numeric(out['Away_Avg_Squad_Age'], errors='coerce')
    out['Exp_Diff'] = pd.to_numeric(out['Home_Avg_Experience_Caps'], errors='coerce') - pd.to_numeric(out['Away_Avg_Experience_Caps'], errors='coerce')

    # report unmatched teams
    unmatched = set()
    for col in ['Home_Avg_Squad_Age','Away_Avg_Squad_Age','Home_Avg_Experience_Caps','Home_FIFA_Rank']:
        for i, v in out[col].items():
            if pd.isna(v):
                unmatched.add(out.loc[i, '_home_team_norm'])
                unmatched.add(out.loc[i, '_away_team_norm'])
    # remove empty strings
    unmatched = {u for u in unmatched if u}

    out_fp = matches_fp.with_name(matches_fp.stem + '_with_team_features.csv')
    out.to_csv(out_fp, index=False)
    print('Wrote merged file:', out_fp, 'rows:', len(out))
    print('Unique unmatched sample (up to 30):', list(unmatched)[:30])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge team-level features into matches')
    parser.add_argument('--matches', type=str, required=True, help='Path to matches CSV (cleaned)')
    args = parser.parse_args()
    main(Path(args.matches))
