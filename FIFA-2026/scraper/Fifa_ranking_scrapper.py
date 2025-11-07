"""
All-in-one FIFA rankings extractor

This script bundles the heuristics used in the workspace into a single file:
- static FIFA page table extractor (requests + BeautifulSoup + pandas.read_html fallback)
- per-year Wikipedia page extractor (multiple URL patterns + API search fallback)
- basic cleaning and World-Cup-year filtering with optional fuzzy team matching

Usage examples (from the command line):
    python scrapper_FIFAranking_allinone.py --aggregate --out fifa_rankings_2002_2022_wikipedia.csv
    python scrapper_FIFAranking_allinone.py --clean-wc --in fifa_rankings_2002_2022_wikipedia.csv \
        --teams world_cup_all_teams_complete_analysis.csv --out fifa_rankings_wc_years_2002_2022_clean.csv

The script is written to be safe to import; no scraping runs on import.
"""

from __future__ import annotations
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
from typing import Dict, List, Optional, Tuple
import difflib


def scrape_fifa_rankings(year: int, url: str, timeout: int = 15) -> pd.DataFrame:
    """Scrape a FIFA ranking page for a static Team/Points table.

    Returns a DataFrame with columns ['Year','Rank','Team','Points'] or raises ValueError.
    This mirrors the safe static approach used previously.
    """
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
            '(KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36'
        )
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, 'html.parser')

    # Heuristics: find tables whose header contains 'team' and 'point(s)'
    candidate_tables = []
    for tbl in soup.find_all('table'):
        ths = tbl.find_all('th')
        header_text = ' | '.join([th.get_text(' ', strip=True).lower() for th in ths])
        if 'team' in header_text and any(p in header_text for p in ('point', 'pts', 'points')):
            candidate_tables.append(tbl)

    df = None
    # Try parsing candidate tables found with BeautifulSoup
    for tbl in candidate_tables:
        rows = tbl.find_all('tr')
        if not rows:
            continue
        header_cells = rows[0].find_all(['th', 'td'])
        cols = [hc.get_text(' ', strip=True) for hc in header_cells]
        data = []
        for row in rows[1:]:
            cells = row.find_all(['td', 'th'])
            if not cells:
                continue
            vals = [c.get_text(' ', strip=True) for c in cells]
            # pad
            if len(vals) < len(cols):
                vals += [''] * (len(cols) - len(vals))
            data.append(dict(zip(cols, vals)))
        if data:
            df = pd.DataFrame(data)
            break

    # Fallback to pandas.read_html
    if df is None:
        try:
            tables = pd.read_html(resp.text)
            for t in tables:
                cols = [str(c).lower() for c in t.columns]
                if any('team' in c for c in cols) and any(p in c for p in ('point', 'pts', 'points')):
                    df = t.copy()
                    break
        except Exception:
            df = None

    if df is None:
        raise ValueError(f'No static ranking table found on page {url} (page may be JS-driven)')

    # Normalize column names
    colmap = {}
    for c in df.columns:
        lc = str(c).lower()
        if 'team' in lc or 'country' in lc or 'nation' in lc:
            colmap[c] = 'Team'
        if 'rank' in lc or 'position' in lc:
            colmap[c] = 'Rank'
        if 'point' in lc or 'pts' in lc or 'points' in lc:
            colmap[c] = 'Points'
    df = df.rename(columns=colmap)

    # Try to find Points column if not matched
    if 'Points' not in df.columns:
        for c in df.columns:
            try:
                if pd.api.types.is_numeric_dtype(df[c]) or df[c].astype(str).str.match(r'^\d+(?:\.\d+)?$').any():
                    df = df.rename(columns={c: 'Points'})
                    break
            except Exception:
                continue

    # Ensure Rank exists
    if 'Rank' not in df.columns:
        df.insert(0, 'Rank', range(1, len(df) + 1))

    # Keep only Rank, Team, Points
    keep = [c for c in ('Rank', 'Team', 'Points') if c in df.columns]
    df = df[keep].copy()
    df['Year'] = year

    # sanitize Points and Rank
    if 'Points' in df.columns:
        df['Points'] = pd.to_numeric(df['Points'].astype(str).str.replace(',', ''), errors='coerce')
    df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')
    # if Rank parse failed entirely, fill sequentially
    if df['Rank'].isna().all():
        df['Rank'] = range(1, len(df) + 1)
    df['Rank'] = df['Rank'].fillna(method='ffill').astype(int)

    return df[['Year', 'Rank', 'Team', 'Points']]


def scrape_wikipedia_year(year: int, timeout: int = 15) -> Optional[pd.DataFrame]:
    """Attempt to extract a ranking table for a specific year from Wikipedia.

    Tries a few likely URL patterns and falls back to the search API.
    Returns a DataFrame or None if none found.
    """
    HEADERS = {'User-Agent': 'Mozilla/5.0'}
    patterns = [
        'https://en.wikipedia.org/wiki/FIFA_World_Rankings_in_{year}',
        'https://en.wikipedia.org/wiki/FIFA_World_Ranking_in_{year}',
        'https://en.wikipedia.org/wiki/FIFA_World_Rankings_{year}',
        'https://en.wikipedia.org/wiki/FIFA_World_Ranking_{year}',
    ]
    for p in patterns:
        url = p.format(year=year)
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code != 200:
                continue
            tables = pd.read_html(r.text)
        except Exception:
            continue
        for df in tables:
            # flatten MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [' '.join([str(p) for p in col if str(p) != 'nan']).strip() for col in df.columns.values]
            cols = [str(c).strip().lower() for c in df.columns.astype(str)]
            has_team = any('team' in c or 'country' in c or 'nation' in c for c in cols)
            has_rank = any('rank' in c or 'position' in c or 'pos' in c for c in cols)
            if not (has_team and has_rank):
                # heuristic: first column numeric and second column contains letters
                try:
                    s0 = str(df.iat[0, 0])
                    s1 = str(df.iat[0, 1])
                    if re.search(r'\d', s0) and re.search('[A-Za-z]', s1):
                        has_rank = True
                        has_team = True
                except Exception:
                    pass
            if has_team and has_rank:
                # pick likely columns
                try:
                    rank_idx = next((i for i, c in enumerate(cols) if 'rank' in c or 'position' in c or c == 'pos'), 0)
                    team_idx = next((i for i, c in enumerate(cols) if 'team' in c or 'country' in c or 'nation' in c), 1 if len(cols) > 1 else 0)
                    pts_idx = next((i for i, c in enumerate(cols) if 'point' in c or 'pts' in c or 'rating' in c), None)
                except Exception:
                    rank_idx, team_idx, pts_idx = 0, 1 if df.shape[1] > 1 else 0, 2 if df.shape[1] > 2 else None
                out = pd.DataFrame()
                out['Rank'] = df.iloc[:, rank_idx]
                out['Team'] = df.iloc[:, team_idx]
                out['Points'] = df.iloc[:, pts_idx] if (pts_idx is not None and pts_idx < df.shape[1]) else pd.NA
                out['Year'] = year
                out['Team'] = out['Team'].astype(str).str.replace('\[.*?\]', '', regex=True).str.replace('\n', ' ', regex=False).str.strip()
                out['Points'] = pd.to_numeric(out['Points'].astype(str).str.replace('[^0-9.+-]', '', regex=True), errors='coerce')
                out['Rank'] = out['Rank'].astype(str).str.extract(r'(\d+)')
                return out[['Year', 'Rank', 'Team', 'Points']]

    # Fallback: use the Wikipedia search API to find candidate pages
    api = 'https://en.wikipedia.org/w/api.php'
    params = {'action': 'query', 'list': 'search', 'srsearch': f'FIFA World Ranking {year}', 'format': 'json', 'srlimit': 10}
    try:
        r = requests.get(api, params=params, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        results = r.json().get('query', {}).get('search', [])
    except Exception:
        return None
    for res in results:
        title = res.get('title')
        if not title or 'FIFA' not in title:
            continue
        url = 'https://en.wikipedia.org/wiki/' + title.replace(' ', '_')
        try:
            pr = requests.get(url, headers=HEADERS, timeout=timeout)
            if pr.status_code != 200:
                continue
            tables = pd.read_html(pr.text)
        except Exception:
            continue
        for df in tables:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [' '.join([str(p) for p in col if str(p) != 'nan']).strip() for col in df.columns.values]
            cols = [str(c).strip().lower() for c in df.columns.astype(str)]
            has_team = any('team' in c or 'country' in c or 'nation' in c for c in cols)
            has_rank = any('rank' in c or 'position' in c or 'pos' in c for c in cols)
            if has_team and has_rank:
                try:
                    rank_idx = next((i for i, c in enumerate(cols) if 'rank' in c or 'position' in c or c == 'pos'), 0)
                    team_idx = next((i for i, c in enumerate(cols) if 'team' in c or 'country' in c or 'nation' in c), 1 if len(cols) > 1 else 0)
                    pts_idx = next((i for i, c in enumerate(cols) if 'point' in c or 'pts' in c or 'rating' in c), None)
                except Exception:
                    rank_idx, team_idx, pts_idx = 0, 1 if df.shape[1] > 1 else 0, 2 if df.shape[1] > 2 else None
                out = pd.DataFrame()
                out['Rank'] = df.iloc[:, rank_idx]
                out['Team'] = df.iloc[:, team_idx]
                out['Points'] = df.iloc[:, pts_idx] if (pts_idx is not None and pts_idx < df.shape[1]) else pd.NA
                out['Year'] = year
                out['Team'] = out['Team'].astype(str).str.replace('\[.*?\]', '', regex=True).str.replace('\n', ' ', regex=False).str.strip()
                out['Points'] = pd.to_numeric(out['Points'].astype(str).str.replace('[^0-9.+-]', '', regex=True), errors='coerce')
                out['Rank'] = out['Rank'].astype(str).str.extract(r'(\d+)')
                return out[['Year', 'Rank', 'Team', 'Points']]
    return None


def aggregate_urls(urls_by_year: Dict[int, str]) -> Tuple[pd.DataFrame, List[int]]:
    """Aggregate ranking data for the provided mapping of year->url.

    Returns (df, missing_years).
    """
    frames = []
    missing = []
    for year, url in sorted(urls_by_year.items()):
        try:
            df = scrape_fifa_rankings(year, url)
            frames.append(df)
            time.sleep(0.2)
        except Exception as e:
            missing.append(year)
    if frames:
        all_df = pd.concat(frames, ignore_index=True)
        # basic cleaning of obvious meta rows
        all_df = all_df[~all_df['Team'].astype(str).str.lower().str.contains('player|note|reference|statistic')]
        return all_df, missing
    return pd.DataFrame(columns=['Year', 'Rank', 'Team', 'Points']), missing


def aggregate_wikipedia(years: List[int]) -> pd.DataFrame:
    """Scrape per-year Wikipedia pages for the supplied years and return concatenated DataFrame."""
    frames = []
    for y in years:
        try:
            df = scrape_wikipedia_year(y)
            if df is not None:
                frames.append(df)
                print(f'Wikipedia: found table for {y} rows={len(df)}')
            else:
                print(f'Wikipedia: no table for {y}')
        except Exception as e:
            print(f'Error scraping wikipedia for {y}: {e}')
        time.sleep(0.2)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=['Year', 'Rank', 'Team', 'Points'])


def clean_and_filter_wc_years(df: pd.DataFrame, canonical_teams_fp: str, wc_years: Optional[List[int]] = None, fuzzy: bool = True) -> pd.DataFrame:
    """Filter DF to World Cup years and canonical national teams; optionally use fuzzy matching.

    Returns cleaned DataFrame with columns ['Year','Rank','Team','Points'] and reassigned ranks per year.
    """
    if wc_years is None:
        wc_years = [2002, 2006, 2010, 2014, 2018, 2022]
    canon = pd.read_csv(canonical_teams_fp)
    canon_set = set(canon['Team'].astype(str).str.strip())

    df = df[df['Year'].isin(wc_years)].copy()
    df['Team'] = df['Team'].astype(str).str.replace('\s+', ' ', regex=True).str.strip()

    matched_rows = []

    # Exact and casefold matching
    canon_cf = {t.casefold(): t for t in canon_set}
    for _, row in df.iterrows():
        team = row['Team']
        if team in canon_set:
            matched_rows.append((team, row))
            continue
        team_cf = team.casefold()
        if team_cf in canon_cf:
            matched_rows.append((canon_cf[team_cf], row))
            continue
        # fuzzy match
        if fuzzy:
            choices = difflib.get_close_matches(team, list(canon_set), n=1, cutoff=0.8)
            if choices:
                matched_rows.append((choices[0], row))
                continue
        # otherwise skip

    if not matched_rows:
        return pd.DataFrame(columns=['Year', 'Rank', 'Team', 'Points'])

    out_rows = []
    for matched_team, row in matched_rows:
        out_rows.append({'Year': int(row['Year']), 'Rank': row.get('Rank'), 'Team': matched_team, 'Points': row.get('Points')})

    out = pd.DataFrame(out_rows)
    # coerce and clean
    out['Rank'] = pd.to_numeric(out['Rank'], errors='coerce')
    out['Points'] = pd.to_numeric(out['Points'], errors='coerce')

    # per-year ordering: prefer Points desc, then existing Rank asc, then Team
    fixed = []
    for y in sorted(out['Year'].unique()):
        sub = out[out['Year'] == y].copy()
        sub['Points_key'] = sub['Points'].fillna(-1e9)
        sub['Rank_key'] = sub['Rank'].fillna(1e9)
        sub = sub.sort_values(['Points_key', 'Rank_key', 'Team'], ascending=[False, True, True]).reset_index(drop=True)
        sub['Rank'] = range(1, len(sub) + 1)
        sub = sub.drop(columns=['Points_key', 'Rank_key'])
        fixed.append(sub)
    if fixed:
        final = pd.concat(fixed, ignore_index=True)
    else:
        final = pd.DataFrame(columns=['Year', 'Rank', 'Team', 'Points'])
    final = final[['Year', 'Rank', 'Team', 'Points']]
    final['Rank'] = final['Rank'].astype(int)
    return final


if __name__ == '__main__':
    # Minimal CLI for convenience. Don't run anything heavy on import.
    import argparse

    parser = argparse.ArgumentParser(description='All-in-one FIFA rankings extractor')
    parser.add_argument('--aggregate', action='store_true', help='Try to aggregate FIFA static pages (example URLs)')
    parser.add_argument('--out', type=str, default='fifa_rankings_aggregated.csv', help='Output CSV for aggregated rankings')
    parser.add_argument('--clean-wc', action='store_true', help='Clean aggregated file for World Cup years using canonical team list')
    parser.add_argument('--in', dest='infile', type=str, default='fifa_rankings_aggregated.csv', help='Input CSV to clean')
    parser.add_argument('--teams', type=str, default='world_cup_all_teams_complete_analysis.csv', help='Canonical teams CSV')
    parser.add_argument('--out-clean', type=str, default='fifa_rankings_wc_years_2002_2022_clean.csv', help='Output cleaned WC CSV')
    args = parser.parse_args()

    if args.aggregate:
        # Example FIFA URLs used previously. Replace or extend as needed.
        urls_by_year = {
            2018: 'https://www.fifa.com/fifa-world-ranking/men?dateId=id13931',
            2022: 'https://www.fifa.com/fifa-world-ranking/men?dateId=id13792',
        }
        df_fifa, missing = aggregate_urls(urls_by_year)
        if not df_fifa.empty:
            df_fifa.to_csv(args.out, index=False)
            print('Wrote', args.out, 'rows', len(df_fifa))
        if missing:
            print('Missing years from FIFA static scraper:', missing)

    if args.clean_wc:
        # load input
        try:
            df_in = pd.read_csv(args.infile)
        except Exception as e:
            raise SystemExit(f'Failed to load input file {args.infile}: {e}')
        final = clean_and_filter_wc_years(df_in, args.teams)
        final.to_csv(args.out_clean, index=False)
        print('Wrote cleaned WC CSV to', args.out_clean, 'rows', len(final))
