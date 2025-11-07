import requests
import pandas as pd
from bs4 import BeautifulSoup
import re

def get_avg_experience(year, url):
    """
    Scrape average international caps (experience) of each national team from Wikipedia World Cup squad page.
    Args:
        year: int
        url: string Wikipedia URL containing squad tables
    Returns:
        DataFrame with columns ['Year','Team','Avg_Experience_Caps']
    """
    # set a browser-like User-Agent to avoid HTTP 403 from some hosts (e.g., Wikipedia)
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0 Safari/537.36'}
    r = requests.get(url, headers=headers, timeout=15)
    soup = BeautifulSoup(r.text, 'html.parser')

    team_experience = []
    for heading in soup.find_all(['h2', 'h3', 'h4']):
        # extract team name from heading id or mw-headline span
        team = None
        if heading.get('id'):
            team = heading.get('id')
        else:
            span = heading.find('span', {'class': 'mw-headline'})
            if span and span.text.strip():
                team = span.text.strip()
        if not team:
            continue
        # skip non-team headings
        if str(team).lower().startswith(('group', 'statistics', 'notes', 'references', 'external', 'players', 'see_also')):
            continue

        # traverse forward in document order to find the first table that contains caps/appearances info
        table = None
        for sib in heading.next_elements:
            if getattr(sib, 'name', None) == 'table':
                # inspect header cells
                ths = sib.find_all('th')
                headers = [th.get_text(separator=' ').strip() for th in ths]
                header_text = ' | '.join(headers)
                if re.search('(caps|appearances|apps)', header_text, flags=re.I):
                    table = sib
                    break
        if table is None:
            print(f"Skipped {team}: No suitable table with caps column found.")
            continue

        # parse the table rows with BeautifulSoup to avoid pandas/lxml dependency
        rows = table.find_all('tr')
        if not rows or len(rows) < 2:
            print(f"Skipped {team}: table has no data rows")
            continue
        # header cells
        header_cells = rows[0].find_all(['th', 'td'])
        headers = [hc.get_text(separator=' ').strip() for hc in header_cells]
        caps_idx = None
        for i, h in enumerate(headers):
            if re.search('(caps|appearances|apps)', h, flags=re.I):
                caps_idx = i
                break

        caps_list = []
        for row in rows[1:]:
            cells = row.find_all(['td', 'th'])
            if not cells:
                continue
            val = None
            if caps_idx is not None and caps_idx < len(cells):
                val = cells[caps_idx].get_text(separator=' ').strip()
            else:
                # try to find any numeric token in the row that looks like caps (int between 0 and 500)
                text = ' '.join([c.get_text(separator=' ').strip() for c in cells])
                m = re.findall(r'\b(\d{1,3})\b', text)
                if m:
                    # heuristics: pick the first reasonable small integer that is not part of date
                    for token in m:
                        n = int(token)
                        if 0 <= n <= 500:
                            val = token
                            break
            if not val:
                continue
            # sanitize numeric value (remove commas, parentheses)
            mnum = re.search(r'(\d{1,3})', str(val))
            if mnum:
                try:
                    caps_list.append(int(mnum.group(1)))
                except Exception:
                    continue

        if not caps_list:
            print(f"Skipped {team}: no caps values parsed from table")
            continue

        avg_caps = float(pd.Series(caps_list).mean())
        team_experience.append({'Year': year, 'Team': team, 'Avg_Experience_Caps': round(avg_caps, 2)})

    if not team_experience:
        print("WARNING: No team experience data found.")
    return pd.DataFrame(team_experience)

# Example usage for 2022 World Cup:
def run_years_experience(years, out_path='world_cup_2002_2022_team_avg_experience.csv'):
    all_rows = []
    for year in years:
        url = f"https://en.wikipedia.org/wiki/{year}_FIFA_World_Cup_squads"
        print(f"Processing experience for {year} -> {url}")
        try:
            df = get_avg_experience(year, url)
            if not df.empty:
                all_rows.append(df)
            else:
                print(f"No experience data for year {year}")
        except Exception as e:
            print(f"Error processing {year}: {e}")
    if all_rows:
        master = pd.concat(all_rows, ignore_index=True)
        # filter out non-team meta rows (e.g., Player_representation..., Coaches..., Notes)
        skip_keywords = ['player', 'players', 'representation', 'coach', 'average', 'goalkeeper', 'captain', 'statistics', 'note', 'reference', 'external', 'age', 'outfield']
        def is_meta(team_name: str) -> bool:
            tn = str(team_name).lower()
            for k in skip_keywords:
                if k in tn:
                    return True
            return False

        before = len(master)
        master = master[~master['Team'].apply(is_meta)].reset_index(drop=True)
        after = len(master)
        master.to_csv(out_path, index=False)
        print(f"Saved aggregated experience CSV to: {out_path} (rows before filter: {before}, after filter: {after})")
    else:
        print('No experience data collected for requested years')


if __name__ == '__main__':
    wc_years = [2002, 2006, 2010, 2014, 2018, 2022]
    run_years_experience(wc_years)