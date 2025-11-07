import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import time
from typing import List

def get_squad_avg_age(year, url):
    # set a browser-like User-Agent to avoid HTTP 403 from some hosts (e.g., Wikipedia)
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0 Safari/537.36'}
    r = requests.get(url, headers=headers, timeout=15)
    soup = BeautifulSoup(r.text, 'html.parser')

    team_ages = []
    # Try all possible heading levels used for teams
    for heading in soup.find_all(['h2', 'h3', 'h4']):
        # Extract team name from several possible heading formats:
        # 1) <h3 id="Argentina">Argentina</h3>
        # 2) <h3><span class="mw-headline" id="Argentina">Argentina</span></h3>
        team = None
        if heading.get('id'):
            team = heading.get('id')
        else:
            span = heading.find('span', {'class': 'mw-headline'})
            if span and span.text.strip():
                team = span.text.strip()
        if not team:
            continue
        # skip non-team headings like Group_*, Statistics, Notes, References
        skip_prefixes = ('group', 'statistics', 'notes', 'references', 'external', 'average_age', 'outfield', 'goalkeepers', 'captains', 'coaches', 'player_representation')
        if str(team).lower().startswith(skip_prefixes):
            # print(f"Skipping non-team heading: {team}")
            continue
        # Walk siblings to find the first following <table> that contains a parsable age or DOB column
        table = None
        # traverse forward in document order to find tables that may be nested inside containers
        for sib in heading.next_elements:
            if getattr(sib, 'name', None) == 'table':
                # inspect header cells
                ths = sib.find_all('th')
                headers = [th.get_text(separator=' ').strip() for th in ths]
                header_text = ' | '.join(headers)
                if re.search('[Aa]ge', header_text) or re.search('Date of birth|DOB|Birth', header_text, flags=re.IGNORECASE):
                    table = sib
                    break
        if table is None:
            print(f"Skipped {team}: no suitable table with an age or DOB column found")
            continue
        # Parse the table rows to extract ages without pandas to avoid external parser deps
        # Identify header cell index for age or DOB
        header_cells = table.find_all('tr')[0].find_all(['th','td'])
        headers = [hc.get_text(separator=' ').strip() for hc in header_cells]
        age_idx = None
        dob_idx = None
        for i, h in enumerate(headers):
            if re.search('[Aa]ge', h):
                age_idx = i
                break
            if re.search('Date of birth|DOB|Birth', h, flags=re.IGNORECASE):
                dob_idx = i
        ages_list = []
        for row in table.find_all('tr')[1:]:
            cells = row.find_all(['td','th'])
            if not cells:
                continue
            # pick appropriate cell
            text_val = None
            if age_idx is not None and age_idx < len(cells):
                text_val = cells[age_idx].get_text(separator=' ').strip()
            elif dob_idx is not None and dob_idx < len(cells):
                text_val = cells[dob_idx].get_text(separator=' ').strip()
            if not text_val:
                continue
            # Try to extract age from parentheses like '1 Jan 1990 (32)'
            m = re.search(r'\((\d{1,3})\)', text_val)
            if m:
                try:
                    ages_list.append(int(m.group(1)))
                    continue
                except Exception:
                    pass
            # If no parentheses age, try to parse a year from the text (e.g., '1 Jan 1990')
            ymatch = re.search(r'(19\d{2}|20\d{2})', text_val)
            if ymatch:
                try:
                    birth_year = int(ymatch.group(1))
                    ages_list.append(year - birth_year)
                    continue
                except Exception:
                    pass
            # otherwise skip
        ages = pd.to_numeric(pd.Series(ages_list), errors='coerce')
        avg_age = ages.mean()
        if not pd.isna(avg_age):
            team_ages.append({'Year': year, 'Team': team, 'Avg_Squad_Age': round(avg_age, 2)})
        else:
            print(f"Skipped {team}: could not compute mean age")
    if not team_ages:
        print("WARNING: No team data found!")
    return pd.DataFrame(team_ages)

# Usage:
def run_years(years: List[int], out_path: str = 'world_cup_2002_2022_squad_avg_age.csv'):
    all_rows = []
    for year in years:
        url = f"https://en.wikipedia.org/wiki/{year}_FIFA_World_Cup_squads"
        print(f"Processing {year} -> {url}")
        try:
            df = get_squad_avg_age(year, url)
            if not df.empty:
                all_rows.append(df)
            else:
                print(f"No data for year {year}")
        except Exception as e:
            print(f"Error processing {year}: {e}")
        time.sleep(1.0)  # be polite

    if all_rows:
        master = pd.concat(all_rows, ignore_index=True)
        master.to_csv(out_path, index=False)
        print(f"Saved aggregated CSV to: {out_path} (rows: {len(master)})")
    else:
        print('No data collected for requested years')


if __name__ == '__main__':
    # default: World Cup tournament years between 2002 and 2022 inclusive
    wc_years = [2002, 2006, 2010, 2014, 2018, 2022]
    run_years(wc_years)
