import requests
from pprint import pprint

URL = 'https://www.fifa.com/fifa-world-ranking/men?dateId=id13792'
headers = {'User-Agent':'Mozilla/5.0'}
resp = requests.get(URL, headers=headers, timeout=15)
text = resp.text

candidates = ['"rank"', '"rankings"', 'ranking', 'points', 'pts', 'window.__', 'dataLayer', 'application/json']
found = {}
for token in candidates:
    idx = text.lower().find(token)
    found[token] = idx

print('Length of HTML:', len(text))
print('Token positions (negative means not found):')
for k,v in found.items():
    print(f'{k}: {v}')

# show first 2000 chars to inspect
print('\n--- head 2000 chars ---')
print(text[:2000])

# show first <script> tags to look for embedded JSON
import re
scripts = re.findall(r'<script[^>]*>(.*?)</script>', text, flags=re.S|re.I)
print(f'Found {len(scripts)} <script> tags; printing first 5 lengths:')
for i,s in enumerate(scripts[:5]):
    print(i, len(s))
    print(s[:400])
    print('----')
