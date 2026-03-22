"""
Enrich IPL 2026 squad list with cricsheet_id, cricinfo_id, and player metadata.
Matches against multiple sources: auction enriched data, player_metadata, all_players_enriched.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from difflib import SequenceMatcher

DATA = Path(__file__).parent.parent / 'data'

# ── Load squad data ──────────────────────────────────────────────────────
squads = pd.read_csv(DATA / 'ipl_2026_full_squads.csv')
print(f"Squad players: {len(squads)}")

# ── Load all metadata sources ────────────────────────────────────────────
# Source 1: Auction enriched (best - has cricinfo_id, cricsheet_id, playing_role)
auction = pd.read_csv(DATA / 'ipl_2026_auction_enriched.csv')
auction = auction.rename(columns={'PLAYER': 'player_auction', 'COUNTRY': 'country_auction'})
auction_lookup = {}
for _, row in auction.iterrows():
    if pd.notna(row.get('cricsheet_id')) and str(row.get('cricsheet_id')).strip():
        name = str(row['player_auction']).strip()
        auction_lookup[name.lower()] = {
            'cricinfo_id': row.get('cricinfo_id'),
            'cricsheet_id': row.get('cricsheet_id'),
            'full_name': row.get('full_name'),
            'country': row.get('country'),
            'dob': row.get('dob'),
            'batting_style': row.get('batting_style'),
            'bowling_style': row.get('bowling_style'),
            'playing_role': row.get('playing_role'),
        }

# Source 2: all_players_enriched (cricsheet ball-by-ball players)
all_players = pd.read_csv(DATA / 'all_players_enriched.csv')
allp_lookup = {}
for _, row in all_players.iterrows():
    name = str(row.get('name', '')).strip()
    full = str(row.get('full_name', '')).strip()
    cid = row.get('cricsheet_id', '')
    if pd.notna(cid) and str(cid).strip():
        entry = {
            'cricinfo_id': row.get('cricinfo_id'),
            'cricsheet_id': cid,
            'full_name': full if full else None,
            'country': row.get('country'),
            'dob': row.get('dob'),
            'batting_style': row.get('batting_style'),
            'bowling_style': row.get('bowling_style'),
        }
        allp_lookup[name.lower()] = entry
        if full:
            allp_lookup[full.lower()] = entry

# Source 3: player_metadata.csv (from cricketdata R package)
meta = pd.read_csv(DATA / 'player_metadata.csv')
meta_lookup = {}
for _, row in meta.iterrows():
    pname = str(row.get('player_name', '')).strip()
    fname = str(row.get('name', '')).strip()
    pid = row.get('player_id', '')
    if pd.notna(pid) and str(pid).strip():
        entry = {
            'cricinfo_id': row.get('cricinfo_id') if 'cricinfo_id' in row.index else None,
            'cricsheet_id': pid,
            'full_name': row.get('full_name'),
            'country': row.get('country'),
            'batting_style': row.get('batting_style'),
            'bowling_style': row.get('bowling_style'),
            'playing_role': row.get('playing_role'),
        }
        meta_lookup[pname.lower()] = entry
        meta_lookup[fname.lower()] = entry

# Source 4: player_metadata_full.csv
meta_full = pd.read_csv(DATA / 'player_metadata_full.csv')
metaf_lookup = {}
for _, row in meta_full.iterrows():
    pname = str(row.get('player_name', '')).strip()
    fname = str(row.get('full_name', '')).strip()
    pid = row.get('player_id', '')
    if pd.notna(pid) and str(pid).strip():
        entry = {
            'cricinfo_id': row.get('cricinfo_id'),
            'cricsheet_id': pid,
            'full_name': fname if fname else None,
            'country': row.get('country'),
            'dob': row.get('dob'),
            'batting_style': row.get('batting_style'),
            'bowling_style': row.get('bowling_style'),
            'playing_role': row.get('playing_role'),
        }
        metaf_lookup[pname.lower()] = entry
        if fname:
            metaf_lookup[fname.lower()] = entry

# Source 5: 2025 player list (for team context)
players_2025 = pd.read_csv(DATA / 'IPL_2025_Players_List.csv')
p2025_lookup = {}
for _, row in players_2025.iterrows():
    name = str(row.get('Player Name', '')).strip()
    p2025_lookup[name.lower()] = name


# ── Name normalization helpers ───────────────────────────────────────────
NAME_ALIASES = {
    'ms dhoni': 'mahendra singh dhoni',
    'kl rahul': 'lokesh rahul',
    'mohammed siraj': 'mohammed siraj',
    'abishek porel': 'abishek porel',
    'varun chakaravarthy': 'varun chakravarthy',
    'allah ghazanfar': 'allah ghazanfar',
    'shahrukh khan': 'shahrukh khan',
    'gurnoor singh brar': 'gurnoor brar',
    'raj angad bawa': 'raj angad bawa',
    't natarajan': 't natarajan',
    'lungisani ngidi': 'lungi ngidi',
    'm. siddharth': 'manimaran siddharth',
    'mohammad shami': 'mohammed shami',
    'blessing muzarabani': 'blessing muzarabani',
    'mitch owen': 'mitchell owen',
    'vyshak vijaykumar': 'vijaykumar vyshak',
    'pravin dubey': 'praveen dubey',
    'sai kishore': 'r sai kishore',
    'mohd. arshad khan': 'arshad khan',
    'shahbaz ahamad': 'shahbaz ahmed',
}

# Players that should NOT fuzzy match - they are genuinely new/young players
# with no prior cricsheet data. Set to None to prevent bad fuzzy matches.
BAD_FUZZY_BLOCKLIST = {
    'kartik sharma',         # NOT Karn Sharma
    'gurjapneet singh',      # NOT Gurpreet Singh
    'sahil parakh',          # NOT Subham Pradhan
    'mukul choudhary',       # NOT Mukesh Choudhary
    'naman tiwari',          # NOT a fuzzy match to nan
    'mohammad izhar',        # NOT Mohd Israr
    'ravi singh',            # NOT Gavin Singh
    'sushant mishra',        # NOT Shantanu Mishra
    'mangesh yadav',         # NOT Umeshkumar Yadav
    'abhinandan singh',      # NOT Anand Singh
    'vihaan malhotra',       # NOT Ishan Malhotra
    'shivang kumar',         # NOT Shivam Kumar
    'prashant veer',         # young player, no prior data
    'tejasvi singh',         # NOT another Tejasvi
    'harnoor pannu',         # NOT Harnoor Singh (different player)
    'digvesh singh',         # NOT another Digvesh
    'sahil parakh',          # young player, no prior data
}


def fuzzy_match(name, lookup_dict, threshold=0.80):
    """Find best fuzzy match in a lookup dict."""
    name_lower = name.lower().strip()
    best_match = None
    best_score = 0
    for key in lookup_dict:
        score = SequenceMatcher(None, name_lower, key).ratio()
        if score > best_score and score >= threshold:
            best_score = score
            best_match = key
    return best_match, best_score


def lookup_player(name):
    """Try all sources to find player metadata."""
    name_lower = name.lower().strip()

    # Try exact match across all sources (in priority order)
    for source_name, lookup in [
        ('auction', auction_lookup),
        ('all_players', allp_lookup),
        ('meta', meta_lookup),
        ('meta_full', metaf_lookup),
    ]:
        if name_lower in lookup:
            return lookup[name_lower], source_name, 'exact'

    # Try alias
    if name_lower in NAME_ALIASES:
        alias = NAME_ALIASES[name_lower]
        for source_name, lookup in [
            ('auction', auction_lookup),
            ('all_players', allp_lookup),
            ('meta', meta_lookup),
            ('meta_full', metaf_lookup),
        ]:
            if alias in lookup:
                return lookup[alias], source_name, 'alias'

    # Skip fuzzy matching for blocklisted players
    if name_lower in BAD_FUZZY_BLOCKLIST:
        return None, None, 'unmatched'

    # Try fuzzy match across all sources
    all_lookups = {
        'auction': auction_lookup,
        'all_players': allp_lookup,
        'meta': meta_lookup,
        'meta_full': metaf_lookup,
    }

    best_result = None
    best_score = 0
    best_source = None

    for source_name, lookup in all_lookups.items():
        match_key, score = fuzzy_match(name, lookup)
        if match_key and score > best_score:
            best_score = score
            best_result = lookup[match_key]
            best_source = source_name

    if best_result:
        return best_result, best_source, f'fuzzy_{best_score:.2f}'

    return None, None, 'unmatched'


# ── Enrich each player ──────────────────────────────────────────────────
results = []
match_stats = {'exact': 0, 'alias': 0, 'fuzzy': 0, 'unmatched': 0}

for _, row in squads.iterrows():
    player_name = row['Player']
    info, source, method = lookup_player(player_name)

    if method == 'exact':
        match_stats['exact'] += 1
    elif method == 'alias':
        match_stats['alias'] += 1
    elif method.startswith('fuzzy'):
        match_stats['fuzzy'] += 1
    else:
        match_stats['unmatched'] += 1

    result = {
        'Player': player_name,
        'Team': row['Team'],
        'Price_Cr': row['Price_Cr'],
        'Price_Lakh': row['Price_Lakh'],
        'Acquisition': row['Acquisition'],
        'Player_Type': row['Player_Type'],
        'cricsheet_id': info.get('cricsheet_id') if info else None,
        'cricinfo_id': info.get('cricinfo_id') if info else None,
        'full_name': info.get('full_name') if info else None,
        'country': info.get('country') if info else None,
        'dob': info.get('dob') if info else None,
        'batting_style': info.get('batting_style') if info else None,
        'bowling_style': info.get('bowling_style') if info else None,
        'playing_role': info.get('playing_role') if info else None,
        'match_method': method,
        'match_source': source,
    }
    results.append(result)

enriched = pd.DataFrame(results)

# ── Report ───────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"MATCHING RESULTS")
print(f"{'='*60}")
print(f"  Exact:     {match_stats['exact']}")
print(f"  Alias:     {match_stats['alias']}")
print(f"  Fuzzy:     {match_stats['fuzzy']}")
print(f"  Unmatched: {match_stats['unmatched']}")
print(f"  Total:     {len(enriched)}")
print(f"  Coverage:  {(len(enriched) - match_stats['unmatched']) / len(enriched) * 100:.1f}%")

# Show unmatched
unmatched = enriched[enriched['match_method'] == 'unmatched']
if len(unmatched) > 0:
    print(f"\n  Unmatched players:")
    for _, row in unmatched.iterrows():
        print(f"    - {row['Player']} ({row['Team']})")

# Show fuzzy matches for review
fuzzy = enriched[enriched['match_method'].str.startswith('fuzzy', na=False)]
if len(fuzzy) > 0:
    print(f"\n  Fuzzy matches (review):")
    for _, row in fuzzy.iterrows():
        print(f"    - {row['Player']} → {row['full_name']} ({row['match_method']}, {row['match_source']})")

# ── Save ─────────────────────────────────────────────────────────────────
out_path = DATA / 'ipl_2026_squads_enriched.csv'
enriched.to_csv(out_path, index=False)
print(f"\nSaved to: {out_path}")
print(f"Columns: {list(enriched.columns)}")

# Also show team summary
print(f"\n{'='*60}")
print(f"TEAM SUMMARY")
print(f"{'='*60}")
for team in sorted(enriched['Team'].unique()):
    team_df = enriched[enriched['Team'] == team]
    matched = team_df[team_df['match_method'] != 'unmatched']
    total_spend = team_df['Price_Cr'].sum()
    print(f"  {team}: {len(team_df)} players, {len(matched)} matched, ₹{total_spend:.2f} Cr total")
