"""
Marcel baseline projections for IPL 2026 full-season totals under custom rules.

Marcel = 5-4-3 weighted average of last 3 seasons' points-per-match, regressed
toward a role-specific prior, then blended with 2026 YTD rate, scaled by
remaining group-stage matches (14 - played).

Inputs:
  - data/historical_custom_season_totals.csv    (from 13a)
  - results/FantasyProjections/ipl2026_custom/season_to_date_totals.csv
  - data/Boston IPL 26_Results (1).xlsx         (role + Unsold player list)
  - data/ipl_2026_squads_enriched.csv           (all 250 2026 squad players)

Output:
  - results/FantasyProjections/ipl2026_custom/marcel_projections_2026.csv
"""
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
HIST_FILE = PROJECT_ROOT / 'data' / 'historical_custom_season_totals.csv'
YTD_FILE = PROJECT_ROOT / 'results' / 'FantasyProjections' / 'ipl2026_custom' / 'season_to_date_totals.csv'
BOSTON_FILE = PROJECT_ROOT / 'data' / 'Boston IPL 26_Results (1).xlsx'
SQUAD_FILE = PROJECT_ROOT / 'data' / 'ipl_2026_squads_enriched.csv'
OUT_FILE = PROJECT_ROOT / 'results' / 'FantasyProjections' / 'ipl2026_custom' / 'marcel_projections_2026.csv'

TOTAL_GROUP_MATCHES = 14
TARGET_SEASON = 2026

# Marcel weights for last 3 seasons
W_S1, W_S2, W_S3 = 5, 4, 3
# Shrinkage prior (in player-matches of "league-average" experience)
MARCEL_PRIOR_MATCHES = 100
# YTD blending: how heavily to weight current-season evidence
YTD_PRIOR_MATCHES = 20

# Boston player name → cricsheet name (same as reconcile script)
MANUAL = {
    'Vaibhav Sooryavanshi':'V Suryavanshi','Philip Salt':'PD Salt','Jos Buttler':'JC Buttler',
    'Yashasvi Jaiswal':'YBK Jaiswal','Prabhsimran Singh':'P Simran Singh','Sanju Samson':'SV Samson',
    'Suryakumar Yadav':'SA Yadav','Angkrish Raghuvanshi':'A Raghuvanshi','Ayush Mhatre':'A Mhatre',
    'Shivam Dube':'SM Dube','Shubman Gill':'Shubman Gill','Virat Kohli':'V Kohli','Ishan Kishan':'Ishan Kishan',
    'Heinrich Klaasen':'H Klaasen','Rajat Patidar':'RM Patidar','Shreyas Iyer':'SS Iyer',
    'KL Rahul':'KL Rahul','Travis Head':'TM Head','Jofra Archer':'JC Archer','Arshdeep Singh':'Arshdeep Singh',
    'Noor Ahmad':'Noor Ahmad','Mohammed Siraj':'Mohammed Siraj','Devdutt Padikkal':'D Padikkal',
    'Quinton de Kock':'Q de Kock','Hardik Pandya':'HH Pandya','Jasprit Bumrah':'JJ Bumrah',
    'Rishabh Pant':'RR Pant','Abhishek Sharma':'Abhishek Sharma','Ruturaj Gaikwad':'RD Gaikwad',
    'Sai Sudharsan':'B Sai Sudharsan','Rohit Sharma':'RG Sharma','Nicholas Pooran':'N Pooran',
    'Priyansh Arya':'Priyansh Arya','Ajinkya Rahane':'AM Rahane','Pathum Nissanka':'WPN Nissanka',
    'Josh Hazlewood':'JR Hazlewood','Jitesh Sharma':'Jitesh Sharma','Cameron Green':'CD Green',
    'Mitchell Marsh':'MR Marsh','Kuldeep Yadav':'Kuldeep Yadav','Tilak Varma':'Tilak Varma',
    'Dewald Brevis':'DR Brevis','Tim David':'TH David','Nehal Wadhera':'N Wadhera','Naman Dhir':'N Dhir',
    'Shimron Hetmyer':'SO Hetmyer','Prasidh Krishna':'M Prasidh Krishna','Bhuvneshwar Kumar':'B Kumar',
    'Marco Jansen':'M Jansen','Sunil Narine':'SP Narine','Rinku Singh':'RK Singh','Aiden Markram':'AK Markram',
    'Varun Chakaravarthy':'V Chakravarthy','Manish Pandey':'MK Pandey','Rovman Powell':'R Powell',
    'Nitish Rana':'N Rana','Finn Allen':'FH Allen','Trent Boult':'TA Boult',
    'Lockie Ferguson':'LH Ferguson','Kyle Jamieson':'KA Jamieson','Pat Cummins':'PJ Cummins',
    'Rashid Khan':'Rashid Khan','Ravindra Jadeja':'RA Jadeja','Axar Patel':'AR Patel','David Miller':'DA Miller',
    'Yuzvendra Chahal':'YS Chahal','Prashant Veer':'Prashant Veer','Jacob Bethell':'JG Bethell',
    'Matheesha Pathirana':'M Pathirana',
}


def normalize_role(r: str) -> str:
    if not isinstance(r, str):
        return 'BAT'  # default
    rl = r.lower()
    if 'wicket' in rl: return 'WK'
    if 'all' in rl or 'rounder' in rl: return 'AR'
    if 'bowl' in rl: return 'BOWL'
    return 'BAT'


def load_roster() -> pd.DataFrame:
    """Build the roster of every player we need to project: Boston drafted +
    Unsold + any extra 2026 squad players from the enriched squad file."""
    # Boston file first (has Role)
    boston = []
    x = pd.ExcelFile(BOSTON_FILE)
    for s in x.sheet_names:
        if s == 'Awards':
            continue
        d = pd.read_excel(x, sheet_name=s)
        if 'Player' not in d.columns:
            continue
        d = d[d['Player'].notna()].copy()
        d['Player'] = d['Player'].astype(str).str.strip()
        d = d[~d['Player'].str.contains('TOTAL', case=False, na=False)]
        d['owner'] = 'Unsold' if s == 'Unsold Players' else s
        keep = ['owner', 'Player']
        for c in ('Role', 'Team', 'Price (₹L)'):
            if c in d.columns:
                keep.append(c)
        boston.append(d[keep])
    boston = pd.concat(boston, ignore_index=True).rename(
        columns={'Player': 'player', 'Role': 'role_raw', 'Team': 'ipl_team'})
    boston['role'] = boston['role_raw'].apply(normalize_role)
    boston['cricsheet_name'] = boston['player'].map(MANUAL)

    # Enriched squad — names not in Boston get added; use playing_role + Player_Type for role
    squad = pd.read_csv(SQUAD_FILE)
    boston_names = set(boston['player'])
    extras = squad[~squad['Player'].isin(boston_names)].copy()
    extras = extras.rename(columns={'Player': 'player', 'Team': 'ipl_team'})
    extras['owner'] = 'Squad (not in Boston)'
    extras['role'] = extras['playing_role'].apply(normalize_role)
    extras['Price (₹L)'] = np.nan
    extras['role_raw'] = extras['playing_role']
    extras['cricsheet_name'] = extras['player'].map(MANUAL)  # mostly NaN
    extras = extras[['owner', 'player', 'role_raw', 'ipl_team', 'Price (₹L)', 'role', 'cricsheet_name']]

    roster = pd.concat([boston, extras], ignore_index=True)
    roster = roster.drop_duplicates(subset='player', keep='first').reset_index(drop=True)
    return roster


def infer_role_from_history(name: str, hist: pd.DataFrame) -> str:
    """Best-effort role inference from historical stats when not in Boston/squad files."""
    h = hist[hist['player_name'] == name]
    if h.empty:
        return 'BAT'
    wkts = h['wickets'].sum()
    runs = h['runs'].sum()
    balls = h['balls_faced'].sum()
    overs = h['overs'].sum()
    stumps = h['stumpings'].sum()
    if stumps >= 2: return 'WK'
    if overs >= 20 and runs >= 200 and balls >= 150: return 'AR'
    if wkts >= 5 and overs >= 10 and runs < 200: return 'BOWL'
    return 'BAT'


def get_season_ppm(hist: pd.DataFrame, name: str, season: int) -> tuple[float, int]:
    """Returns (ppm, matches) for a player-season. (0, 0) if absent."""
    h = hist[(hist['player_name'] == name) & (hist['season'] == season)]
    if h.empty:
        return 0.0, 0
    r = h.iloc[0]
    return float(r['ppm']), int(r['matches'])


def compute_role_prior_ppm(hist: pd.DataFrame) -> dict:
    """Mean PPM per role across 2022-2025 seasons (weighted by matches).
    Role is inferred per player-season from their stats."""
    recent = hist[hist['season'].isin([2022, 2023, 2024, 2025])].copy()
    # Infer role for each player-season
    def role_from_row(r):
        if r['stumpings'] >= 2: return 'WK'
        if r['overs'] >= 20 and r['runs'] >= 150: return 'AR'
        if r['wickets'] >= 5 and r['overs'] >= 10 and r['runs'] < 150: return 'BOWL'
        return 'BAT'
    recent['role'] = recent.apply(role_from_row, axis=1)
    prior = {}
    for role, g in recent.groupby('role'):
        total_pts = g['total_pts'].sum()
        total_m = g['matches'].sum()
        prior[role] = total_pts / total_m if total_m else 0.0
    return prior


def main():
    print("Loading inputs...")
    hist = pd.read_csv(HIST_FILE)
    ytd = pd.read_csv(YTD_FILE)
    ytd_points_col = 'V_NO_SR_total'  # closest to published rules
    roster = load_roster()

    # Team played matches as of the YTD snapshot (for remaining-match calc)
    # We parse match_id prefix "YYYYMMDD_teamA_vs_teamB" from per-match file
    per_match = pd.read_csv(PROJECT_ROOT / 'results' / 'FantasyProjections' / 'ipl2026_custom' / 'per_match_points.csv')
    per_match['date_str'] = per_match['match_id'].str.extract(r'^(\d{8})_')
    per_match['teamA'] = per_match['match_id'].str.extract(r'^\d{8}_([a-z]+)_vs_')
    per_match['teamB'] = per_match['match_id'].str.extract(r'_vs_([a-z]+)$')
    team_games = {}
    for mid in per_match['match_id'].unique():
        teams = mid.split('_vs_')
        teamA = teams[0].split('_', 1)[1]
        teamB = teams[1]
        team_games[teamA] = team_games.get(teamA, 0) + 1
        team_games[teamB] = team_games.get(teamB, 0) + 1

    IPL_TEAM_MAP = {
        'Chennai': 'chennaisuperkings', 'Delhi': 'delhicapitals', 'Gujarat': 'gujarattitans',
        'Hyderabad': 'sunrisershyderabad', 'Kolkata': 'kolkataknightriders',
        'Lucknow': 'lucknowsupergiants', 'Mumbai': 'mumbaiindians', 'Punjab': 'punjabkings',
        'Rajasthan': 'rajasthanroyals', 'Bengaluru': 'royalchallengersbengaluru',
        # Enriched squad labels
        'CSK': 'chennaisuperkings', 'DC': 'delhicapitals', 'GT': 'gujarattitans',
        'SRH': 'sunrisershyderabad', 'KKR': 'kolkataknightriders',
        'LSG': 'lucknowsupergiants', 'MI': 'mumbaiindians', 'PBKS': 'punjabkings',
        'RR': 'rajasthanroyals', 'RCB': 'royalchallengersbengaluru',
    }

    def team_played(team_str):
        key = IPL_TEAM_MAP.get(team_str, '')
        return team_games.get(key, 0)

    roster['team_played'] = roster['ipl_team'].apply(team_played)
    roster['matches_remaining'] = (TOTAL_GROUP_MATCHES - roster['team_played']).clip(lower=0)

    # Role prior
    role_prior = compute_role_prior_ppm(hist)
    print(f"Role prior PPM: {role_prior}")

    # Compute Marcel per player
    print("Computing Marcel projections...")
    rows = []
    for _, p in roster.iterrows():
        name = p['cricsheet_name'] if isinstance(p.get('cricsheet_name'), str) else None
        # Fall back to raw player name (cricsheet may use raw for most names already)
        hist_name = name if name and ((hist['player_name'] == name).any()) else p['player']

        # Last 3 seasons
        ppm_25, m_25 = get_season_ppm(hist, hist_name, 2025)
        ppm_24, m_24 = get_season_ppm(hist, hist_name, 2024)
        ppm_23, m_23 = get_season_ppm(hist, hist_name, 2023)

        role = p['role']
        prior_ppm = role_prior.get(role, 28.0)

        num = W_S1 * ppm_25 * m_25 + W_S2 * ppm_24 * m_24 + W_S3 * ppm_23 * m_23 + MARCEL_PRIOR_MATCHES * prior_ppm
        denom = W_S1 * m_25 + W_S2 * m_24 + W_S3 * m_23 + MARCEL_PRIOR_MATCHES
        marcel_ppm = num / denom

        # YTD from 2026
        ytd_hit = ytd[ytd['player_name'] == hist_name]
        if ytd_hit.empty:
            ytd_ppm = 0.0
            ytd_matches = 0
            ytd_pts = 0
        else:
            r = ytd_hit.iloc[0]
            ytd_matches = int(r['matches'])
            ytd_pts = int(r[ytd_points_col])
            ytd_ppm = ytd_pts / ytd_matches if ytd_matches else 0.0

        # Blend YTD + Marcel
        if ytd_matches > 0:
            ytd_weight = ytd_matches / (ytd_matches + YTD_PRIOR_MATCHES)
            blended_ppm = ytd_weight * ytd_ppm + (1 - ytd_weight) * marcel_ppm
        else:
            blended_ppm = marcel_ppm

        # Rookie flag: no prior IPL and no YTD
        is_rookie = (m_23 + m_24 + m_25 == 0) and (ytd_matches == 0)
        history_matches = m_23 + m_24 + m_25

        remaining = int(p['matches_remaining'])
        full_season_proj = ytd_pts + blended_ppm * remaining

        rows.append({
            'player': p['player'],
            'cricsheet_name': hist_name,
            'owner': p['owner'],
            'ipl_team': p['ipl_team'],
            'role': role,
            'role_raw': p.get('role_raw'),
            'ytd_matches': ytd_matches,
            'ytd_points': ytd_pts,
            'ytd_ppm': round(ytd_ppm, 2),
            'matches_remaining': remaining,
            'ppm_2023': round(ppm_23, 2),  'm_2023': m_23,
            'ppm_2024': round(ppm_24, 2),  'm_2024': m_24,
            'ppm_2025': round(ppm_25, 2),  'm_2025': m_25,
            'history_matches': history_matches,
            'is_rookie': is_rookie,
            'role_prior_ppm': round(prior_ppm, 2),
            'marcel_ppm': round(marcel_ppm, 2),
            'blended_ppm': round(blended_ppm, 2),
            'marcel_rest_of_season': round(blended_ppm * remaining, 1),
            'marcel_full_season': round(full_season_proj, 1),
            'price_lakhs': p.get('Price (₹L)'),
        })

    proj = pd.DataFrame(rows).sort_values('marcel_full_season', ascending=False).reset_index(drop=True)
    proj['rank'] = proj.index + 1
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    proj.to_csv(OUT_FILE, index=False)

    print(f"\n✓ {OUT_FILE}  ({len(proj)} players)")
    print(f"  Rookies (no IPL history, no YTD): {proj['is_rookie'].sum()}")
    print(f"\n── Top 20 projections ──")
    print(proj.head(20)[['rank','player','owner','ipl_team','role','ytd_points',
                         'matches_remaining','blended_ppm','marcel_full_season']].to_string(index=False))
    print(f"\n── Summary by role ──")
    print(proj.groupby('role').agg(n=('player','count'),
                                    mean_proj=('marcel_full_season','mean'),
                                    max_proj=('marcel_full_season','max')).round(1))


if __name__ == '__main__':
    main()
