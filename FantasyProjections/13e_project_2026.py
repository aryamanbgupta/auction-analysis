"""
Project IPL 2026 full-season custom-fantasy points for every squad player
using the trained XGBoost model (with Marcel fallback for rookies).

Steps:
  1. Re-score 2026 ball-by-ball under published rules (matches 13a scoring).
  2. Tag each 2026 match as is_early (both teams entering had <6 games).
  3. Build features per player:
      - career + lag + marcel + role_prior from pre-2026 historical data
      - early_* from early-2026 matches
  4. Predict full-season total via XGBoost; blend with observed 2026 YTD.
  5. Fallback to Marcel role-prior × 14 for rookies with no history and no
     2026 appearance.

Inputs:
  - data/historical_custom_points_per_match.parquet
  - data/ipl_json (4)/                                 (2026 cricsheet JSONs)
  - data/Boston IPL 26_Results (1).xlsx
  - data/ipl_2026_squads_enriched.csv
  - results/FantasyProjections/ipl2026_custom/xgb_model.json
  - results/FantasyProjections/ipl2026_custom/marcel_projections_2026.csv

Output:
  - results/FantasyProjections/ipl2026_custom/xgb_projections_2026.csv
"""
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'WARprojections'))
from utils import extract_player_id, get_wicket_info, get_extras_info, create_match_id  # noqa

HIST_PM = PROJECT_ROOT / 'data' / 'historical_custom_points_per_match.parquet'
JSON_DIR = PROJECT_ROOT / 'data' / 'ipl_json (4)'
BOSTON_FILE = PROJECT_ROOT / 'data' / 'Boston IPL 26_Results (1).xlsx'
SQUAD_FILE = PROJECT_ROOT / 'data' / 'ipl_2026_squads_enriched.csv'
MODEL_FILE = PROJECT_ROOT / 'results' / 'FantasyProjections' / 'ipl2026_custom' / 'xgb_model.json'
MARCEL_FILE = PROJECT_ROOT / 'results' / 'FantasyProjections' / 'ipl2026_custom' / 'marcel_projections_2026.csv'
OUT = PROJECT_ROOT / 'results' / 'FantasyProjections' / 'ipl2026_custom' / 'xgb_projections_2026.csv'

SEASON = 2026
GROUP_STAGE_GAMES = 14
EARLY_CUTOFF = 6
MARCEL_PRIOR_N = 100
W_S1, W_S2, W_S3 = 5, 4, 3

BOWLING_WKT = {'caught', 'bowled', 'lbw', 'stumped', 'caught and bowled', 'hit wicket'}

FEATURE_COLS = [
    'career_matches', 'career_ppm',
    'career_bat_ppm', 'career_bowl_ppm', 'career_field_ppm',
    'lag1_matches', 'lag1_ppm',
    'lag2_matches', 'lag2_ppm',
    'lag3_matches', 'lag3_ppm',
    'marcel_ppm', 'role_prior_ppm',
    'early_matches', 'early_ppm',
    'early_bat_ppm', 'early_bowl_ppm', 'early_field_ppm',
    'early_runs', 'early_wickets', 'early_overs',
    'role_BAT', 'role_BOWL', 'role_AR', 'role_WK',
]

MANUAL = {
    'Vaibhav Sooryavanshi': 'V Suryavanshi', 'Philip Salt': 'PD Salt', 'Jos Buttler': 'JC Buttler',
    'Yashasvi Jaiswal': 'YBK Jaiswal', 'Prabhsimran Singh': 'P Simran Singh', 'Sanju Samson': 'SV Samson',
    'Suryakumar Yadav': 'SA Yadav', 'Angkrish Raghuvanshi': 'A Raghuvanshi', 'Ayush Mhatre': 'A Mhatre',
    'Shivam Dube': 'SM Dube', 'Shubman Gill': 'Shubman Gill', 'Virat Kohli': 'V Kohli',
    'Ishan Kishan': 'Ishan Kishan', 'Heinrich Klaasen': 'H Klaasen', 'Rajat Patidar': 'RM Patidar',
    'Shreyas Iyer': 'SS Iyer', 'KL Rahul': 'KL Rahul', 'Travis Head': 'TM Head',
    'Jofra Archer': 'JC Archer', 'Arshdeep Singh': 'Arshdeep Singh', 'Noor Ahmad': 'Noor Ahmad',
    'Mohammed Siraj': 'Mohammed Siraj', 'Devdutt Padikkal': 'D Padikkal', 'Quinton de Kock': 'Q de Kock',
    'Hardik Pandya': 'HH Pandya', 'Jasprit Bumrah': 'JJ Bumrah', 'Rishabh Pant': 'RR Pant',
    'Abhishek Sharma': 'Abhishek Sharma', 'Ruturaj Gaikwad': 'RD Gaikwad',
    'Sai Sudharsan': 'B Sai Sudharsan', 'Rohit Sharma': 'RG Sharma', 'Nicholas Pooran': 'N Pooran',
    'Priyansh Arya': 'Priyansh Arya', 'Ajinkya Rahane': 'AM Rahane', 'Pathum Nissanka': 'WPN Nissanka',
    'Faf du Plessis': 'F du Plessis', 'Trent Boult': 'TA Boult', 'Ravindra Jadeja': 'RA Jadeja',
    'Mitchell Marsh': 'MR Marsh', 'Glenn Maxwell': 'GJ Maxwell', 'Sunil Narine': 'SP Narine',
    'Marcus Stoinis': 'MP Stoinis', 'Cameron Green': 'C Green', 'Washington Sundar': 'Washington Sundar',
    'Krunal Pandya': 'KH Pandya', 'Andre Russell': 'AD Russell', 'Yuzvendra Chahal': 'YS Chahal',
    'Prasidh Krishna': 'Prasidh Krishna', 'Harshal Patel': 'HV Patel', 'Mayank Yadav': 'Mayank Yadav',
    'Mohammed Shami': 'Mohammed Shami', 'Umran Malik': 'Umran Malik', 'Kuldeep Yadav': 'Kuldeep Yadav',
    'Axar Patel': 'AR Patel', 'Ravichandran Ashwin': 'R Ashwin', 'Mitchell Starc': 'MA Starc',
    'Liam Livingstone': 'LS Livingstone', 'Nitish Kumar Reddy': 'N Reddy', 'Dhruv Jurel': 'DC Jurel',
    'Riyan Parag': 'R Parag', 'Sandeep Sharma': 'Sandeep Sharma', 'Rahul Tripathi': 'RA Tripathi',
    'Rinku Singh': 'RK Singh', 'Venkatesh Iyer': 'VR Iyer', 'Tilak Varma': 'T Varma',
    'Rachin Ravindra': 'RR Ravindra', 'Marco Jansen': 'M Jansen', 'Rashid Khan': 'Rashid Khan',
    'Anrich Nortje': 'A Nortje', 'Kagiso Rabada': 'K Rabada', 'Nitish Rana': 'N Rana',
    'Deepak Chahar': 'DL Chahar', 'Matheesha Pathirana': 'M Pathirana', 'Tristan Stubbs': 'T Stubbs',
    'Lungi Ngidi': 'L Ngidi', 'Wanindu Hasaranga': 'PW Hasaranga', 'David Miller': 'DA Miller',
    'Moeen Ali': 'MM Ali',
}

IPL_TEAM_MAP = {
    'Bengaluru': 'Royal Challengers Bengaluru',
    'Chennai': 'Chennai Super Kings',
    'Delhi': 'Delhi Capitals',
    'Gujarat': 'Gujarat Titans',
    'Hyderabad': 'Sunrisers Hyderabad',
    'Kolkata': 'Kolkata Knight Riders',
    'Lucknow': 'Lucknow Super Giants',
    'Mumbai': 'Mumbai Indians',
    'Punjab': 'Punjab Kings',
    'Rajasthan': 'Rajasthan Royals',
    'RCB': 'Royal Challengers Bengaluru',
    'CSK': 'Chennai Super Kings',
    'DC': 'Delhi Capitals',
    'GT': 'Gujarat Titans',
    'SRH': 'Sunrisers Hyderabad',
    'KKR': 'Kolkata Knight Riders',
    'LSG': 'Lucknow Super Giants',
    'MI': 'Mumbai Indians',
    'PBKS': 'Punjab Kings',
    'RR': 'Rajasthan Royals',
}


# ── Scoring (same rules as 13a / published) ─────────────────────────────

def sr_pts(sr):
    if sr < 50: return -6
    if sr < 60: return -4
    if sr <= 70: return -2
    if sr <= 130: return 0
    if sr <= 150: return 2
    if sr <= 170: return 4
    return 6


def econ_pts(e):
    if e < 5: return 6
    if e < 6: return 4
    if e <= 7: return 2
    if e <= 10: return 0
    if e <= 11: return -2
    if e <= 12: return -4
    return -6


def milestone(r):
    p = 0
    if r >= 30: p += 4
    if r >= 50: p += 4
    if r >= 100: p += 8
    return p


def haul(w):
    if w >= 5: return 16
    if w >= 4: return 8
    if w >= 3: return 4
    return 0


def extract_2026_balls():
    rows = []
    for jf in JSON_DIR.glob('*.json'):
        try:
            with open(jf) as f:
                md = json.load(f)
        except Exception:
            continue
        info = md.get('info', {})
        if str(info.get('season')) != str(SEASON):
            continue
        registry = info.get('registry', {}).get('people', {})
        match_id = create_match_id(info)
        match_date = info.get('dates', [''])[0]
        teams = info.get('teams', [])
        for i_idx, innings in enumerate(md.get('innings', [])):
            batting_team = innings.get('team', '')
            bowling_team = (teams[1] if teams and teams[0] == batting_team else teams[0] if teams else '')
            for over_data in innings.get('overs', []):
                o = over_data.get('over', 0)
                for delivery in over_data.get('deliveries', []):
                    ri = delivery.get('runs', {})
                    ex = get_extras_info(delivery)
                    wk = get_wicket_info(delivery)
                    fielders = (wk.get('fielders', []) if wk else []) or []
                    f1 = fielders[0].get('name', '') if len(fielders) >= 1 else ''
                    f2 = fielders[1].get('name', '') if len(fielders) >= 2 else ''
                    rows.append({
                        'match_id': match_id,
                        'match_date': match_date,
                        'batting_team': batting_team,
                        'bowling_team': bowling_team,
                        'over': o,
                        'batter_name': delivery.get('batter', ''),
                        'batter_id': extract_player_id(delivery.get('batter', ''), registry) or '',
                        'bowler_name': delivery.get('bowler', ''),
                        'bowler_id': extract_player_id(delivery.get('bowler', ''), registry) or '',
                        'batter_runs': ri.get('batter', 0),
                        'total_runs': ri.get('total', 0),
                        'wides': ex['wides'], 'noballs': ex['noballs'],
                        'byes': ex['byes'], 'legbyes': ex['legbyes'],
                        'is_wicket': wk is not None,
                        'dismissal_type': wk.get('kind', '') if wk else '',
                        'dismissed_player': wk.get('player_out', '') if wk else '',
                        'fielder1_name': f1,
                        'fielder2_name': f2,
                    })
    return pd.DataFrame(rows)


def score_batting(df):
    legal = df[df['wides'] == 0].copy()
    bat = legal.groupby(['match_id', 'batter_name']).agg(
        runs=('batter_runs', 'sum'),
        balls_faced=('batter_runs', 'count'),
        fours=('batter_runs', lambda x: (x == 4).sum()),
        sixes=('batter_runs', lambda x: (x == 6).sum()),
    ).reset_index()
    dismissals = df[df['is_wicket']].groupby(
        ['match_id', 'dismissed_player']).size().reset_index(name='times_out')
    bat = bat.merge(dismissals, left_on=['match_id', 'batter_name'],
                    right_on=['match_id', 'dismissed_player'], how='left').drop(columns='dismissed_player')
    bat['times_out'] = bat['times_out'].fillna(0).astype(int)
    bat['is_duck'] = (bat['runs'] == 0) & (bat['times_out'] > 0)
    bat['bat_pts'] = (bat['runs'] + bat['fours'] + 2 * bat['sixes']
                     + bat['runs'].apply(milestone) + bat['is_duck'].astype(int) * -2)
    has_balls = bat['balls_faced'] > 0
    sr = bat.loc[has_balls, 'runs'] / bat.loc[has_balls, 'balls_faced'] * 100
    bat.loc[has_balls, 'bat_pts'] += sr.apply(sr_pts)
    return bat.rename(columns={'batter_name': 'player_name'})


def score_bowling(df):
    d = df.copy()
    d['runs_conceded_ball'] = d['total_runs'] - d['byes'] - d['legbyes']
    d['is_bowling_wicket'] = d['is_wicket'] & d['dismissal_type'].isin(BOWLING_WKT)
    d['is_legal'] = (d['wides'] == 0) & (d['noballs'] == 0)
    d['is_dot'] = d['is_legal'] & (d['total_runs'] == 0)
    d['is_wide_ball'] = d['wides'] > 0
    d['is_noball'] = d['noballs'] > 0
    g = d.groupby(['match_id', 'bowler_name']).agg(
        legal_balls=('is_legal', 'sum'),
        runs_conceded=('runs_conceded_ball', 'sum'),
        wickets=('is_bowling_wicket', 'sum'),
        dots=('is_dot', 'sum'),
        wides=('is_wide_ball', 'sum'),
        noballs=('is_noball', 'sum'),
    ).reset_index()
    g['overs'] = g['legal_balls'] / 6.0
    legal = d[d['is_legal']]
    per_over = legal.groupby(['match_id', 'bowler_name', 'over']).agg(
        balls=('runs_conceded_ball', 'count'),
        over_runs=('runs_conceded_ball', 'sum'),
    ).reset_index()
    per_over['is_maiden'] = (per_over['balls'] == 6) & (per_over['over_runs'] == 0)
    maidens = per_over[per_over['is_maiden']].groupby(
        ['match_id', 'bowler_name']).size().reset_index(name='maidens')
    g = g.merge(maidens, on=['match_id', 'bowler_name'], how='left')
    g['maidens'] = g['maidens'].fillna(0).astype(int)
    g['bowl_pts'] = (g['dots'] + g['wickets'] * 20 + g['wickets'].apply(haul)
                    + g['maidens'] * 12 - g['wides'] - g['noballs'] * 2)
    has_ov = g['overs'] > 0
    e = g.loc[has_ov, 'runs_conceded'] / g.loc[has_ov, 'overs']
    g.loc[has_ov, 'bowl_pts'] += e.apply(econ_pts)
    return g.rename(columns={'bowler_name': 'player_name'})


def score_fielding(df):
    w = df[df['is_wicket']].copy()
    recs = []
    for _, r in w.iterrows():
        mid, kind = r['match_id'], r['dismissal_type']
        f1, f2 = r['fielder1_name'], r['fielder2_name']
        if f1:
            if kind in ('caught', 'caught and bowled'):
                recs.append((mid, f1, 'catch'))
            elif kind == 'stumped':
                recs.append((mid, f1, 'stumping'))
            elif kind == 'run out':
                recs.append((mid, f1, 'runout'))
        if f2 and kind == 'run out':
            recs.append((mid, f2, 'runout'))
    if not recs:
        return pd.DataFrame(columns=['match_id', 'player_name', 'catches', 'stumpings', 'runouts', 'field_pts'])
    fd = pd.DataFrame(recs, columns=['match_id', 'player_name', 'act'])
    fg = fd.groupby(['match_id', 'player_name']).agg(
        catches=('act', lambda x: (x == 'catch').sum()),
        stumpings=('act', lambda x: (x == 'stumping').sum()),
        runouts=('act', lambda x: (x == 'runout').sum()),
    ).reset_index()
    fg['field_pts'] = (fg['catches'] * 8 + fg['stumpings'] * 6 + fg['runouts'] * 6
                      + (fg['catches'] >= 3).astype(int) * 4)
    return fg


def infer_role(bat, bowl, stump):
    if stump >= 3: return 'WK'
    total = bat + bowl
    if total <= 0: return 'AR'
    share = bat / total
    if bowl >= 50 and share > 0.6 and bat >= 100: return 'AR'
    if share >= 0.75: return 'BAT'
    if share <= 0.25: return 'BOWL'
    return 'AR'


def normalize_role(r):
    if pd.isna(r): return 'AR'
    r = str(r).upper().strip()
    if 'WK' in r or 'WICKET' in r or r == 'WICKETKEEPER': return 'WK'
    if r in ('BAT', 'BATTER', 'BATSMAN'): return 'BAT'
    if r in ('BOWL', 'BOWLER'): return 'BOWL'
    if r in ('AR', 'ALL-ROUNDER', 'ALLROUNDER', 'ALL ROUNDER'): return 'AR'
    return 'AR'


def main():
    print("Loading historical per-match...")
    pm_hist = pd.read_parquet(HIST_PM)
    print(f"  {len(pm_hist):,} rows")

    print("Extracting 2026 balls...")
    balls = extract_2026_balls()
    print(f"  {len(balls):,} balls from 2026")
    if balls.empty:
        print("No 2026 data found!")
        return

    # Score 2026 per-match
    bat = score_batting(balls)
    bowl = score_bowling(balls)
    field = score_fielding(balls)

    # Combine per (match_id, player)
    keys = ['match_id', 'player_name']
    union = pd.concat([bat[keys], bowl[keys], field[keys]], ignore_index=True).drop_duplicates()
    pm26 = union.merge(bat, on=keys, how='left') \
                .merge(bowl, on=keys, how='left') \
                .merge(field, on=keys, how='left')
    for c in ['runs', 'balls_faced', 'fours', 'sixes', 'bat_pts',
              'legal_balls', 'runs_conceded', 'wickets', 'dots',
              'wides', 'noballs', 'maidens', 'overs', 'bowl_pts',
              'catches', 'stumpings', 'runouts', 'field_pts']:
        if c in pm26.columns:
            pm26[c] = pm26[c].fillna(0)
    pm26['total_pts'] = pm26['bat_pts'] + pm26['bowl_pts'] + pm26['field_pts']

    # Tag early 2026 matches
    mt = balls.groupby(['match_id', 'match_date'])['batting_team'].unique().reset_index()
    mt['team1'] = mt['batting_team'].str[0]
    mt['team2'] = mt['batting_team'].str[1]
    mt = mt.drop(columns='batting_team').sort_values(['match_date', 'match_id']).reset_index(drop=True)
    team_count = {}
    is_early = []
    for _, r in mt.iterrows():
        t1, t2 = r['team1'], r['team2']
        c1, c2 = team_count.get(t1, 0), team_count.get(t2, 0)
        is_early.append((c1 < EARLY_CUTOFF) and (c2 < EARLY_CUTOFF))
        team_count[t1] = c1 + 1
        team_count[t2] = c2 + 1
    mt['is_early'] = is_early
    early_ids = set(mt.loc[mt['is_early'], 'match_id'])
    team_games = team_count  # final per-team played counts
    print(f"  {len(early_ids)} early-2026 matches (out of {len(mt)} total)")
    print(f"  team games played: {team_games}")

    # ── Feature build ────────────────────────────────────────────────────

    # Career pre-2026 aggregates
    career = pm_hist.groupby(['player_name']).agg(
        career_matches=('match_id', 'nunique'),
        career_total=('total_pts', 'sum'),
        career_bat=('bat_pts', 'sum'),
        career_bowl=('bowl_pts', 'sum'),
        career_field=('field_pts', 'sum'),
        career_stumpings=('stumpings', 'sum'),
    ).reset_index()
    career['career_ppm'] = career['career_total'] / career['career_matches'].clip(lower=1)
    career['career_bat_ppm'] = career['career_bat'] / career['career_matches'].clip(lower=1)
    career['career_bowl_ppm'] = career['career_bowl'] / career['career_matches'].clip(lower=1)
    career['career_field_ppm'] = career['career_field'] / career['career_matches'].clip(lower=1)
    career['hist_role'] = career.apply(
        lambda r: infer_role(r['career_bat'], r['career_bowl'], r['career_stumpings']), axis=1
    )

    # Lag features (2025, 2024, 2023)
    def lag(s, n):
        d = pm_hist[pm_hist['season'] == s].groupby('player_name').agg(
            matches=('match_id', 'nunique'),
            total=('total_pts', 'sum'),
        ).reset_index()
        d['ppm'] = d['total'] / d['matches'].clip(lower=1)
        return d.rename(columns={'matches': f'lag{n}_matches', 'ppm': f'lag{n}_ppm'})[
            ['player_name', f'lag{n}_matches', f'lag{n}_ppm']
        ]

    lag1 = lag(2025, 1)
    lag2 = lag(2024, 2)
    lag3 = lag(2023, 3)

    # Role prior PPM from pre-2026 career, weighted by matches
    role_prior = career.groupby('hist_role').apply(
        lambda g: g['career_total'].sum() / g['career_matches'].sum(), include_groups=False
    ).to_dict()
    default_prior = career['career_total'].sum() / career['career_matches'].sum()
    print(f"  role priors: {role_prior}")

    # Early-2026 aggregates
    early_pm = pm26[pm26['match_id'].isin(early_ids)]
    early = early_pm.groupby('player_name').agg(
        early_matches=('match_id', 'nunique'),
        early_total=('total_pts', 'sum'),
        early_bat=('bat_pts', 'sum'),
        early_bowl=('bowl_pts', 'sum'),
        early_field=('field_pts', 'sum'),
        early_runs=('runs', 'sum'),
        early_wickets=('wickets', 'sum'),
        early_overs=('overs', 'sum'),
    ).reset_index()
    early['early_ppm'] = early['early_total'] / early['early_matches'].clip(lower=1)
    early['early_bat_ppm'] = early['early_bat'] / early['early_matches'].clip(lower=1)
    early['early_bowl_ppm'] = early['early_bowl'] / early['early_matches'].clip(lower=1)
    early['early_field_ppm'] = early['early_field'] / early['early_matches'].clip(lower=1)

    # Full 2026 YTD (all games so far — to compute observed "floor" and games-left)
    ytd = pm26.groupby('player_name').agg(
        ytd_matches=('match_id', 'nunique'),
        ytd_total=('total_pts', 'sum'),
    ).reset_index()

    # ── Roster: drafted + unsold + any cricsheet 2026 player ─────────────
    boston = pd.read_excel(BOSTON_FILE, sheet_name=None)
    rows = []
    owners = [k for k in boston.keys() if k.lower() != 'unsold' and k.lower() != 'all']
    # Try to detect the unsold sheet
    unsold_key = next((k for k in boston.keys() if 'unsold' in k.lower()), None)
    for sheet_name, bdf in boston.items():
        if bdf is None or bdf.empty: continue
        cols = {c.lower().strip(): c for c in bdf.columns}
        name_col = cols.get('player') or cols.get('name') or list(bdf.columns)[0]
        role_col = cols.get('role')
        team_col = cols.get('team')
        price_col = cols.get('price (₹l)') or cols.get('price') or cols.get('price (l)')
        pts_col = cols.get('points') or cols.get('pts')
        is_unsold = 'unsold' in sheet_name.lower()
        owner = 'Unsold' if is_unsold else sheet_name
        for _, r in bdf.iterrows():
            pname = r[name_col]
            if not isinstance(pname, str) or not pname.strip():
                continue
            upper = pname.upper()
            if any(tag in upper for tag in ('TOTAL', 'PURSE', 'REMAINING', 'BOUGHT', 'SPENT', 'RATING', 'SQUAD')):
                continue
            rows.append({
                'player_name_boston': pname.strip(),
                'owner': owner,
                'role_boston': r[role_col] if role_col else None,
                'ipl_team': r[team_col] if team_col else None,
                'price_L': r[price_col] if price_col else None,
                'app_points': r[pts_col] if pts_col else None,
            })
    boston_df = pd.DataFrame(rows).drop_duplicates('player_name_boston')
    print(f"  {len(boston_df)} Boston rows (drafted + unsold)")

    # Squad file adds any non-Boston player
    squad = pd.read_csv(SQUAD_FILE)
    squad_cols = {c.lower().strip(): c for c in squad.columns}
    sname = squad_cols.get('player') or squad_cols.get('player_name') or squad_cols.get('name')
    srole = squad_cols.get('playing_role') or squad_cols.get('role')
    steam = squad_cols.get('team')
    squad_rows = []
    for _, r in squad.iterrows():
        pn = r[sname] if sname else None
        if not isinstance(pn, str): continue
        squad_rows.append({
            'player_name_squad': pn.strip(),
            'role_squad': r[srole] if srole else None,
            'ipl_team_squad': r[steam] if steam else None,
        })
    squad_df = pd.DataFrame(squad_rows).drop_duplicates('player_name_squad')

    # ── Map boston/squad names → cricsheet names ─────────────────────────
    known_names = set(career['player_name']).union(set(early['player_name']))

    def resolve(name: str, fallback_hits: set) -> str:
        if name in MANUAL and MANUAL[name] in fallback_hits:
            return MANUAL[name]
        if name in fallback_hits:
            return name
        lower = name.lower().replace(' ', '')
        # Try "X Lastname" -> "X.Lastname"
        parts = name.split()
        if len(parts) >= 2:
            alt = f"{parts[0][0]} {' '.join(parts[1:])}"
            if alt in fallback_hits:
                return alt
        # Fuzzy last-name match
        lastname = parts[-1].lower() if parts else ''
        cand = [n for n in fallback_hits if n.lower().endswith(lastname)]
        if len(cand) == 1:
            return cand[0]
        return None

    # Build unified player frame: every Boston + squad (IPL 2026) player only.
    # Cricsheet names are used as feature-lookup aliases, not separate entities.
    all_boston = set(boston_df['player_name_boston'])
    all_squad = set(squad_df['player_name_squad'])
    all_players = sorted(all_boston.union(all_squad))

    master = []
    for p in all_players:
        # Determine cricsheet alias: used for feature lookup
        alias = resolve(p, known_names) if p not in known_names else p
        bdisplay = p
        boston_row = boston_df[boston_df['player_name_boston'] == p]
        squad_row = squad_df[squad_df['player_name_squad'] == p]
        owner = boston_row['owner'].iloc[0] if len(boston_row) else None
        role_str = None
        ipl_team = None
        price = None
        app_pts = None
        if len(boston_row):
            role_str = boston_row['role_boston'].iloc[0]
            ipl_team = boston_row['ipl_team'].iloc[0]
            price = boston_row['price_L'].iloc[0]
            app_pts = boston_row['app_points'].iloc[0]
        if not role_str and len(squad_row):
            role_str = squad_row['role_squad'].iloc[0]
        if not ipl_team and len(squad_row):
            ipl_team = squad_row['ipl_team_squad'].iloc[0]
        master.append({
            'player_name': p,
            'cricsheet_name': alias,
            'owner': owner,
            'ipl_team': ipl_team,
            'role': normalize_role(role_str),
            'price_L': price,
            'app_points': app_pts,
        })
    master_df = pd.DataFrame(master)
    # Dedupe: if same cricsheet_name appears for multiple Boston/squad names,
    # keep the Boston-owner row first. Rows with no cricsheet alias (rookies)
    # dedupe by player_name — they all stay unique.
    master_df['__has_owner'] = master_df['owner'].notna().astype(int)
    master_df['__dedup_key'] = master_df['cricsheet_name'].fillna(
        '__rookie__' + master_df['player_name']
    )
    master_df = (master_df.sort_values('__has_owner', ascending=False)
                 .drop_duplicates(subset=['__dedup_key'], keep='first')
                 .drop(columns=['__has_owner', '__dedup_key']))
    print(f"  {len(master_df)} unique players to project")

    # Merge features onto master_df using cricsheet_name
    def merge(left, right):
        right = right.rename(columns={'player_name': 'cricsheet_name'})
        return left.merge(right, on='cricsheet_name', how='left')

    df = master_df
    df = merge(df, career[['player_name', 'career_matches', 'career_total', 'career_ppm',
                           'career_bat_ppm', 'career_bowl_ppm', 'career_field_ppm', 'hist_role']])
    df = merge(df, lag1)
    df = merge(df, lag2)
    df = merge(df, lag3)
    df = merge(df, early[['player_name', 'early_matches', 'early_ppm',
                          'early_bat_ppm', 'early_bowl_ppm', 'early_field_ppm',
                          'early_runs', 'early_wickets', 'early_overs']])
    df = merge(df, ytd)

    # Fill
    num_zero = ['career_matches', 'career_total', 'career_ppm',
                'career_bat_ppm', 'career_bowl_ppm', 'career_field_ppm',
                'lag1_matches', 'lag1_ppm', 'lag2_matches', 'lag2_ppm',
                'lag3_matches', 'lag3_ppm',
                'early_matches', 'early_ppm',
                'early_bat_ppm', 'early_bowl_ppm', 'early_field_ppm',
                'early_runs', 'early_wickets', 'early_overs',
                'ytd_matches', 'ytd_total']
    for c in num_zero:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # Role: prefer Boston role; fallback to hist_role; else AR
    df['role'] = df['role'].fillna('AR')
    df.loc[df['role'] == 'AR', 'role'] = df.loc[df['role'] == 'AR', 'hist_role'].fillna('AR')
    df['role'] = df['role'].fillna('AR')

    df['role_prior_ppm'] = df['role'].map(role_prior).fillna(default_prior)

    # Marcel ppm
    num = (W_S1 * df['lag1_ppm'] * df['lag1_matches']
           + W_S2 * df['lag2_ppm'] * df['lag2_matches']
           + W_S3 * df['lag3_ppm'] * df['lag3_matches']
           + MARCEL_PRIOR_N * df['role_prior_ppm'])
    den = (W_S1 * df['lag1_matches'] + W_S2 * df['lag2_matches']
           + W_S3 * df['lag3_matches'] + MARCEL_PRIOR_N)
    df['marcel_ppm'] = num / den

    # One-hot role
    for r in ['BAT', 'BOWL', 'AR', 'WK']:
        df[f'role_{r}'] = (df['role'] == r).astype(int)

    # Predict
    print("\nLoading XGBoost model...")
    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_FILE))
    X = df[FEATURE_COLS].values
    df['xgb_full_season'] = np.clip(model.predict(X), 0, None).astype(float)

    # Blend: XGB projects full season. Observed YTD is ground truth for games
    # already played by that player. Fill rest via rate implied by XGB.
    # full_projected = ytd_total + xgb_remaining_rate * team_games_left
    # Map Boston team to cricsheet team short
    def team_games_remaining(ipl_team):
        if not isinstance(ipl_team, str): return GROUP_STAGE_GAMES
        key = IPL_TEAM_MAP.get(ipl_team, None)
        played = team_games.get(key, 0)
        return max(GROUP_STAGE_GAMES - played, 0)

    df['team_games_left'] = df['ipl_team'].apply(team_games_remaining)
    df['team_games_played'] = GROUP_STAGE_GAMES - df['team_games_left']

    # Implied remaining points: xgb predicted full - ytd observed, floored at 0
    df['xgb_remaining'] = np.clip(df['xgb_full_season'] - df['ytd_total'], 0, None)
    # If player has no historical footprint AND no 2026 appearance (true rookie)
    # use role_prior_ppm × GROUP_STAGE_GAMES as a simple Marcel fallback
    is_rookie = (df['career_matches'] == 0) & (df['ytd_matches'] == 0) & (df['early_matches'] == 0)
    df.loc[is_rookie, 'xgb_full_season'] = df.loc[is_rookie, 'role_prior_ppm'] * GROUP_STAGE_GAMES * 0.5
    df.loc[is_rookie, 'xgb_remaining'] = df.loc[is_rookie, 'xgb_full_season']
    df['final_projection'] = df['ytd_total'] + df['xgb_remaining']
    df['is_rookie'] = is_rookie

    # Final output columns
    out = df[['player_name', 'cricsheet_name', 'owner', 'ipl_team', 'role',
              'price_L', 'app_points',
              'ytd_matches', 'ytd_total',
              'team_games_played', 'team_games_left',
              'career_matches', 'career_ppm',
              'lag1_matches', 'lag1_ppm',
              'early_matches', 'early_ppm',
              'marcel_ppm', 'role_prior_ppm',
              'xgb_full_season', 'xgb_remaining',
              'is_rookie', 'final_projection']].copy()
    # Round numeric columns
    for c in ['career_ppm', 'lag1_ppm', 'early_ppm', 'marcel_ppm', 'role_prior_ppm',
              'xgb_full_season', 'xgb_remaining', 'final_projection']:
        out[c] = out[c].round(1)
    out = out.sort_values('final_projection', ascending=False).reset_index(drop=True)
    out.insert(0, 'rank', out.index + 1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\n✓ {OUT}  ({len(out)} players)")
    print(f"  Rookies (pure Marcel): {int(out['is_rookie'].sum())}")

    print("\n── Top 25 projections ──")
    display = out.head(25)[['rank', 'player_name', 'owner', 'ipl_team', 'role',
                            'ytd_total', 'xgb_full_season', 'final_projection']]
    print(display.to_string(index=False))

    # Compare Boston-sold mean projection per owner
    sold = out[out['owner'].notna() & (out['owner'] != 'Unsold')]
    if len(sold):
        print("\n── Team strength (sum of final projection per owner) ──")
        print(sold.groupby('owner').agg(
            n=('player_name', 'count'),
            total_proj=('final_projection', 'sum'),
            mean_proj=('final_projection', 'mean'),
        ).sort_values('total_proj', ascending=False).round(1).to_string())


if __name__ == '__main__':
    main()
