"""
Compute CUSTOM fantasy points for IPL 2026 mid-season (24 matches played).

Uses the user-provided scoring system (different from Dream11 —
milestones +4/+4/+8, wicket +20, dots +1, wides -1, noballs -2,
run out / stumping +6, no playing-XI bonus).

Produces 4 variants so the user can spot-check which rules their
fantasy app applies:
  V_BASE    : cumulative milestones + 10-ball SR min + 2-over econ min
  V_HIGHEST : only-highest milestone (50 = +4 not +8, 100 = +8 not +16)
  V_NO_SR   : no SR min-balls qualification
  V_NO_ECON : no economy min-overs qualification

Outputs (results/FantasyProjections/ipl2026_custom/):
  - season_to_date_totals.csv  (per player, all 4 variant totals)
  - per_match_points.csv       (per player per match, baseline)
  - variant_diffs.csv          (players whose total differs across variants)
"""

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / 'WARprojections'))
from utils import extract_player_id, get_wicket_info, get_extras_info, create_match_id, save_dataframe

# ── Input ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
CRICSHEET_DIR = PROJECT_ROOT / 'data' / 'ipl_json (4)'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'FantasyProjections' / 'ipl2026_custom'
SEASON_TARGET = 2026

# ── Scoring constants ────────────────────────────────────────────────────
PTS_PER_RUN = 1
FOUR_BONUS = 1        # extra for hitting a 4
SIX_BONUS = 2         # extra for hitting a 6
DUCK_PENALTY = -2     # applied to anyone who gets out for 0

PTS_DOT = 1
WICKET_PTS = 20
MAIDEN_PTS = 12
WIDE_PENALTY = -1
NOBALL_PENALTY = -2
HAUL_3 = 4
HAUL_4 = 8
HAUL_5 = 16

CATCH_PTS = 8
THREE_CATCH_BONUS = 4
RUNOUT_PTS = 6
STUMPING_PTS = 6

SR_MIN_BALLS = 10     # applied only in variants with SR qualification
ECON_MIN_OVERS = 2.0  # applied only in variants with econ qualification

# SR bands — inclusive on both ends using half-open logic
def sr_pts(sr: float) -> int:
    if sr < 50: return -6
    if sr < 60: return -4
    if sr <= 70: return -2
    if sr <= 130: return 0
    if sr <= 150: return 2
    if sr <= 170: return 4
    return 6

def econ_pts(econ: float) -> int:
    if econ < 5: return 6
    if econ < 6: return 4
    if econ <= 7: return 2
    if econ <= 10: return 0
    if econ <= 11: return -2
    if econ <= 12: return -4
    return -6

def milestone_cumulative(runs: int) -> int:
    pts = 0
    if runs >= 30: pts += 4
    if runs >= 50: pts += 4
    if runs >= 100: pts += 8
    return pts

def milestone_highest(runs: int) -> int:
    if runs >= 100: return 8
    if runs >= 50: return 4
    if runs >= 30: return 4
    return 0

def haul(wickets: int) -> int:
    if wickets >= 5: return HAUL_5
    if wickets >= 4: return HAUL_4
    if wickets >= 3: return HAUL_3
    return 0


# ── Ball-by-ball extraction (season 2026 only) ───────────────────────────

def extract_match_balls(match_data: dict) -> list:
    info = match_data.get('info', {})
    if info.get('season') != SEASON_TARGET and str(info.get('season')) != str(SEASON_TARGET):
        return []

    innings_list = match_data.get('innings', [])
    registry = info.get('registry', {}).get('people', {}) or info.get('registry', {})
    match_id = create_match_id(info)
    match_date = info.get('dates', [''])[0]
    teams = info.get('teams', [])

    balls = []
    for innings_idx, innings in enumerate(innings_list):
        innings_number = innings_idx + 1
        batting_team = innings.get('team', '')
        bowling_team = (teams[1] if len(teams) > 1 and teams[0] == batting_team
                        else teams[0] if len(teams) > 1 else '')

        for over_data in innings.get('overs', []):
            over_number = over_data.get('over', 0)
            for ball_number, delivery in enumerate(over_data.get('deliveries', [])):
                runs_info = delivery.get('runs', {})
                extras_info = get_extras_info(delivery)
                wicket = get_wicket_info(delivery)

                fielders = (wicket.get('fielders', []) if wicket else []) or []
                f1 = fielders[0].get('name', '') if len(fielders) >= 1 else ''
                f2 = fielders[1].get('name', '') if len(fielders) >= 2 else ''
                f1_id = extract_player_id(f1, registry) or '' if f1 else ''
                f2_id = extract_player_id(f2, registry) or '' if f2 else ''

                balls.append({
                    'match_id': match_id,
                    'match_date': match_date,
                    'innings': innings_number,
                    'batting_team': batting_team,
                    'bowling_team': bowling_team,
                    'over': over_number,
                    'ball_in_over': ball_number,
                    'batter_name': delivery.get('batter', ''),
                    'batter_id': extract_player_id(delivery.get('batter', ''), registry) or '',
                    'bowler_name': delivery.get('bowler', ''),
                    'bowler_id': extract_player_id(delivery.get('bowler', ''), registry) or '',
                    'batter_runs': runs_info.get('batter', 0),
                    'extras': runs_info.get('extras', 0),
                    'total_runs': runs_info.get('total', 0),
                    'wides': extras_info['wides'],
                    'noballs': extras_info['noballs'],
                    'byes': extras_info['byes'],
                    'legbyes': extras_info['legbyes'],
                    'is_wicket': wicket is not None,
                    'dismissal_type': wicket.get('kind', '') if wicket else '',
                    'dismissed_player': wicket.get('player_out', '') if wicket else '',
                    'fielder1_name': f1, 'fielder1_id': f1_id,
                    'fielder2_name': f2, 'fielder2_id': f2_id,
                })
    return balls


def load_all_2026_balls() -> pd.DataFrame:
    json_files = list(CRICSHEET_DIR.glob('*.json'))
    all_balls, n_matches = [], 0
    for jf in tqdm(json_files, desc="Scanning matches"):
        try:
            with open(jf) as f: md = json.load(f)
        except Exception:
            continue
        balls = extract_match_balls(md)
        if balls:
            all_balls.extend(balls)
            n_matches += 1
    print(f"✓ {n_matches} IPL 2026 matches, {len(all_balls):,} balls")
    return pd.DataFrame(all_balls)


# ── Per-match, per-player stats ──────────────────────────────────────────

def batting_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per batter × match: runs, balls faced, 4s, 6s, duck flag."""
    legal = df[df['wides'] == 0].copy()  # wides don't count as balls faced
    bat = legal.groupby(['match_id', 'batter_name', 'batter_id']).agg(
        runs=('batter_runs', 'sum'),
        balls_faced=('batter_runs', 'count'),
        fours=('batter_runs', lambda x: (x == 4).sum()),
        sixes=('batter_runs', lambda x: (x == 6).sum()),
    ).reset_index()

    dismissals = df[df['is_wicket']].groupby(
        ['match_id', 'dismissed_player']).size().reset_index(name='times_out')
    dismissals = dismissals.rename(columns={'dismissed_player': 'batter_name'})
    bat = bat.merge(dismissals, on=['match_id', 'batter_name'], how='left')
    bat['times_out'] = bat['times_out'].fillna(0).astype(int)
    bat['is_duck'] = (bat['runs'] == 0) & (bat['times_out'] > 0)
    return bat.rename(columns={'batter_name': 'player_name', 'batter_id': 'player_id'})


def bowling_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per bowler × match: legal balls, runs conceded, wickets, maidens, dots, wides, noballs."""
    d = df.copy()
    d['runs_conceded_ball'] = d['total_runs'] - d['byes'] - d['legbyes']
    # Bowler credited wickets (exclude run out, retired, etc.)
    bowling_dismissals = {'caught', 'bowled', 'lbw', 'stumped', 'caught and bowled', 'hit wicket'}
    d['is_bowling_wicket'] = d['is_wicket'] & d['dismissal_type'].isin(bowling_dismissals)

    d['is_legal'] = (d['wides'] == 0) & (d['noballs'] == 0)
    d['is_dot'] = d['is_legal'] & (d['total_runs'] == 0)
    d['is_wide_ball'] = d['wides'] > 0
    d['is_noball'] = d['noballs'] > 0

    g = d.groupby(['match_id', 'bowler_name', 'bowler_id']).agg(
        legal_balls=('is_legal', 'sum'),
        runs_conceded=('runs_conceded_ball', 'sum'),
        wickets=('is_bowling_wicket', 'sum'),
        dots=('is_dot', 'sum'),
        wides=('is_wide_ball', 'sum'),
        noballs=('is_noball', 'sum'),
    ).reset_index()
    g['overs'] = g['legal_balls'] / 6.0

    # Maidens — count overs with 6 legal balls + 0 runs conceded
    legal_only = d[d['is_legal']]
    per_over = legal_only.groupby(['match_id', 'bowler_name', 'over']).agg(
        balls=('runs_conceded_ball', 'count'),
        over_runs=('runs_conceded_ball', 'sum'),
    ).reset_index()
    per_over['is_maiden'] = (per_over['balls'] == 6) & (per_over['over_runs'] == 0)
    maidens = per_over[per_over['is_maiden']].groupby(
        ['match_id', 'bowler_name']).size().reset_index(name='maidens')
    g = g.merge(maidens, on=['match_id', 'bowler_name'], how='left')
    g['maidens'] = g['maidens'].fillna(0).astype(int)

    return g.rename(columns={'bowler_name': 'player_name', 'bowler_id': 'player_id'})


def fielding_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per fielder × match: catches, stumpings, run-outs."""
    w = df[df['is_wicket']].copy()
    recs = []

    for _, r in w[w['fielder1_name'] != ''].iterrows():
        kind = r['dismissal_type']
        if kind in ('caught', 'caught and bowled'):
            recs.append({'match_id': r['match_id'], 'player_name': r['fielder1_name'],
                         'player_id': r['fielder1_id'], 'act': 'catch'})
        elif kind == 'stumped':
            recs.append({'match_id': r['match_id'], 'player_name': r['fielder1_name'],
                         'player_id': r['fielder1_id'], 'act': 'stumping'})
        elif kind == 'run out':
            recs.append({'match_id': r['match_id'], 'player_name': r['fielder1_name'],
                         'player_id': r['fielder1_id'], 'act': 'runout'})

    for _, r in w[w['fielder2_name'] != ''].iterrows():
        if r['dismissal_type'] == 'run out':
            recs.append({'match_id': r['match_id'], 'player_name': r['fielder2_name'],
                         'player_id': r['fielder2_id'], 'act': 'runout'})

    if not recs:
        return pd.DataFrame(columns=['match_id', 'player_name', 'player_id',
                                     'catches', 'stumpings', 'runouts'])
    fd = pd.DataFrame(recs)
    return fd.groupby(['match_id', 'player_name', 'player_id']).agg(
        catches=('act', lambda x: (x == 'catch').sum()),
        stumpings=('act', lambda x: (x == 'stumping').sum()),
        runouts=('act', lambda x: (x == 'runout').sum()),
    ).reset_index()


# ── Scoring per variant ───────────────────────────────────────────────────

def score_player_match(row, *, milestone_fn, sr_min, econ_min):
    """Score one player-match under a given variant."""
    # Batting
    runs = int(row.get('runs', 0) or 0)
    balls = int(row.get('balls_faced', 0) or 0)
    fours = int(row.get('fours', 0) or 0)
    sixes = int(row.get('sixes', 0) or 0)
    is_duck = bool(row.get('is_duck', False))

    bat_pts = 0
    if balls > 0 or is_duck:
        bat_pts += runs * PTS_PER_RUN
        bat_pts += fours * FOUR_BONUS
        bat_pts += sixes * SIX_BONUS
        bat_pts += milestone_fn(runs)
        if is_duck:
            bat_pts += DUCK_PENALTY
        if balls >= sr_min and balls > 0:
            sr = runs / balls * 100.0
            bat_pts += sr_pts(sr)

    # Bowling
    overs = float(row.get('overs', 0) or 0)
    legal_balls = int(row.get('legal_balls', 0) or 0)
    runs_conceded = int(row.get('runs_conceded', 0) or 0)
    wickets = int(row.get('wickets', 0) or 0)
    dots = int(row.get('dots', 0) or 0)
    wides = int(row.get('wides', 0) or 0)
    noballs = int(row.get('noballs', 0) or 0)
    maidens = int(row.get('maidens', 0) or 0)

    bowl_pts = 0
    if legal_balls > 0 or wides > 0 or noballs > 0:
        bowl_pts += dots * PTS_DOT
        bowl_pts += wickets * WICKET_PTS
        bowl_pts += maidens * MAIDEN_PTS
        bowl_pts += wides * WIDE_PENALTY
        bowl_pts += noballs * NOBALL_PENALTY
        bowl_pts += haul(wickets)
        if overs >= econ_min:
            econ = runs_conceded / overs
            bowl_pts += econ_pts(econ)

    # Fielding
    catches = int(row.get('catches', 0) or 0)
    stumpings = int(row.get('stumpings', 0) or 0)
    runouts = int(row.get('runouts', 0) or 0)
    field_pts = (catches * CATCH_PTS + stumpings * STUMPING_PTS + runouts * RUNOUT_PTS
                 + (THREE_CATCH_BONUS if catches >= 3 else 0))

    return bat_pts + bowl_pts + field_pts, bat_pts, bowl_pts, field_pts


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load & extract
    balls = load_all_2026_balls()
    if balls.empty:
        print("No 2026 matches found!"); return
    n_matches = balls['match_id'].nunique()

    # 2. Per-match stats
    print("\nAggregating per-player per-match stats...")
    bat = batting_stats(balls)
    bowl = bowling_stats(balls)
    field = fielding_stats(balls)

    # 3. Combine by (match, player)
    keys = ['match_id', 'player_name', 'player_id']
    combined = (bat[keys].drop_duplicates()
                .merge(bowl[keys].drop_duplicates(), on=keys, how='outer')
                .merge(field[keys].drop_duplicates(), on=keys, how='outer'))
    combined = combined.merge(bat, on=keys, how='left')
    combined = combined.merge(bowl, on=keys, how='left')
    combined = combined.merge(field, on=keys, how='left')
    for c in ['runs','balls_faced','fours','sixes','times_out',
              'legal_balls','runs_conceded','wickets','dots','wides','noballs','maidens','overs',
              'catches','stumpings','runouts']:
        if c in combined.columns:
            combined[c] = combined[c].fillna(0)
    combined['is_duck'] = combined['is_duck'].fillna(False)

    # 4. Score under each variant
    variants = {
        'V_BASE':    dict(milestone_fn=milestone_cumulative, sr_min=SR_MIN_BALLS, econ_min=ECON_MIN_OVERS),
        'V_HIGHEST': dict(milestone_fn=milestone_highest,    sr_min=SR_MIN_BALLS, econ_min=ECON_MIN_OVERS),
        'V_NO_SR':   dict(milestone_fn=milestone_cumulative, sr_min=0,             econ_min=ECON_MIN_OVERS),
        'V_NO_ECON': dict(milestone_fn=milestone_cumulative, sr_min=SR_MIN_BALLS, econ_min=0.0),
    }

    print("Scoring under variants:", list(variants))
    for vname, cfg in variants.items():
        tot, bat_p, bowl_p, field_p = [], [], [], []
        for _, r in combined.iterrows():
            t, b, bw, f = score_player_match(r, **cfg)
            tot.append(t); bat_p.append(b); bowl_p.append(bw); field_p.append(f)
        combined[f'{vname}_total']    = tot
        combined[f'{vname}_bat']      = bat_p
        combined[f'{vname}_bowl']     = bowl_p
        combined[f'{vname}_field']    = field_p

    # 5. Per-match output (baseline)
    per_match_cols = (keys + ['runs','balls_faced','fours','sixes','is_duck',
                              'legal_balls','overs','runs_conceded','wickets','dots','wides','noballs','maidens',
                              'catches','stumpings','runouts']
                      + [f'V_BASE_{s}' for s in ('bat','bowl','field','total')])
    combined[per_match_cols].sort_values(['match_id','V_BASE_total'], ascending=[True, False]) \
        .to_csv(OUTPUT_DIR / 'per_match_points.csv', index=False)
    print(f"✓ per_match_points.csv  ({len(combined):,} rows)")

    # 6. Season totals per player — one row per player with all variants
    season = combined.groupby(['player_name', 'player_id']).agg(
        matches=('match_id', 'nunique'),
        runs=('runs','sum'),
        balls_faced=('balls_faced','sum'),
        fours=('fours','sum'),
        sixes=('sixes','sum'),
        wickets=('wickets','sum'),
        maidens=('maidens','sum'),
        dots=('dots','sum'),
        wides=('wides','sum'),
        noballs=('noballs','sum'),
        overs=('overs','sum'),
        runs_conceded=('runs_conceded','sum'),
        catches=('catches','sum'),
        stumpings=('stumpings','sum'),
        runouts=('runouts','sum'),
        V_BASE_total=('V_BASE_total','sum'),
        V_HIGHEST_total=('V_HIGHEST_total','sum'),
        V_NO_SR_total=('V_NO_SR_total','sum'),
        V_NO_ECON_total=('V_NO_ECON_total','sum'),
    ).reset_index()

    # 7. Variant diffs — any player whose total changes between variants
    season['diff_milestone']  = season['V_HIGHEST_total'] - season['V_BASE_total']
    season['diff_sr_rule']    = season['V_NO_SR_total']  - season['V_BASE_total']
    season['diff_econ_rule']  = season['V_NO_ECON_total'] - season['V_BASE_total']

    season = season.sort_values('V_BASE_total', ascending=False)
    season.to_csv(OUTPUT_DIR / 'season_to_date_totals.csv', index=False)
    print(f"✓ season_to_date_totals.csv ({len(season):,} rows)")

    # 8. Print summary + spot-check lists
    print(f"\n{'='*70}\nSEASON TO DATE: {n_matches} matches\n{'='*70}")

    print(f"\n── TOP 20 PLAYERS (V_BASE totals) ──")
    for _, r in season.head(20).iterrows():
        print(f"  {r['player_name']:28s}  {r['V_BASE_total']:6.0f} pts  "
              f"({int(r['matches'])} m)  "
              f"[hi:{r['V_HIGHEST_total']:5.0f} noSR:{r['V_NO_SR_total']:5.0f} noEcon:{r['V_NO_ECON_total']:5.0f}]")

    def _show_diffs(label, col):
        diffs = season[season[col] != 0].copy()
        diffs['abs_diff'] = diffs[col].abs()
        diffs = diffs.sort_values('abs_diff', ascending=False).head(20)
        print(f"\n── PLAYERS AFFECTED BY {label} ({len(season[season[col]!=0])} total) ──")
        if diffs.empty:
            print("   (none)")
        for _, r in diffs.iterrows():
            print(f"  {r['player_name']:28s}  V_BASE={r['V_BASE_total']:5.0f}  "
                  f"variant={r['V_BASE_total']+r[col]:5.0f}  Δ={r[col]:+.0f}")

    _show_diffs('MILESTONE RULE (cumulative → highest)', 'diff_milestone')
    _show_diffs('SR QUALIFICATION (drop 10-ball min)',     'diff_sr_rule')
    _show_diffs('ECON QUALIFICATION (drop 2-over min)',    'diff_econ_rule')

    # Variant diff CSV
    diff_out = season[['player_name', 'player_id', 'matches', 'V_BASE_total',
                       'V_HIGHEST_total', 'diff_milestone',
                       'V_NO_SR_total', 'diff_sr_rule',
                       'V_NO_ECON_total', 'diff_econ_rule']]
    diff_out = diff_out[(diff_out['diff_milestone']!=0) | (diff_out['diff_sr_rule']!=0) | (diff_out['diff_econ_rule']!=0)]
    diff_out.to_csv(OUTPUT_DIR / 'variant_diffs.csv', index=False)
    print(f"\n✓ variant_diffs.csv ({len(diff_out)} players differ between variants)")

    print(f"\nAll outputs in: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
