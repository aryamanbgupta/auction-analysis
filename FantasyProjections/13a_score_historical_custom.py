"""
Score all historical IPL seasons (2007-2025) under the published custom rules
used by the Boston IPL 26 fantasy league.

Source: data/ipl_matches_fantasy.parquet (278K balls, fielder-enriched).
Outputs:
  - data/historical_custom_points_per_match.parquet (one row per player-match)
  - data/historical_custom_season_totals.csv       (one row per player-season)

Scoring mirrors 10_ipl2026_custom_fantasy.py but drops the SR min-balls and
econ min-overs qualifications (published rules have no such thresholds).
"""
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
SRC = PROJECT_ROOT / 'data' / 'ipl_matches_fantasy.parquet'
OUT_MATCH = PROJECT_ROOT / 'data' / 'historical_custom_points_per_match.parquet'
OUT_SEASON = PROJECT_ROOT / 'data' / 'historical_custom_season_totals.csv'

BOWLING_WKT = {'caught', 'bowled', 'lbw', 'stumped', 'caught and bowled', 'hit wicket'}


def sr_pts(sr: float) -> int:
    if sr < 50: return -6
    if sr < 60: return -4
    if sr <= 70: return -2
    if sr <= 130: return 0
    if sr <= 150: return 2
    if sr <= 170: return 4
    return 6


def econ_pts(e: float) -> int:
    if e < 5: return 6
    if e < 6: return 4
    if e <= 7: return 2
    if e <= 10: return 0
    if e <= 11: return -2
    if e <= 12: return -4
    return -6


def milestone(runs: int) -> int:
    p = 0
    if runs >= 30: p += 4
    if runs >= 50: p += 4
    if runs >= 100: p += 8
    return p


def haul(w: int) -> int:
    if w >= 5: return 16
    if w >= 4: return 8
    if w >= 3: return 4
    return 0


def score_batting(df: pd.DataFrame) -> pd.DataFrame:
    """Per (season, match_id, batter): bat points under custom rules."""
    legal = df[df['wides'] == 0].copy()  # wides don't count as balls faced
    bat = legal.groupby(['season', 'match_id', 'batter_name', 'batter_id']).agg(
        runs=('batter_runs', 'sum'),
        balls_faced=('batter_runs', 'count'),
        fours=('batter_runs', lambda x: (x == 4).sum()),
        sixes=('batter_runs', lambda x: (x == 6).sum()),
    ).reset_index()

    dismissals = df[df['is_wicket']].groupby(
        ['season', 'match_id', 'dismissed_player']).size().reset_index(name='times_out')
    bat = bat.merge(dismissals,
                    left_on=['season', 'match_id', 'batter_name'],
                    right_on=['season', 'match_id', 'dismissed_player'],
                    how='left').drop(columns='dismissed_player')
    bat['times_out'] = bat['times_out'].fillna(0).astype(int)
    bat['is_duck'] = (bat['runs'] == 0) & (bat['times_out'] > 0)

    bat['bat_pts'] = (
        bat['runs']
        + bat['fours'] * 1
        + bat['sixes'] * 2
        + bat['runs'].apply(milestone)
        + bat['is_duck'].astype(int) * -2
    )
    # SR bonus/penalty (no min-balls in published rules)
    has_balls = bat['balls_faced'] > 0
    sr = bat.loc[has_balls, 'runs'] / bat.loc[has_balls, 'balls_faced'] * 100.0
    bat.loc[has_balls, 'bat_pts'] += sr.apply(sr_pts)

    return bat.rename(columns={'batter_name': 'player_name', 'batter_id': 'player_id'})


def score_bowling(df: pd.DataFrame) -> pd.DataFrame:
    """Per (season, match_id, bowler): bowling points under custom rules."""
    d = df.copy()
    d['runs_conceded_ball'] = d['total_runs'] - d['byes'] - d['legbyes']
    d['is_bowling_wicket'] = d['is_wicket'] & d['dismissal_type'].isin(BOWLING_WKT)
    d['is_legal'] = (d['wides'] == 0) & (d['noballs'] == 0)
    d['is_dot'] = d['is_legal'] & (d['total_runs'] == 0)
    d['is_wide_ball'] = d['wides'] > 0
    d['is_noball'] = d['noballs'] > 0

    g = d.groupby(['season', 'match_id', 'bowler_name', 'bowler_id']).agg(
        legal_balls=('is_legal', 'sum'),
        runs_conceded=('runs_conceded_ball', 'sum'),
        wickets=('is_bowling_wicket', 'sum'),
        dots=('is_dot', 'sum'),
        wides=('is_wide_ball', 'sum'),
        noballs=('is_noball', 'sum'),
    ).reset_index()
    g['overs'] = g['legal_balls'] / 6.0

    # Maidens
    legal = d[d['is_legal']]
    per_over = legal.groupby(['season', 'match_id', 'bowler_name', 'over']).agg(
        balls=('runs_conceded_ball', 'count'),
        over_runs=('runs_conceded_ball', 'sum'),
    ).reset_index()
    per_over['is_maiden'] = (per_over['balls'] == 6) & (per_over['over_runs'] == 0)
    maidens = per_over[per_over['is_maiden']].groupby(
        ['season', 'match_id', 'bowler_name']).size().reset_index(name='maidens')
    g = g.merge(maidens, on=['season', 'match_id', 'bowler_name'], how='left')
    g['maidens'] = g['maidens'].fillna(0).astype(int)

    g['bowl_pts'] = (
        g['dots'] * 1
        + g['wickets'] * 20
        + g['wickets'].apply(haul)
        + g['maidens'] * 12
        + g['wides'] * -1
        + g['noballs'] * -2
    )
    # Econ bonus/penalty (no min-overs in published rules)
    has_overs = g['overs'] > 0
    e = g.loc[has_overs, 'runs_conceded'] / g.loc[has_overs, 'overs']
    g.loc[has_overs, 'bowl_pts'] += e.apply(econ_pts)

    return g.rename(columns={'bowler_name': 'player_name', 'bowler_id': 'player_id'})


def score_fielding(df: pd.DataFrame) -> pd.DataFrame:
    """Per (season, match_id, fielder): fielding points under custom rules."""
    w = df[df['is_wicket']].copy()
    recs = []

    for _, r in w.iterrows():
        mid, season = r['match_id'], r['season']
        f1, f1id = r['fielder1_name'], r['fielder1_id']
        f2, f2id = r['fielder2_name'], r['fielder2_id']
        kind = r['dismissal_type']
        if f1:
            if kind in ('caught', 'caught and bowled'):
                recs.append((season, mid, f1, f1id, 'catch'))
            elif kind == 'stumped':
                recs.append((season, mid, f1, f1id, 'stumping'))
            elif kind == 'run out':
                recs.append((season, mid, f1, f1id, 'runout'))
        if f2 and kind == 'run out':
            recs.append((season, mid, f2, f2id, 'runout'))

    if not recs:
        return pd.DataFrame(columns=['season', 'match_id', 'player_name', 'player_id',
                                     'catches', 'stumpings', 'runouts', 'field_pts'])
    fd = pd.DataFrame(recs, columns=['season', 'match_id', 'player_name', 'player_id', 'act'])
    fg = fd.groupby(['season', 'match_id', 'player_name', 'player_id']).agg(
        catches=('act', lambda x: (x == 'catch').sum()),
        stumpings=('act', lambda x: (x == 'stumping').sum()),
        runouts=('act', lambda x: (x == 'runout').sum()),
    ).reset_index()
    fg['field_pts'] = (
        fg['catches'] * 8
        + fg['stumpings'] * 6
        + fg['runouts'] * 6
        + (fg['catches'] >= 3).astype(int) * 4
    )
    return fg


def main():
    print(f"Loading {SRC}...")
    df = pd.read_parquet(SRC)
    print(f"  {len(df):,} balls across {df['season'].nunique()} seasons ({df['season'].min()}-{df['season'].max()})")

    print("\nScoring batting...")
    bat = score_batting(df)
    print(f"  {len(bat):,} batter-match rows")

    print("Scoring bowling...")
    bowl = score_bowling(df)
    print(f"  {len(bowl):,} bowler-match rows")

    print("Scoring fielding...")
    field = score_fielding(df)
    print(f"  {len(field):,} fielder-match rows")

    # Combine per (season, match_id, player)
    print("\nCombining...")
    keys = ['season', 'match_id', 'player_name', 'player_id']

    # match_date lookup
    match_dates = df[['season', 'match_id', 'match_date']].drop_duplicates()

    # Start from union of player-match keys across all three frames
    union = pd.concat([bat[keys], bowl[keys], field[keys]], ignore_index=True).drop_duplicates()
    combined = union.merge(match_dates, on=['season', 'match_id'], how='left')

    combined = combined.merge(
        bat[keys + ['runs', 'balls_faced', 'fours', 'sixes', 'is_duck', 'bat_pts']],
        on=keys, how='left'
    )
    combined = combined.merge(
        bowl[keys + ['legal_balls', 'overs', 'runs_conceded', 'wickets',
                     'dots', 'wides', 'noballs', 'maidens', 'bowl_pts']],
        on=keys, how='left'
    )
    combined = combined.merge(
        field[keys + ['catches', 'stumpings', 'runouts', 'field_pts']],
        on=keys, how='left'
    )

    # Fill NaNs — a player may bat but not bowl, etc.
    num_cols = ['runs', 'balls_faced', 'fours', 'sixes',
                'legal_balls', 'overs', 'runs_conceded', 'wickets', 'dots',
                'wides', 'noballs', 'maidens',
                'catches', 'stumpings', 'runouts',
                'bat_pts', 'bowl_pts', 'field_pts']
    for c in num_cols:
        combined[c] = combined[c].fillna(0)
    combined['is_duck'] = combined['is_duck'].fillna(False)

    combined['total_pts'] = (combined['bat_pts'] + combined['bowl_pts'] + combined['field_pts']).astype(int)

    # Cast integer columns
    for c in ('runs', 'balls_faced', 'fours', 'sixes',
              'legal_balls', 'runs_conceded', 'wickets', 'dots',
              'wides', 'noballs', 'maidens',
              'catches', 'stumpings', 'runouts',
              'bat_pts', 'bowl_pts', 'field_pts'):
        combined[c] = combined[c].astype(int)

    # Sort and save per-match
    combined = combined.sort_values(['season', 'match_date', 'match_id', 'total_pts'],
                                    ascending=[True, True, True, False]).reset_index(drop=True)
    combined.to_parquet(OUT_MATCH, index=False)
    print(f"\n✓ {OUT_MATCH}  ({len(combined):,} rows)")

    # Season totals per player
    season = combined.groupby(['season', 'player_name', 'player_id']).agg(
        matches=('match_id', 'nunique'),
        total_pts=('total_pts', 'sum'),
        bat_pts=('bat_pts', 'sum'),
        bowl_pts=('bowl_pts', 'sum'),
        field_pts=('field_pts', 'sum'),
        runs=('runs', 'sum'),
        balls_faced=('balls_faced', 'sum'),
        fours=('fours', 'sum'),
        sixes=('sixes', 'sum'),
        wickets=('wickets', 'sum'),
        maidens=('maidens', 'sum'),
        dots=('dots', 'sum'),
        wides=('wides', 'sum'),
        noballs=('noballs', 'sum'),
        overs=('overs', 'sum'),
        runs_conceded=('runs_conceded', 'sum'),
        catches=('catches', 'sum'),
        stumpings=('stumpings', 'sum'),
        runouts=('runouts', 'sum'),
    ).reset_index()
    season['ppm'] = (season['total_pts'] / season['matches']).round(2)
    season = season.sort_values(['season', 'total_pts'], ascending=[True, False]).reset_index(drop=True)
    season.to_csv(OUT_SEASON, index=False)
    print(f"✓ {OUT_SEASON}  ({len(season):,} player-seasons)")

    print("\n── Top 5 scorers per season (sanity) ──")
    for s in sorted(season['season'].unique()):
        top = season[season['season'] == s].nlargest(3, 'total_pts')
        names = ', '.join(f"{r.player_name} ({int(r.total_pts)})" for _, r in top.iterrows())
        print(f"  {s}: {names}")


if __name__ == '__main__':
    main()
