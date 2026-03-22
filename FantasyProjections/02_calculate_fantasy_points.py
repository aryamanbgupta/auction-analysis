"""
Calculate Dream11 Fantasy Points per player per match from ball-by-ball data.

Dream11 T20 Scoring:
  Batting: +1/run, +1 boundary bonus, +2 six bonus, +4/+8/+16 milestones (30/50/100),
           -2 duck (BAT/WK/AR only), SR bonuses/penalties (min 10 balls)
  Bowling: +25/wicket, +8 bowled/LBW, +12 maiden, +4/+8/+16 haul (3/4/5 wkts),
           economy bonuses/penalties (min 2 overs)
  Fielding: +8 catch, +12 stumping, +12 direct run out, +6 indirect run out, +4 for 3+ catches
  Other: +4 playing XI appearance

Outputs:
  - data/fantasy_points_per_match.csv   (player × match level)
  - data/fantasy_points_per_season.csv  (player × season level)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / 'WARprojections'))
from utils import save_dataframe


# ── Dream11 scoring constants ──────────────────────────────────────────────

POINTS_PER_RUN = 1
BOUNDARY_BONUS = 1       # extra for hitting a 4
SIX_BONUS = 2            # extra for hitting a 6
MILESTONE_30 = 4
MILESTONE_50 = 8
MILESTONE_100 = 16
DUCK_PENALTY = -2        # only for BAT/WK/AR roles

SR_MIN_BALLS = 10
SR_BONUSES = [
    (170.01, float('inf'), 6),
    (150.01, 170.0, 4),
    (130.0, 150.0, 2),
    (60.0, 70.0, -2),
    (50.0, 59.99, -4),
    (0, 49.99, -6),
]

WICKET_POINTS = 25
BOWLED_LBW_BONUS = 8
MAIDEN_POINTS = 12
HAUL_3 = 4
HAUL_4 = 8
HAUL_5 = 16

ECON_MIN_OVERS = 2
ECON_BONUSES = [
    (0, 4.99, 6),
    (5.0, 5.99, 4),
    (6.0, 7.0, 2),
    (10.0, 11.0, -2),
    (11.01, 12.0, -4),
    (12.01, float('inf'), -6),
]

CATCH_POINTS = 8
STUMPING_POINTS = 12
DIRECT_RUNOUT_POINTS = 12
INDIRECT_RUNOUT_POINTS = 6
THREE_CATCH_BONUS = 4

PLAYING_XI_BONUS = 4


# ── Helper functions ───────────────────────────────────────────────────────

def calc_sr_bonus(runs: int, balls: int) -> int:
    if balls < SR_MIN_BALLS:
        return 0
    sr = (runs / balls) * 100
    for lo, hi, pts in SR_BONUSES:
        if lo <= sr <= hi:
            return pts
    return 0


def calc_milestone_bonus(runs: int) -> int:
    """Only the highest milestone bonus applies."""
    if runs >= 100:
        return MILESTONE_100
    elif runs >= 50:
        return MILESTONE_50
    elif runs >= 30:
        return MILESTONE_30
    return 0


def calc_econ_bonus(runs_conceded: int, overs: float) -> int:
    if overs < ECON_MIN_OVERS:
        return 0
    econ = runs_conceded / overs
    for lo, hi, pts in ECON_BONUSES:
        if lo <= econ <= hi:
            return pts
    return 0


def calc_haul_bonus(wickets: int) -> int:
    """Only the highest haul bonus applies."""
    if wickets >= 5:
        return HAUL_5
    elif wickets >= 4:
        return HAUL_4
    elif wickets >= 3:
        return HAUL_3
    return 0


# ── Main calculation ───────────────────────────────────────────────────────

def calculate_batting_points(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate batting fantasy points per batter per match."""
    # Exclude wides from balls faced (wides don't count as balls faced)
    legal_balls = df[df['wides'] == 0].copy()

    batting = legal_balls.groupby(['match_id', 'season', 'batter_name', 'batter_id']).agg(
        runs=('batter_runs', 'sum'),
        balls_faced=('batter_runs', 'count'),
        fours=('batter_runs', lambda x: (x == 4).sum()),
        sixes=('batter_runs', lambda x: (x == 6).sum()),
    ).reset_index()

    # Check if batter was dismissed (duck detection)
    dismissals = df[df['is_wicket']].groupby(
        ['match_id', 'dismissed_player']
    ).size().reset_index(name='times_out')
    dismissals = dismissals.rename(columns={'dismissed_player': 'batter_name'})

    batting = batting.merge(dismissals, on=['match_id', 'batter_name'], how='left')
    batting['times_out'] = batting['times_out'].fillna(0).astype(int)
    batting['is_out'] = batting['times_out'] > 0
    batting['is_duck'] = (batting['runs'] == 0) & (batting['is_out'])

    # Calculate points
    batting['run_pts'] = batting['runs'] * POINTS_PER_RUN
    batting['boundary_pts'] = batting['fours'] * BOUNDARY_BONUS + batting['sixes'] * SIX_BONUS
    batting['milestone_pts'] = batting['runs'].apply(calc_milestone_bonus)
    batting['sr_pts'] = batting.apply(
        lambda r: calc_sr_bonus(r['runs'], r['balls_faced']), axis=1
    )
    # Duck penalty — we'll apply role filtering later (only BAT/WK/AR)
    batting['duck_pts'] = batting['is_duck'].astype(int) * DUCK_PENALTY

    batting['batting_points'] = (
        batting['run_pts'] + batting['boundary_pts'] +
        batting['milestone_pts'] + batting['sr_pts'] + batting['duck_pts']
    )

    return batting.rename(columns={
        'batter_name': 'player_name',
        'batter_id': 'player_id',
    })


def calculate_bowling_points(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate bowling fantasy points per bowler per match."""
    # Legal deliveries only (no wides or noballs for over counting)
    legal = df[(df['wides'] == 0) & (df['noballs'] == 0)].copy()

    # Runs conceded = total_runs - byes - legbyes (byes/legbyes aren't charged to bowler)
    df_bowl = df.copy()
    df_bowl['runs_conceded'] = df_bowl['total_runs'] - df_bowl['byes'] - df_bowl['legbyes']

    # Wickets: only bowling dismissals (not run outs, retired, obstructing)
    bowling_dismissals = {'caught', 'bowled', 'lbw', 'stumped', 'caught and bowled', 'hit wicket'}
    bowled_lbw_types = {'bowled', 'lbw'}

    # Per-ball wicket flags
    df_bowl['is_bowling_wicket'] = (
        df_bowl['is_wicket'] & df_bowl['dismissal_type'].isin(bowling_dismissals)
    )
    df_bowl['is_bowled_lbw'] = (
        df_bowl['is_wicket'] & df_bowl['dismissal_type'].isin(bowled_lbw_types)
    )

    # Count legal balls per bowler per match (for overs calc)
    legal_bowl = df_bowl[(df_bowl['wides'] == 0) & (df_bowl['noballs'] == 0)]
    legal_balls = legal_bowl.groupby(
        ['match_id', 'season', 'bowler_name', 'bowler_id']
    ).size().reset_index(name='legal_balls')

    bowling = df_bowl.groupby(['match_id', 'season', 'bowler_name', 'bowler_id']).agg(
        runs_conceded=('runs_conceded', 'sum'),
        wickets=('is_bowling_wicket', 'sum'),
        bowled_lbw=('is_bowled_lbw', 'sum'),
    ).reset_index()

    bowling = bowling.merge(legal_balls, on=['match_id', 'season', 'bowler_name', 'bowler_id'], how='left')
    bowling['legal_balls'] = bowling['legal_balls'].fillna(0).astype(int)
    bowling['overs'] = bowling['legal_balls'] / 6.0

    # Maiden over detection: group by match+bowler+over, check if 0 runs conceded in legal balls
    maiden_check = legal_bowl.copy()
    maiden_check['runs_conceded'] = maiden_check['total_runs'] - maiden_check['byes'] - maiden_check['legbyes']
    maiden_overs = maiden_check.groupby(
        ['match_id', 'bowler_name', 'over']
    ).agg(
        over_runs=('runs_conceded', 'sum'),
        balls_in_over=('runs_conceded', 'count'),
    ).reset_index()
    # A maiden = 6 legal balls, 0 runs conceded
    maiden_overs['is_maiden'] = (maiden_overs['balls_in_over'] == 6) & (maiden_overs['over_runs'] == 0)
    maidens = maiden_overs[maiden_overs['is_maiden']].groupby(
        ['match_id', 'bowler_name']
    ).size().reset_index(name='maidens')

    bowling = bowling.merge(maidens, on=['match_id', 'bowler_name'], how='left')
    bowling['maidens'] = bowling['maidens'].fillna(0).astype(int)

    # Calculate points
    bowling['wicket_pts'] = bowling['wickets'] * WICKET_POINTS
    bowling['bowled_lbw_pts'] = bowling['bowled_lbw'] * BOWLED_LBW_BONUS
    bowling['maiden_pts'] = bowling['maidens'] * MAIDEN_POINTS
    bowling['haul_pts'] = bowling['wickets'].apply(calc_haul_bonus)
    bowling['econ_pts'] = bowling.apply(
        lambda r: calc_econ_bonus(r['runs_conceded'], r['overs']), axis=1
    )

    bowling['bowling_points'] = (
        bowling['wicket_pts'] + bowling['bowled_lbw_pts'] +
        bowling['maiden_pts'] + bowling['haul_pts'] + bowling['econ_pts']
    )

    return bowling.rename(columns={
        'bowler_name': 'player_name',
        'bowler_id': 'player_id',
    })


def calculate_fielding_points(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate fielding fantasy points from fielder columns."""
    wicket_balls = df[df['is_wicket']].copy()

    records = []

    # Fielder1 credits
    f1 = wicket_balls[wicket_balls['fielder1_name'] != ''].copy()
    for _, row in f1.iterrows():
        kind = row['dismissal_type']
        fielder = row['fielder1_name']
        fielder_id = row['fielder1_id']
        match_id = row['match_id']
        season = row['season']

        if kind in ('caught', 'caught and bowled'):
            pts = CATCH_POINTS
            action = 'catch'
        elif kind == 'stumped':
            pts = STUMPING_POINTS
            action = 'stumping'
        elif kind == 'run out':
            # If only 1 fielder, it's a direct hit
            has_f2 = row['fielder2_name'] != ''
            if has_f2:
                pts = INDIRECT_RUNOUT_POINTS  # last 2 fielders each get 6
                action = 'run_out_indirect'
            else:
                pts = DIRECT_RUNOUT_POINTS
                action = 'run_out_direct'
        else:
            continue

        records.append({
            'match_id': match_id,
            'season': season,
            'player_name': fielder,
            'player_id': fielder_id,
            'action': action,
            'points': pts,
        })

    # Fielder2 credits (run outs with 2 fielders)
    f2 = wicket_balls[wicket_balls['fielder2_name'] != ''].copy()
    for _, row in f2.iterrows():
        if row['dismissal_type'] == 'run out':
            records.append({
                'match_id': row['match_id'],
                'season': row['season'],
                'player_name': row['fielder2_name'],
                'player_id': row['fielder2_id'],
                'action': 'run_out_indirect',
                'points': INDIRECT_RUNOUT_POINTS,
            })

    if not records:
        return pd.DataFrame(columns=[
            'match_id', 'season', 'player_name', 'player_id',
            'catches', 'stumpings', 'run_outs', 'fielding_points'
        ])

    field_df = pd.DataFrame(records)

    # Aggregate per player per match
    agg = field_df.groupby(['match_id', 'season', 'player_name', 'player_id']).agg(
        catches=('action', lambda x: (x.isin(['catch'])).sum()),
        stumpings=('action', lambda x: (x == 'stumping').sum()),
        run_outs=('action', lambda x: x.str.startswith('run_out').sum()),
        fielding_points=('points', 'sum'),
    ).reset_index()

    # 3+ catches bonus
    agg['three_catch_bonus'] = (agg['catches'] >= 3).astype(int) * THREE_CATCH_BONUS
    agg['fielding_points'] += agg['three_catch_bonus']
    agg = agg.drop(columns=['three_catch_bonus'])

    return agg


def main():
    project_root = Path(__file__).parent.parent
    input_file = project_root / 'data' / 'ipl_matches_fantasy.parquet'
    match_output = project_root / 'data' / 'fantasy_points_per_match.csv'
    season_output = project_root / 'data' / 'fantasy_points_per_season.csv'

    print("Loading ball-by-ball data...")
    df = pd.read_parquet(input_file)
    print(f"  {len(df):,} balls, {df['match_id'].nunique()} matches, seasons {df['season'].min()}-{df['season'].max()}")

    # ── Batting ────────────────────────────────────────────────────────
    print("\nCalculating batting points...")
    bat = calculate_batting_points(df)
    print(f"  {len(bat):,} batter-match records")

    # ── Bowling ────────────────────────────────────────────────────────
    print("Calculating bowling points...")
    bowl = calculate_bowling_points(df)
    print(f"  {len(bowl):,} bowler-match records")

    # ── Fielding ───────────────────────────────────────────────────────
    print("Calculating fielding points...")
    field = calculate_fielding_points(df)
    print(f"  {len(field):,} fielder-match records")

    # ── Combine per player per match ───────────────────────────────────
    print("\nCombining all contributions...")

    # Batting columns to keep
    bat_cols = bat[['match_id', 'season', 'player_name', 'player_id',
                     'runs', 'balls_faced', 'fours', 'sixes', 'is_duck',
                     'batting_points']].copy()

    # Bowling columns to keep
    bowl_cols = bowl[['match_id', 'season', 'player_name', 'player_id',
                       'runs_conceded', 'legal_balls', 'overs', 'wickets',
                       'bowled_lbw', 'maidens', 'bowling_points']].copy()

    # Start with all unique player-match appearances
    # A player appears if they batted OR bowled OR fielded
    all_appearances = set()

    for src in [bat_cols, bowl_cols, field]:
        if len(src) > 0:
            for _, row in src[['match_id', 'season', 'player_name', 'player_id']].drop_duplicates().iterrows():
                all_appearances.add((row['match_id'], row['season'], row['player_name'], row['player_id']))

    combined = pd.DataFrame(
        list(all_appearances),
        columns=['match_id', 'season', 'player_name', 'player_id']
    )

    # Merge batting
    combined = combined.merge(
        bat_cols, on=['match_id', 'season', 'player_name', 'player_id'], how='left'
    )
    combined['batting_points'] = combined['batting_points'].fillna(0)

    # Merge bowling
    combined = combined.merge(
        bowl_cols, on=['match_id', 'season', 'player_name', 'player_id'], how='left'
    )
    combined['bowling_points'] = combined['bowling_points'].fillna(0)

    # Merge fielding
    combined = combined.merge(
        field[['match_id', 'season', 'player_name', 'player_id',
               'catches', 'stumpings', 'run_outs', 'fielding_points']],
        on=['match_id', 'season', 'player_name', 'player_id'], how='left'
    )
    combined['fielding_points'] = combined['fielding_points'].fillna(0)

    # Playing XI bonus (+4 for every appearance)
    combined['appearance_points'] = PLAYING_XI_BONUS

    # Total fantasy points
    combined['total_fantasy_points'] = (
        combined['batting_points'] + combined['bowling_points'] +
        combined['fielding_points'] + combined['appearance_points']
    )

    # Fill NAs in stat columns
    for col in ['runs', 'balls_faced', 'fours', 'sixes', 'is_duck',
                'runs_conceded', 'legal_balls', 'overs', 'wickets',
                'bowled_lbw', 'maidens', 'catches', 'stumpings', 'run_outs']:
        if col in combined.columns:
            combined[col] = combined[col].fillna(0)

    combined = combined.sort_values(['season', 'match_id', 'total_fantasy_points'], ascending=[True, True, False])

    print(f"  {len(combined):,} player-match records")
    print(f"  Mean fantasy points/match: {combined['total_fantasy_points'].mean():.1f}")
    print(f"  Median: {combined['total_fantasy_points'].median():.1f}")
    print(f"  Max: {combined['total_fantasy_points'].max():.1f}")

    # ── Top performances sanity check ──────────────────────────────────
    print("\n--- Top 10 Match Performances (All Time) ---")
    top = combined.nlargest(10, 'total_fantasy_points')
    for _, r in top.iterrows():
        print(f"  {r['player_name']:25s}  {r['total_fantasy_points']:6.0f} pts  "
              f"(bat:{r['batting_points']:.0f} bowl:{r['bowling_points']:.0f} "
              f"field:{r['fielding_points']:.0f})  season {int(r['season'])}")

    # ── Save match-level ───────────────────────────────────────────────
    print(f"\nSaving match-level to {match_output}...")
    save_dataframe(combined, match_output, format='csv')

    # ── Aggregate to season level ──────────────────────────────────────
    print("Aggregating to season level...")
    season_agg = combined.groupby(['season', 'player_name', 'player_id']).agg(
        matches=('match_id', 'nunique'),
        total_fantasy_pts=('total_fantasy_points', 'sum'),
        avg_fantasy_pts=('total_fantasy_points', 'mean'),
        std_fantasy_pts=('total_fantasy_points', 'std'),
        median_fantasy_pts=('total_fantasy_points', 'median'),
        max_fantasy_pts=('total_fantasy_points', 'max'),
        min_fantasy_pts=('total_fantasy_points', 'min'),
        total_batting_pts=('batting_points', 'sum'),
        total_bowling_pts=('bowling_points', 'sum'),
        total_fielding_pts=('fielding_points', 'sum'),
        total_runs=('runs', 'sum'),
        total_balls_faced=('balls_faced', 'sum'),
        total_fours=('fours', 'sum'),
        total_sixes=('sixes', 'sum'),
        total_wickets=('wickets', 'sum'),
        total_maidens=('maidens', 'sum'),
        total_catches=('catches', 'sum'),
        total_stumpings=('stumpings', 'sum'),
        total_run_outs=('run_outs', 'sum'),
    ).reset_index()

    season_agg['std_fantasy_pts'] = season_agg['std_fantasy_pts'].fillna(0)

    # Points breakdown shares
    total = season_agg['total_fantasy_pts'].replace(0, np.nan)
    season_agg['batting_pts_share'] = season_agg['total_batting_pts'] / total
    season_agg['bowling_pts_share'] = season_agg['total_bowling_pts'] / total
    season_agg['fielding_pts_share'] = season_agg['total_fielding_pts'] / total
    season_agg[['batting_pts_share', 'bowling_pts_share', 'fielding_pts_share']] = \
        season_agg[['batting_pts_share', 'bowling_pts_share', 'fielding_pts_share']].fillna(0)

    print(f"\n--- Season-Level Summary ---")
    print(f"  {len(season_agg):,} player-season records")
    print(f"  Avg fantasy pts/match (overall): {season_agg['avg_fantasy_pts'].mean():.1f}")

    print("\n--- Top 10 Season Averages (min 7 matches) ---")
    qualified = season_agg[season_agg['matches'] >= 7]
    top_season = qualified.nlargest(10, 'avg_fantasy_pts')
    for _, r in top_season.iterrows():
        print(f"  {r['player_name']:25s}  {r['avg_fantasy_pts']:5.1f} pts/match  "
              f"({int(r['matches'])} matches, season {int(r['season'])})")

    print(f"\nSaving season-level to {season_output}...")
    save_dataframe(season_agg, season_output, format='csv')
    print("✓ Done")


if __name__ == '__main__':
    main()
