"""
Calculate Dream11 Fantasy Points per player per T20I match.

Reuses the SAME scoring functions from 02_calculate_fantasy_points.py
applied to the T20I ball-by-ball parquet.

OUTPUT: data/t20i_fantasy_points_per_match.csv
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'WARprojections'))
from utils import save_dataframe

# Import scoring functions from the IPL fantasy points calculator
from importlib.util import spec_from_file_location, module_from_spec
_spec = spec_from_file_location(
    "fp02",
    str(Path(__file__).parent / '02_calculate_fantasy_points.py'),
)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)
calculate_batting_points = _mod.calculate_batting_points
calculate_bowling_points = _mod.calculate_bowling_points
calculate_fielding_points = _mod.calculate_fielding_points
PLAYING_XI_BONUS = _mod.PLAYING_XI_BONUS


def main():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    input_file = data_dir / 't20i_matches_fantasy.parquet'
    output_file = data_dir / 't20i_fantasy_points_per_match.csv'

    print("=" * 60)
    print("T20I DREAM11 FANTASY POINTS CALCULATION")
    print("=" * 60)

    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df):,} balls, {df['match_id'].nunique()} T20I matches")

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

    bat_cols = bat[['match_id', 'season', 'player_name', 'player_id',
                     'runs', 'balls_faced', 'fours', 'sixes',
                     'batting_points']].copy()

    bowl_cols = bowl[['match_id', 'season', 'player_name', 'player_id',
                       'runs_conceded', 'legal_balls', 'overs', 'wickets',
                       'bowling_points']].copy()

    # All unique player-match appearances
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
    if len(field) > 0:
        combined = combined.merge(
            field[['match_id', 'season', 'player_name', 'player_id', 'fielding_points']],
            on=['match_id', 'season', 'player_name', 'player_id'], how='left'
        )
    else:
        combined['fielding_points'] = 0
    combined['fielding_points'] = combined['fielding_points'].fillna(0)

    # Playing XI bonus
    combined['appearance_points'] = PLAYING_XI_BONUS

    # Total
    combined['total_fantasy_points'] = (
        combined['batting_points'] + combined['bowling_points'] +
        combined['fielding_points'] + combined['appearance_points']
    )

    # Fill NAs
    for col in ['runs', 'balls_faced', 'fours', 'sixes',
                'runs_conceded', 'legal_balls', 'overs', 'wickets']:
        if col in combined.columns:
            combined[col] = combined[col].fillna(0)

    # Add match_date from the parquet (one date per match)
    match_dates = df[['match_id', 'match_date', 'team1', 'team2']].drop_duplicates(subset='match_id')
    combined = combined.merge(match_dates, on='match_id', how='left')

    combined = combined.sort_values(['match_date', 'match_id', 'total_fantasy_points'],
                                     ascending=[True, True, False])

    print(f"  {len(combined):,} player-match records")
    print(f"  Mean fantasy points/match: {combined['total_fantasy_points'].mean():.1f}")
    print(f"  Median: {combined['total_fantasy_points'].median():.1f}")

    # ── Sanity check ──────────────────────────────────────────────────
    print("\n--- Top 10 T20I Match Performances ---")
    top = combined.nlargest(10, 'total_fantasy_points')
    for _, r in top.iterrows():
        date_str = str(r['match_date'])[:10] if pd.notna(r['match_date']) else '?'
        print(f"  {r['player_name']:25s}  {r['total_fantasy_points']:6.0f} pts  "
              f"(bat:{r['batting_points']:.0f} bowl:{r['bowling_points']:.0f})  "
              f"{r['team1']} vs {r['team2']}  {date_str}")

    # ── IPL player stats ─────────────────────────────────────────────
    # Load IPL player IDs to show overlap stats
    ipl_df = pd.read_parquet(data_dir / 'ipl_matches_fantasy.parquet')
    ipl_ids = set(ipl_df['batter_id'].unique()) | set(ipl_df['bowler_id'].unique())
    t20i_ids = set(combined['player_id'].unique())
    overlap = ipl_ids & t20i_ids
    print(f"\nT20I players: {len(t20i_ids)}")
    print(f"IPL overlap: {len(overlap)} ({100*len(overlap)/len(ipl_ids):.1f}% of IPL universe)")

    save_dataframe(combined, output_file, format='csv')
    print(f"\n✓ Saved to {output_file}")


if __name__ == '__main__':
    main()
