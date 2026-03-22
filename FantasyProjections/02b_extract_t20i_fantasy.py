"""
Extract T20I ball-by-ball data for IPL players with ALL columns needed for Dream11 scoring.

Reuses extraction logic from 01_extract_with_fielders.py but targets T20I matches.
Filters to only matches where at least one player is in the IPL universe.

OUTPUT: data/t20i_matches_fantasy.parquet
"""

import sys
from pathlib import Path
from typing import Set
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / 'WARprojections'))
from utils import (
    load_cricsheet_match,
    extract_player_id,
    normalize_runs,
    get_wicket_info,
    get_extras_info,
    create_match_id,
    save_dataframe,
)

# Reuse fielder extraction from 01
from importlib.util import spec_from_file_location, module_from_spec
_spec = spec_from_file_location(
    "extract01",
    str(Path(__file__).parent / '01_extract_with_fielders.py'),
)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)
extract_fielder_info = _mod.extract_fielder_info
extract_ball_data = _mod.extract_ball_data


def load_ipl_player_ids(ipl_parquet: Path) -> Set[str]:
    """Load set of all player IDs who have appeared in IPL."""
    df = pd.read_parquet(ipl_parquet)
    batters = set(df['batter_id'].dropna().unique())
    bowlers = set(df['bowler_id'].dropna().unique())
    ipl_players = batters | bowlers
    print(f"IPL universe: {len(ipl_players)} unique players")
    return ipl_players


def get_match_player_ids(match_data):
    """Get set of player IDs from a match's registry."""
    info = match_data.get('info', {})
    registry = info.get('registry', {}).get('people', {})
    return set(registry.values())


def main():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    t20i_dir = data_dir / 'other_t20_data' / 't20s_json'
    output_file = data_dir / 't20i_matches_fantasy.parquet'

    print("=" * 60)
    print("T20I EXTRACTION FOR FANTASY SCORING")
    print("=" * 60)

    if not t20i_dir.exists():
        raise FileNotFoundError(f"T20I directory not found: {t20i_dir}")

    # Load IPL player universe
    ipl_parquet = data_dir / 'ipl_matches_fantasy.parquet'
    ipl_players = load_ipl_player_ids(ipl_parquet)

    json_files = sorted(t20i_dir.glob('*.json'))
    print(f"T20I JSON files: {len(json_files)}")

    all_balls = []
    match_count = 0
    skipped = 0

    for json_file in tqdm(json_files, desc="Extracting T20I"):
        try:
            match_data = load_cricsheet_match(json_file)

            # Filter: at least one IPL player in this match
            match_players = get_match_player_ids(match_data)
            if not match_players & ipl_players:
                skipped += 1
                continue

            # Reuse the same extraction function as IPL
            balls = extract_ball_data(match_data)

            # Add T20I-specific columns
            info = match_data.get('info', {})
            teams = info.get('teams', [])
            team1 = teams[0] if len(teams) > 0 else ''
            team2 = teams[1] if len(teams) > 1 else ''

            for ball in balls:
                ball['league'] = 'T20I'
                ball['team1'] = team1
                ball['team2'] = team2

            all_balls.extend(balls)
            match_count += 1

        except Exception as e:
            print(f"\nError {json_file.name}: {e}")
            continue

    print(f"\n✓ Extracted {match_count} T20I matches (skipped {skipped} with no IPL players)")
    print(f"✓ Total balls: {len(all_balls):,}")

    df = pd.DataFrame(all_balls)

    # Derived columns (same as IPL extraction)
    df['season'] = df['season'].astype(str).apply(lambda x: x.split('/')[0])
    df['season'] = pd.to_numeric(df['season'], errors='coerce').fillna(0).astype(int)
    df['batter_runs_normalized'] = df['batter_runs'].apply(normalize_runs)
    df['balls_remaining'] = 120 - df['total_balls']
    df['phase'] = df['over'].apply(
        lambda o: 'powerplay' if o < 6 else ('death' if o >= 16 else 'middle')
    )
    df['is_powerplay'] = df['over'] < 6

    # Convert match_date to datetime for windowing later
    df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')

    print(f"\n{'='*60}")
    print("T20I EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Total matches: {df['match_id'].nunique()}")
    print(f"Total balls: {len(df):,}")
    print(f"Seasons: {sorted(df['season'].unique())}")
    print(f"Unique batters: {df['batter_id'].nunique()}")
    print(f"Unique bowlers: {df['bowler_id'].nunique()}")
    all_players = set(df['batter_id'].unique()) | set(df['bowler_id'].unique())
    ipl_overlap = all_players & ipl_players
    print(f"Players also in IPL: {len(ipl_overlap)}")
    print(f"\nDismissal type breakdown:")
    print(df[df['is_wicket']]['dismissal_type'].value_counts().to_string())

    save_dataframe(df, output_file, format='parquet')
    print(f"\n✓ Saved to {output_file}")


if __name__ == '__main__':
    main()
