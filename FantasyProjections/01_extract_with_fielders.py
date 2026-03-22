"""
Extract FULL IPL ball-by-ball data (2008-2025) from Cricsheet JSON files,
INCLUDING fielder information for Dream11 fantasy point calculations.

Adds: fielder1_name, fielder1_id, fielder2_name, fielder2_id
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from tqdm import tqdm

# Add WARprojections to path so we can reuse utils
sys.path.insert(0, str(Path(__file__).parent.parent / 'WARprojections'))
from utils import (
    load_cricsheet_match,
    is_ipl_match,
    extract_player_id,
    normalize_runs,
    get_wicket_info,
    get_extras_info,
    create_match_id,
    save_dataframe,
)


def extract_fielder_info(
    wicket: Optional[Dict[str, Any]], registry: Dict[str, str]
) -> Dict[str, Any]:
    """Extract fielder names and IDs from a wicket dictionary."""
    result = {
        'fielder1_name': '',
        'fielder1_id': '',
        'fielder2_name': '',
        'fielder2_id': '',
    }
    if wicket is None:
        return result

    fielders = wicket.get('fielders', [])
    if len(fielders) >= 1:
        f1 = fielders[0]
        name = f1.get('name', '')
        result['fielder1_name'] = name
        result['fielder1_id'] = extract_player_id(name, registry) or ''
    if len(fielders) >= 2:
        f2 = fielders[1]
        name = f2.get('name', '')
        result['fielder2_name'] = name
        result['fielder2_id'] = extract_player_id(name, registry) or ''

    return result


def extract_ball_data(match_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract ball-by-ball data from a match, including fielder info."""
    info = match_data.get('info', {})
    innings_list = match_data.get('innings', [])
    registry = info.get('registry', {})

    match_id = create_match_id(info)
    venue = info.get('venue', '')
    match_date = info.get('dates', [''])[0]
    teams = info.get('teams', [])
    toss_winner = info.get('toss', {}).get('winner', '')
    toss_decision = info.get('toss', {}).get('decision', '')
    season = info.get('season', '')
    match_type = info.get('match_type', '')

    balls = []

    for innings_idx, innings in enumerate(innings_list):
        innings_number = innings_idx + 1
        batting_team = innings.get('team', '')
        bowling_team = (
            teams[1] if len(teams) > 1 and teams[0] == batting_team
            else teams[0] if len(teams) > 1
            else ''
        )

        score = 0
        wickets = 0
        total_balls = 0

        overs = innings.get('overs', [])
        for over_data in overs:
            over_number = over_data.get('over', 0)

            deliveries = over_data.get('deliveries', [])
            for ball_number, delivery in enumerate(deliveries):
                batter_name = delivery.get('batter', '')
                bowler_name = delivery.get('bowler', '')
                non_striker_name = delivery.get('non_striker', '')

                batter_id = extract_player_id(batter_name, registry)
                bowler_id = extract_player_id(bowler_name, registry)
                non_striker_id = extract_player_id(non_striker_name, registry)

                runs_info = delivery.get('runs', {})
                batter_runs = runs_info.get('batter', 0)
                extras = runs_info.get('extras', 0)
                total_runs = runs_info.get('total', 0)

                extras_info = get_extras_info(delivery)
                wicket = get_wicket_info(delivery)
                is_wicket = wicket is not None
                dismissal_type = wicket.get('kind', '') if wicket else ''
                dismissed_player = wicket.get('player_out', '') if wicket else ''

                # NEW: extract fielder information
                fielder_info = extract_fielder_info(wicket, registry)

                ball_data = {
                    'match_id': match_id,
                    'venue': venue,
                    'match_date': match_date,
                    'season': season,
                    'match_type': match_type,
                    'toss_winner': toss_winner,
                    'toss_decision': toss_decision,
                    'innings': innings_number,
                    'batting_team': batting_team,
                    'bowling_team': bowling_team,
                    'over': over_number,
                    'ball_in_over': ball_number,
                    'total_balls': total_balls,
                    'score_before': score,
                    'wickets_before': wickets,
                    'batter_name': batter_name,
                    'batter_id': batter_id,
                    'bowler_name': bowler_name,
                    'bowler_id': bowler_id,
                    'non_striker_name': non_striker_name,
                    'non_striker_id': non_striker_id,
                    'batter_runs': batter_runs,
                    'extras': extras,
                    'total_runs': total_runs,
                    'wides': extras_info['wides'],
                    'noballs': extras_info['noballs'],
                    'byes': extras_info['byes'],
                    'legbyes': extras_info['legbyes'],
                    'is_wicket': is_wicket,
                    'dismissal_type': dismissal_type,
                    'dismissed_player': dismissed_player,
                    # NEW fielder columns
                    'fielder1_name': fielder_info['fielder1_name'],
                    'fielder1_id': fielder_info['fielder1_id'],
                    'fielder2_name': fielder_info['fielder2_name'],
                    'fielder2_id': fielder_info['fielder2_id'],
                }

                balls.append(ball_data)

                score += total_runs
                if is_wicket:
                    wickets += 1
                if extras_info['wides'] == 0 and extras_info['noballs'] == 0:
                    total_balls += 1

    return balls


def main():
    project_root = Path(__file__).parent.parent
    cricsheet_dir = project_root / 'data' / 'ipl_json'
    output_file = project_root / 'data' / 'ipl_matches_fantasy.parquet'

    print(f"Reading Cricsheet files from: {cricsheet_dir}")

    if not cricsheet_dir.exists():
        raise FileNotFoundError(f"Cricsheet directory not found: {cricsheet_dir}")

    json_files = list(cricsheet_dir.glob('*.json'))
    print(f"Found {len(json_files)} JSON files")

    all_balls = []
    ipl_match_count = 0

    print("\nExtracting ALL IPL matches (2008-2025) with fielder data...")

    for json_file in tqdm(json_files, desc="Processing files"):
        try:
            match_data = load_cricsheet_match(json_file)

            if is_ipl_match(match_data, year_start=2008, year_end=2025, exclude_years=[]):
                balls = extract_ball_data(match_data)
                all_balls.extend(balls)
                ipl_match_count += 1

        except Exception as e:
            print(f"\nError processing {json_file.name}: {e}")
            continue

    print(f"\n✓ Extracted {ipl_match_count} IPL matches")
    print(f"✓ Total balls: {len(all_balls):,}")

    df = pd.DataFrame(all_balls)

    # Derived columns (same as WAR pipeline)
    df['season'] = df['season'].astype(str)
    df['batter_runs_normalized'] = df['batter_runs'].apply(normalize_runs)
    df['balls_remaining'] = 120 - df['total_balls']
    df['phase'] = df['over'].apply(
        lambda o: 'powerplay' if o < 6 else ('death' if o >= 16 else 'middle')
    )
    df['is_powerplay'] = df['over'] < 6

    # Fix season names
    df['season'] = df['season'].apply(lambda x: x.split('/')[0] if '/' in x else x)
    df['season'] = pd.to_numeric(df['season'], errors='coerce').fillna(0).astype(int)

    # Summary stats for fielder data
    fielder_balls = (df['fielder1_name'] != '').sum()
    two_fielder_balls = (df['fielder2_name'] != '').sum()

    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Total matches: {df['match_id'].nunique()}")
    print(f"Total balls: {len(df):,}")
    print(f"Seasons: {sorted(df['season'].unique())}")
    print(f"Balls with fielder1 info: {fielder_balls:,}")
    print(f"Balls with fielder2 info: {two_fielder_balls:,}")
    print(f"\nDismissal type breakdown:")
    print(df[df['is_wicket']]['dismissal_type'].value_counts().to_string())

    print(f"\nSaving to {output_file}...")
    save_dataframe(df, output_file, format='parquet')
    print("✓ Done")


if __name__ == '__main__':
    main()
