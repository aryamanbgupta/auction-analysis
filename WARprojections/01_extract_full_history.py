"""
Extract FULL IPL ball-by-ball data (2008-2025) from Cricsheet JSON files.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from tqdm import tqdm
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

def extract_ball_data(match_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract ball-by-ball data from a match."""
    info = match_data.get('info', {})
    innings_list = match_data.get('innings', [])
    registry = info.get('registry', {})

    # Match-level info
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
        bowling_team = teams[1] if teams[0] == batting_team else teams[0] if len(teams) > 1 else ''

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
    output_file = project_root / 'data' / 'ipl_matches_all.parquet'

    print(f"Reading Cricsheet files from: {cricsheet_dir}")
    
    if not cricsheet_dir.exists():
        raise FileNotFoundError(f"Cricsheet directory not found: {cricsheet_dir}")

    json_files = list(cricsheet_dir.glob('*.json'))
    print(f"Found {len(json_files)} JSON files")

    all_balls = []
    ipl_match_count = 0

    print("\nExtracting ALL IPL matches (2008-2025)...")

    for json_file in tqdm(json_files, desc="Processing files"):
        try:
            match_data = load_cricsheet_match(json_file)
            
            # Extract ALL years (2008 to 2025)
            # We still exclude 2020 if needed, but let's keep it for now as we want full history
            # Actually, let's keep the exclude_years=[] to get everything
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

    print("\nAdding derived columns...")
    df['season'] = df['season'].astype(str)
    df['batter_runs_normalized'] = df['batter_runs'].apply(normalize_runs)
    df['balls_remaining'] = 120 - df['total_balls']
    df['phase'] = df['over'].apply(lambda o: 'powerplay' if o < 6 else ('death' if o >= 16 else 'middle'))
    df['is_powerplay'] = df['over'] < 6
    
    # Fix season names (e.g. "2007/08" -> "2008")
    df['season'] = df['season'].apply(lambda x: x.split('/')[0] if '/' in x else x)
    # Ensure numeric
    df['season'] = pd.to_numeric(df['season'], errors='coerce').fillna(0).astype(int)

    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    print(f"Total matches: {df['match_id'].nunique()}")
    print(f"Total balls: {len(df):,}")
    print(f"Seasons: {sorted(df['season'].unique())}")
    
    print(f"\nSaving to {output_file}...")
    save_dataframe(df, output_file, format='parquet')
    print("✓ Done")

if __name__ == '__main__':
    main()
