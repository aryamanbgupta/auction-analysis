"""
Extract global T20 data for IPL players.
Source: data/other_t20_data/
Filter: Only matches involving at least one player who has played in the IPL.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Set
import pandas as pd
from tqdm import tqdm
from utils import (
    load_cricsheet_match,
    extract_player_id,
    normalize_runs,
    get_wicket_info,
    get_extras_info,
    create_match_id,
    save_dataframe,
)

def load_ipl_player_ids(ipl_data_path: Path) -> Set[str]:
    """Load set of all player IDs who have appeared in IPL."""
    print(f"Loading IPL player list from {ipl_data_path}...")
    df = pd.read_parquet(ipl_data_path)
    
    batters = set(df['batter_id'].unique())
    bowlers = set(df['bowler_id'].unique())
    
    ipl_players = batters.union(bowlers)
    print(f"✓ Found {len(ipl_players):,} unique IPL players")
    return ipl_players

def get_match_players(match_data: Dict[str, Any], registry: Dict[str, str]) -> Set[str]:
    """Get set of all player IDs in a match."""
    players = set()
    info = match_data.get('info', {})
    
    # Check registry first (fastest)
    if 'registry' in info and 'people' in info['registry']:
        return set(info['registry']['people'].values())
        
    # Fallback: Iterate innings (slower)
    for innings in match_data.get('innings', []):
        for over in innings.get('overs', []):
            for delivery in over.get('deliveries', []):
                players.add(extract_player_id(delivery.get('batter', ''), registry))
                players.add(extract_player_id(delivery.get('bowler', ''), registry))
                
    return {p for p in players if p} # Remove None

def extract_global_ball_data(match_data: Dict[str, Any], league_name: str) -> List[Dict[str, Any]]:
    """Extract ball-by-ball data with league tag and team information."""
    info = match_data.get('info', {})
    innings_list = match_data.get('innings', [])
    registry = info.get('registry', {})

    match_id = create_match_id(info)
    venue = info.get('venue', '')
    match_date = info.get('dates', [''])[0]
    season = info.get('season', '')
    match_type = info.get('match_type', '')
    
    # Extract teams for T20I stratification
    teams = info.get('teams', [])
    team1 = teams[0] if len(teams) > 0 else ''
    team2 = teams[1] if len(teams) > 1 else ''
    
    # Standardize season (handle "2023/24")
    season = str(season).split('/')[0]

    balls = []

    for innings_idx, innings in enumerate(innings_list):
        innings_number = innings_idx + 1
        batting_team = innings.get('team', '')
        
        score = 0
        wickets = 0
        total_balls = 0

        for over_data in innings.get('overs', []):
            over_number = over_data.get('over', 0)

            for ball_number, delivery in enumerate(over_data.get('deliveries', [])):
                batter_name = delivery.get('batter', '')
                bowler_name = delivery.get('bowler', '')
                
                batter_id = extract_player_id(batter_name, registry)
                bowler_id = extract_player_id(bowler_name, registry)
                
                runs_info = delivery.get('runs', {})
                batter_runs = runs_info.get('batter', 0)
                total_runs = runs_info.get('total', 0)
                
                extras_info = get_extras_info(delivery)
                wicket = get_wicket_info(delivery)
                is_wicket = wicket is not None
                
                ball_data = {
                    'league': league_name,
                    'match_id': match_id,
                    'season': season,
                    'match_date': match_date,
                    'venue': venue,
                    'team1': team1,
                    'team2': team2,
                    'innings': innings_number,
                    'batting_team': batting_team,
                    'over': over_number,
                    'ball_in_over': ball_number,
                    'batter_name': batter_name,
                    'batter_id': batter_id,
                    'bowler_name': bowler_name,
                    'bowler_id': bowler_id,
                    'batter_runs': batter_runs,
                    'total_runs': total_runs,
                    'is_wicket': is_wicket,
                    'wides': extras_info['wides'],
                    'noballs': extras_info['noballs']
                }
                
                balls.append(ball_data)

    return balls

def main():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    source_dir = data_dir / 'other_t20_data'
    output_file = data_dir / 'global_t20_matches.parquet'
    
    # 1. Load IPL Universe
    ipl_players = load_ipl_player_ids(data_dir / 'ipl_matches_all.parquet')
    
    # 2. Iterate Leagues
    all_balls = []
    
    # Map folder names to League Codes
    league_map = {
        'bbl_json': 'BBL',
        'psl_json': 'PSL',
        'cpl_json': 'CPL',
        'sat_json': 'SA20',
        'ilt_json': 'ILT20',
        'mlc_json': 'MLC',
        'sma_json': 'SMAT',
        'ntb_json': 'T20Blast',
        't20s_json': 'T20I',
        'msl_json': 'MSL', # Mzansi Super League
        'bpl_json': 'BPL'  # Bangladesh Premier League
    }
    
    print(f"\nScanning {source_dir}...")
    
    for folder_name, league_code in league_map.items():
        league_dir = source_dir / folder_name
        # Handle "(1)" suffix in folder names if present (as seen in list_dir)
        # The user's list_dir showed "bbl_json (1)", etc.
        # We should check for both exact match and match with suffix
        if not league_dir.exists():
            # Try finding it with glob
            candidates = list(source_dir.glob(f"{folder_name}*"))
            if candidates:
                league_dir = candidates[0]
            else:
                print(f"⚠ Warning: Directory for {league_code} ({folder_name}) not found. Skipping.")
                continue
                
        json_files = list(league_dir.glob('*.json'))
        print(f"\nProcessing {league_code} ({len(json_files)} files)...")
        
        matches_processed = 0
        matches_kept = 0
        
        for json_file in tqdm(json_files, desc=f"{league_code}"):
            try:
                match_data = load_cricsheet_match(json_file)
                
                # Check if match has any IPL player
                match_players = get_match_players(match_data, match_data.get('info', {}).get('registry', {}))
                
                # Intersection check
                if not match_players.intersection(ipl_players):
                    continue
                    
                matches_kept += 1
                
                # Extract data
                balls = extract_global_ball_data(match_data, league_code)
                all_balls.extend(balls)
                
            except Exception as e:
                # print(f"Error in {json_file.name}: {e}")
                continue
                
        print(f"  ✓ Kept {matches_kept}/{len(json_files)} matches")

    # 3. Save
    print(f"\nTotal global balls extracted: {len(all_balls):,}")
    df = pd.DataFrame(all_balls)
    
    # Basic cleaning
    df['season'] = pd.to_numeric(df['season'], errors='coerce').fillna(0).astype(int)
    
    print(f"Saving to {output_file}...")
    save_dataframe(df, output_file, format='parquet')
    print("✓ Done")

if __name__ == "__main__":
    main()
