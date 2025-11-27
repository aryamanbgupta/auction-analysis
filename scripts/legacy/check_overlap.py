
import pandas as pd
from pathlib import Path

def main():
    project_root = Path('.').resolve()
    ipl_file = project_root / 'data' / 'ipl_matches.parquet'
    new_info_file = project_root / 'data' / 'players_info.csv'
    
    print("Loading datasets...")
    ipl_df = pd.read_parquet(ipl_file)
    info_df = pd.read_csv(new_info_file)
    print("Info columns:", info_df.columns.tolist())
    
    # Get unique IPL players
    batters = ipl_df['batter_name'].unique()
    bowlers = ipl_df['bowler_name'].unique()
    non_strikers = ipl_df['non_striker_name'].unique()
    ipl_players = set(batters) | set(bowlers) | set(non_strikers)
    print(f"Unique IPL players: {len(ipl_players)}")
    
    # Get info players
    # Check 'name' and 'full_name'
    info_names = set(info_df['name'].dropna().unique())
    info_full_names = set(info_df['full_name'].dropna().unique())
    
    # Check overlap
    overlap_name = ipl_players.intersection(info_names)
    overlap_full = ipl_players.intersection(info_full_names)
    
    print(f"Overlap with 'name': {len(overlap_name)}")
    print(f"Overlap with 'unique_name': {len(overlap_full)}")
    
    # Heuristic matching
    print("\nAttempting heuristic matching...")
    
    def parse_name(name):
        parts = name.strip().split()
        if not parts:
            return "", ""
        surname = parts[-1].lower()
        initial = parts[0][0].lower() if parts[0] else ""
        return surname, initial

    # Create lookup from info_df
    # Key: (surname, initial), Value: row
    info_lookup = {}
    for idx, row in info_df.iterrows():
        if pd.isna(row['name']): continue
        s, i = parse_name(row['name'])
        info_lookup[(s, i)] = row
        
        if pd.notna(row['full_name']):
            s, i = parse_name(row['full_name'])
            info_lookup[(s, i)] = row

    matches = 0
    matched_players = []
    
    for player in ipl_players:
        s, i = parse_name(player)
        if (s, i) in info_lookup:
            matches += 1
            matched_players.append(player)
            
    print(f"Heuristic matches: {matches}/{len(ipl_players)}")
    
    missing_heuristic = ipl_players - set(matched_players)
    if len(missing_heuristic) > 0:
        print("\nSample still missing:")
        print(list(missing_heuristic)[:10])
    
if __name__ == "__main__":
    main()
