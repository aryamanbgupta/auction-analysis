
import pandas as pd
from pathlib import Path

def main():
    project_root = Path('.').resolve()
    metadata_file = project_root / 'data' / 'player_metadata.csv'
    war_file = project_root / 'results' / '11_war_vs_price' / 'war_vs_price_full.csv'
    
    # Load metadata
    meta = pd.read_csv(metadata_file)
    
    # Load WAR to sort by importance
    # If war file doesn't exist, just use metadata
    if war_file.exists():
        war_df = pd.read_csv(war_file)
        # Merge
        df = war_df.merge(meta, left_on='cricwar_name', right_on='player_name', how='left')
        df = df.sort_values('total_WAR', ascending=False)
    else:
        df = meta
        
    print("Top 50 Players by WAR - Role Inspection:")
    cols = ['player_name_x', 'role_category', 'bowling_type', 'playing_role', 'total_WAR']
    # Handle column name differences if merge happened
    if 'player_name_x' not in df.columns:
        cols[0] = 'player_name'
        
    print(df[cols].head(50).to_string(index=False))

if __name__ == "__main__":
    main()
