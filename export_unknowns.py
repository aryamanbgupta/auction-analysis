
import pandas as pd
from pathlib import Path

def main():
    project_root = Path('.').resolve()
    metadata_file = project_root / 'data' / 'player_metadata.csv'
    output_file = project_root / 'data' / 'unknown_players_export.csv'
    
    print(f"Loading {metadata_file}...")
    if not metadata_file.exists():
        print("Error: player_metadata.csv not found.")
        return

    df = pd.read_csv(metadata_file)
    
    # 1. List unique populated categories
    if 'role_category' in df.columns:
        print("\nUnique 'role_category' values populated:")
        print(df['role_category'].unique().tolist())
    
    if 'playing_role' in df.columns:
        print("\nUnique 'playing_role' values (raw source):")
        # Filter out NaNs for cleaner list
        roles = df['playing_role'].dropna().unique().tolist()
        for role in roles:
            print(f"- {role}")
            
    # 2. Export unknown players
    # Unknown is defined as role_category == 'unknown'
    unknowns = df[df['role_category'] == 'unknown'].copy()
    
    print(f"\nFound {len(unknowns)} players with 'unknown' role_category.")
    
    # Select relevant columns for the user to fill in
    cols_to_export = ['player_id', 'player_name']
    if 'playing_role' in df.columns:
        cols_to_export.append('playing_role')
    
    # Add empty column for user to fill
    unknowns['new_role_category'] = ''
    
    export_df = unknowns[cols_to_export + ['new_role_category']]
    
    print(f"Saving to {output_file}...")
    export_df.to_csv(output_file, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
