import pandas as pd
from pathlib import Path

def main():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    
    # Load the full list of players we just extracted
    all_players_path = data_dir / 'all_players_combined.csv'
    if not all_players_path.exists():
        print(f"Error: {all_players_path} not found.")
        return
        
    all_players_df = pd.read_csv(all_players_path)
    print(f"Total players in Cricsheet data: {len(all_players_df)}")
    
    # Load the players info file
    info_path = data_dir / 'players_info.csv'
    if not info_path.exists():
        print(f"Error: {info_path} not found.")
        return
        
    # Read players_info, handling potential bad lines if any, though pandas usually handles csv well
    # The identifier column seems to be the key
    info_df = pd.read_csv(info_path)
    print(f"Total entries in players_info.csv: {len(info_df)}")
    
    # Process identifiers to get the base Cricsheet ID
    # Assuming format is 'cricsheetID' or 'cricsheetID_suffix'
    info_df['cricsheet_id_base'] = info_df['identifier'].astype(str).apply(lambda x: x.split('_')[0])
    
    # Get unique IDs present in info file
    mapped_ids = set(info_df['cricsheet_id_base'].unique())
    
    # Check coverage
    all_players_df['is_mapped'] = all_players_df['cricsheet_id'].isin(mapped_ids)
    
    mapped_count = all_players_df['is_mapped'].sum()
    total_count = len(all_players_df)
    missing_count = total_count - mapped_count
    
    print("-" * 30)
    print(f"Coverage Analysis:")
    print(f"Mapped Players: {mapped_count} ({mapped_count/total_count*100:.2f}%)")
    print(f"Missing Players: {missing_count} ({missing_count/total_count*100:.2f}%)")
    print("-" * 30)
    
    # Save missing players to a file for inspection
    missing_players = all_players_df[~all_players_df['is_mapped']]
    if not missing_players.empty:
        output_missing = data_dir / 'missing_players_info.csv'
        missing_players.to_csv(output_missing, index=False)
        print(f"Saved list of {len(missing_players)} missing players to {output_missing}")
        
        # Show a few examples
        print("\nExample missing players:")
        print(missing_players.head(10))

if __name__ == "__main__":
    main()
