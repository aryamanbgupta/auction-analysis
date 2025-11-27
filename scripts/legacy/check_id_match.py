
import pandas as pd
from pathlib import Path

def main():
    project_root = Path('.').resolve()
    ipl_file = project_root / 'data' / 'ipl_matches.parquet'
    new_info_file = project_root / 'data' / 'players_info.csv'
    
    print("Loading datasets...")
    ipl_df = pd.read_parquet(ipl_file)
    info_df = pd.read_csv(new_info_file)
    
    # Clean column names
    info_df.columns = info_df.columns.str.strip()
    
    # Get unique IPL player IDs
    # Assuming batter_id, bowler_id are the Cricsheet IDs
    batter_ids = ipl_df['batter_id'].unique()
    bowler_ids = ipl_df['bowler_id'].unique()
    non_striker_ids = ipl_df['non_striker_id'].unique()
    
    ipl_ids = set(batter_ids) | set(bowler_ids) | set(non_striker_ids)
    # Filter out NaNs
    ipl_ids = {x for x in ipl_ids if pd.notna(x)}
    
    print(f"Unique IPL player IDs: {len(ipl_ids)}")
    print(f"Sample IPL IDs: {list(ipl_ids)[:5]}")
    
    # Get info player IDs
    if 'identifier' in info_df.columns:
        info_ids = set(info_df['identifier'].unique())
        print(f"Unique Info IDs: {len(info_ids)}")
        print(f"Sample Info IDs: {list(info_ids)[:5]}")
        
        # Check overlap
        # Ensure types match (str vs int)
        # Convert both to strings for comparison
        ipl_ids_str = {str(x).replace('.0', '') for x in ipl_ids}
        info_ids_str = {str(x).replace('.0', '') for x in info_ids}
        
        overlap = ipl_ids_str.intersection(info_ids_str)
        print(f"Overlap: {len(overlap)} / {len(ipl_ids_str)}")
        
        # Check data quality for matched players
        matched_df = info_df[info_df['identifier'].astype(str).isin(overlap)]
        print(f"\nMatched DataFrame size: {len(matched_df)}")
        
        if 'playing_roles' in matched_df.columns:
            print("\nPlaying Roles distribution for matched players:")
            print(matched_df['playing_roles'].value_counts())
            
            not_found = (matched_df['playing_roles'] == 'Not found').sum()
            print(f"\n'Not found' count: {not_found} ({not_found/len(matched_df)*100:.1f}%)")
        
        if len(overlap) < len(ipl_ids_str):
            missing = ipl_ids_str - overlap
            print(f"\nMissing IDs: {len(missing)}")
            print(f"Sample missing: {list(missing)[:5]}")
    else:
        print("'identifier' column not found in players_info.csv")

if __name__ == "__main__":
    main()
