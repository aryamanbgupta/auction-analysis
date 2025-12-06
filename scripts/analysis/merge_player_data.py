import pandas as pd
from pathlib import Path

def main():
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / 'data'
    
    all_players_path = data_dir / 'all_players_combined.csv'
    meta_path = data_dir / 'r_new_player_meta.csv'
    output_path = data_dir / 'all_players_enriched.csv'
    
    print(f"Reading {all_players_path}...")
    df_all = pd.read_csv(all_players_path)
    
    print(f"Reading {meta_path}...")
    df_meta = pd.read_csv(meta_path)
    
    # Normalize IDs to ensure clean merging
    df_all['cricsheet_id'] = df_all['cricsheet_id'].astype(str).str.strip()
    df_meta['cricsheet_id'] = df_meta['cricsheet_id'].astype(str).str.strip()
    
    print(f"Players in base list: {len(df_all)}")
    print(f"Players in metadata: {len(df_meta)}")
    
    # Merge: Left join to keep all players from the base list
    # We want to bring in all columns from meta. 
    # If 'name' exists in both, the merge will create name_x and name_y.
    # We probably want to prefer the metadata name if available, or keep the original.
    # Let's inspect columns first.
    
    print("Merging data...")
    merged_df = pd.merge(df_all, df_meta, on='cricsheet_id', how='left', suffixes=('_original', '_meta'))
    
    # If name_meta is null (no match), fill with name_original
    if 'name_meta' in merged_df.columns and 'name_original' in merged_df.columns:
        merged_df['name'] = merged_df['name_meta'].fillna(merged_df['name_original'])
        # Drop the temporary columns if you want a clean 'name' column
        # But let's keep them or clean up based on preference. 
        # Usually, we want one 'name' column.
        merged_df = merged_df.drop(columns=['name_original', 'name_meta'])
        
        # Reorder to put name and id first
        cols = ['cricsheet_id', 'name'] + [c for c in merged_df.columns if c not in ['cricsheet_id', 'name']]
        merged_df = merged_df[cols]
        
    print(f"Merged dataset size: {len(merged_df)}")
    
    # Check for missing metadata
    missing_meta = merged_df[merged_df['country'].isna()]
    print(f"Players missing metadata after merge: {len(missing_meta)}")
    
    print(f"Saving to {output_path}...")
    merged_df.to_csv(output_path, index=False)
    print("Done.")
    
    # Show sample
    print("\nFirst 5 rows:")
    print(merged_df.head())

if __name__ == "__main__":
    main()
