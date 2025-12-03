import pandas as pd
from pathlib import Path

def main():
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / 'data'
    
    all_players_path = data_dir / 'all_players_combined.csv'
    new_meta_path = data_dir / 'r_new_player_meta.csv'
    
    print(f"Reading {all_players_path}...")
    df_all = pd.read_csv(all_players_path)
    
    print(f"Reading {new_meta_path}...")
    df_meta = pd.read_csv(new_meta_path)
    
    # Normalize IDs (strip whitespace, ensure string)
    df_all['cricsheet_id'] = df_all['cricsheet_id'].astype(str).str.strip()
    df_meta['cricsheet_id'] = df_meta['cricsheet_id'].astype(str).str.strip()
    
    ids_all = set(df_all['cricsheet_id'])
    ids_meta = set(df_meta['cricsheet_id'])
    
    print(f"Total players in all_players_combined: {len(ids_all)}")
    print(f"Total players in r_new_player_meta: {len(ids_meta)}")
    
    missing_ids = ids_all - ids_meta
    
    print(f"Missing IDs (in all_players but not in new_meta): {len(missing_ids)}")
    
    if missing_ids:
        missing_df = df_all[df_all['cricsheet_id'].isin(missing_ids)].copy()
        output_path = data_dir / 'missing_players_comparison.csv'
        missing_df.to_csv(output_path, index=False)
        print(f"Missing players saved to {output_path}")
        print("First 10 missing players:")
        print(missing_df.head(10))
    else:
        print("All players from all_players_combined are present in r_new_player_meta.")

if __name__ == "__main__":
    main()
