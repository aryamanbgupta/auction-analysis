
import pandas as pd
from pathlib import Path

def main():
    project_root = Path('.').resolve()
    metadata_file = project_root / 'data' / 'player_metadata.csv'
    
    print(f"Checking {metadata_file}...")
    if metadata_file.exists():
        df = pd.read_csv(metadata_file)
        print(f"Loaded {len(df)} rows.")
        
        if 'role_category' in df.columns:
            print("\nUnique role_category values:")
            print(df['role_category'].unique())
            print("\nValue counts:")
            print(df['role_category'].value_counts())
            
            unknown_count = (df['role_category'] == 'unknown').sum()
            print(f"\nUnknown count: {unknown_count} ({unknown_count/len(df)*100:.1f}%)")
            
            if unknown_count < len(df):
                print("\nSUCCESS: role_category is populated!")
            else:
                print("\nFAILURE: role_category is still all unknown.")
        else:
            print("\n'role_category' column not found!")
            
        print("\nSample rows:")
        print(df[['player_name', 'playing_role', 'role_category']].head(20))
        
        if 'playing_role' in df.columns:
            print("\nUnique playing_role values:")
            print(df['playing_role'].unique())
    else:
        print("Metadata file not found.")

if __name__ == "__main__":
    main()
