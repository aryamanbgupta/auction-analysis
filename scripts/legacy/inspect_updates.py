
import pandas as pd
from pathlib import Path

def main():
    project_root = Path('.').resolve()
    csv_file = project_root / 'data' / 'updated_players_export.csv'
    
    print(f"Checking {csv_file}...")
    if csv_file.exists():
        try:
            df = pd.read_csv(csv_file)
            print(f"Loaded {len(df)} rows.")
            print("Columns:", df.columns.tolist())
            print("\nSample rows:")
            print(df.head())
            
            # Check for role columns
            if 'new_role_category' in df.columns:
                print("\nValues in new_role_category:")
                print(df['new_role_category'].value_counts())
            
        except Exception as e:
            print(f"Error reading file: {e}")
    else:
        print("File not found.")

if __name__ == "__main__":
    main()
