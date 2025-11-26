
import pandas as pd
from pathlib import Path

def main():
    project_root = Path('.').resolve()
    csv_file = project_root / 'data' / 'players_info.csv'
    
    print(f"Checking {csv_file}...")
    if csv_file.exists():
        try:
            df = pd.read_csv(csv_file)
            print(f"Loaded {len(df)} rows.")
            print("Columns:", df.columns.tolist())
            print("\nSample rows:")
            print(df.head())
            
            # Check for role columns
            role_cols = [c for c in df.columns if 'role' in c.lower() or 'style' in c.lower()]
            if role_cols:
                print("\nPotential role columns:", role_cols)
                for col in role_cols:
                    print(f"\nValues in {col}:")
                    print(df[col].value_counts().head())
            
            # Check for player ID columns
            id_cols = [c for c in df.columns if 'id' in c.lower()]
            print("\nID columns:", id_cols)
            
        except Exception as e:
            print(f"Error reading file: {e}")
    else:
        print("File not found.")

if __name__ == "__main__":
    main()
