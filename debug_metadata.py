
import pandas as pd
from pathlib import Path
import sys

def main():
    project_root = Path('.').resolve()
    ipl_list_file = project_root / 'data' / 'IPL_2025_Players_List.csv'
    
    print(f"Checking {ipl_list_file}...")
    if ipl_list_file.exists():
        try:
            df = pd.read_csv(ipl_list_file)
            print(f"Loaded {len(df)} rows.")
            print("Columns:", df.columns.tolist())
            print("Sample rows:")
            print(df.head())
            
            # Check for role columns
            role_cols = [c for c in df.columns if 'role' in c.lower() or 'type' in c.lower()]
            if role_cols:
                print("\nPotential role columns:", role_cols)
                for col in role_cols:
                    print(f"\nValues in {col}:")
                    print(df[col].value_counts().head())
        except Exception as e:
            print(f"Error reading file: {e}")
    else:
        print("IPL list file not found.")

    print("\nChecking rpy2...")
    try:
        import rpy2
        print("rpy2 is installed.")
    except ImportError:
        print("rpy2 is NOT installed.")

if __name__ == "__main__":
    main()
