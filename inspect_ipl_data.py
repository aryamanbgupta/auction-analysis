
import pandas as pd
from pathlib import Path

def main():
    project_root = Path('.').resolve()
    ipl_file = project_root / 'data' / 'ipl_matches.parquet'
    
    print(f"Loading {ipl_file}...")
    df = pd.read_parquet(ipl_file)
    print("Columns:", df.columns.tolist())
    
    # Check for dismissal_type info
    if 'dismissal_type' in df.columns:
        print("\nDismissal types:", df['dismissal_type'].unique())
        print("\nDismissal counts:")
        print(df['dismissal_type'].value_counts())

if __name__ == "__main__":
    main()
