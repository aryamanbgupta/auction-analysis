import pandas as pd
from pathlib import Path

def main():
    project_root = Path(__file__).parent.parent.parent
    pool_dir = project_root / 'results' / 'analysis' / 'auction_pool'
    bowlers_path = pool_dir / 'auction_pool_bowlers.csv'
    
    print(f"Loading {bowlers_path}...")
    df = pd.read_csv(bowlers_path, index_col=0)
    
    targets = ["Mohammed Shami", "TA Boult", "DL Chahar", "B Kumar"]
    
    print(f"Total bowlers in pool: {len(df)}")
    
    for t in targets:
        if t in df.index:
            print(f"FOUND: '{t}' is in the pool.")
        else:
            print(f"MISSING: '{t}' is NOT in the pool.")
            # Check for partial match
            matches = [n for n in df.index if t.split()[-1] in str(n)]
            print(f"  Partial matches for '{t}': {matches}")

if __name__ == "__main__":
    main()
