import pandas as pd
from pathlib import Path

def main():
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / 'results' / '06_context_adjustments' / 'ipl_with_raa.parquet'
    
    print("Loading dataset...")
    df = pd.read_parquet(data_path)
    
    all_bowlers = df['bowler_name'].unique()
    print(f"Total unique bowlers in dataset: {len(all_bowlers)}")
    
    targets = ["Shami", "Boult", "Chahar", "Bumrah", "Kumar"]
    
    print("\n--- Searching for targets in dataset ---")
    for t in targets:
        matches = [n for n in all_bowlers if t.lower() in str(n).lower()]
        print(f"Matches for '{t}': {matches}")
        for m in matches:
            seasons = df[df['bowler_name'] == m]['season'].unique()
            print(f"  Seasons for '{m}': {seasons}")

if __name__ == "__main__":
    main()
