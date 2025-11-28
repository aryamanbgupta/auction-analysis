import pandas as pd
from pathlib import Path

def main():
    project_root = Path(__file__).parent.parent.parent
    pool_dir = project_root / 'results' / 'analysis' / 'auction_pool'
    batters_path = pool_dir / 'auction_pool_batters.csv'
    bowlers_path = pool_dir / 'auction_pool_bowlers.csv'
    
    batters = pd.read_csv(batters_path, index_col=0)
    bowlers = pd.read_csv(bowlers_path, index_col=0)
    
    # Exact names to remove (as found in the dataset)
    to_remove = [
        "KK Ahmed",       # Khaleel Ahmed (Retained by DC)
        "Arshad Khan",    # Mohd. Arshad Khan (Retained by MI)
        "Yudhvir Singh",  # Yudhvir Charak (Retained by LSG?)
        "Vijaykumar Vyshak" # Vyshak Vijaykumar (Retained by RCB) - Wait, let's check if he is retained.
        # Unmatched list had "Vyshak Vijaykumar". Dataset has "Vijaykumar Vyshak".
        # Check if he is retained. `unmatched_retentions.txt` says yes.
    ]
    
    # Add Vijaykumar Vyshak if he is in the pool
    # My leak check showed "Vijaykumar Vyshak" in the pool.
    # And "Vyshak Vijaykumar" was in unmatched list.
    # So he IS retained and IS in the pool. Remove him.
    to_remove.append("Vijaykumar Vyshak")

    print(f"Removing {len(to_remove)} players: {to_remove}")
    
    # Remove from Batters
    batters_clean = batters[~batters.index.isin(to_remove)]
    print(f"Batters: {len(batters)} -> {len(batters_clean)}")
    
    # Remove from Bowlers
    bowlers_clean = bowlers[~bowlers.index.isin(to_remove)]
    print(f"Bowlers: {len(bowlers)} -> {len(bowlers_clean)}")
    
    batters_clean.to_csv(batters_path)
    bowlers_clean.to_csv(bowlers_path)
    print("Updated CSVs saved.")

if __name__ == "__main__":
    main()
