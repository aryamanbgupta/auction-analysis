import pandas as pd
from pathlib import Path

def normalize_name(name):
    if pd.isna(name):
        return ""
    return str(name).strip().lower()

def get_last_name(name):
    parts = normalize_name(name).split()
    return parts[-1] if parts else ""

def get_first_initial(name):
    parts = normalize_name(name).split()
    return parts[0][0] if parts else ""

def main():
    project_root = Path(__file__).parent.parent.parent
    pool_dir = project_root / 'results' / 'analysis' / 'auction_pool'
    unmatched_path = pool_dir / 'unmatched_retentions.txt'
    batters_path = pool_dir / 'auction_pool_batters.csv'
    bowlers_path = pool_dir / 'auction_pool_bowlers.csv'

    print("Loading data...")
    try:
        with open(unmatched_path, 'r') as f:
            lines = f.readlines()
        unmatched_names = [line.strip() for line in lines if line.strip() and not line.startswith("=") and not line.startswith("The following")]
    except FileNotFoundError:
        print("Unmatched file not found.")
        return

    batters = pd.read_csv(batters_path, index_col=0)
    bowlers = pd.read_csv(bowlers_path, index_col=0)
    
    print(f"Loaded {len(unmatched_names)} unmatched retentions.")
    print(f"Initial Pool: {len(batters)} batters, {len(bowlers)} bowlers.")
    
    removed_players = set()
    
    for r_name in unmatched_names:
        r_last = get_last_name(r_name)
        r_initial = get_first_initial(r_name)
        
        # Check Batters
        for b_name in batters.index:
            b_last = get_last_name(b_name)
            b_parts = normalize_name(b_name).split()
            
            if b_last == r_last:
                # Check if retention initial matches ANY initial in dataset name (if multiple parts)
                # e.g. PHKD Mendis (dataset) vs Kamindu Mendis (retention) -> K is in PHKD
                match_found = False
                if len(b_parts) > 1:
                    # Check first part (initials usually)
                    initials_part = b_parts[0]
                    if r_initial in initials_part:
                        match_found = True
                
                # Fallback: Standard first initial check
                if not match_found and get_first_initial(b_name) == r_initial:
                    match_found = True

                if match_found:
                    removed_players.add(b_name)
                    print(f"Removing Batter: '{b_name}' (Matched retention '{r_name}')")
                else:
                    print(f"Skipping potential match: '{b_name}' vs '{r_name}' (Initials mismatch)")

        # Check Bowlers
        for bo_name in bowlers.index:
            bo_last = get_last_name(bo_name)
            bo_parts = normalize_name(bo_name).split()
            
            if bo_last == r_last:
                match_found = False
                if len(bo_parts) > 1:
                    initials_part = bo_parts[0]
                    if r_initial in initials_part:
                        match_found = True
                
                if not match_found and get_first_initial(bo_name) == r_initial:
                    match_found = True

                if match_found:
                    removed_players.add(bo_name)
                    print(f"Removing Bowler: '{bo_name}' (Matched retention '{r_name}')")
                else:
                    print(f"Skipping potential match: '{bo_name}' vs '{r_name}' (Initials mismatch)")

    # Remove from dataframes
    batters_clean = batters[~batters.index.isin(removed_players)]
    bowlers_clean = bowlers[~bowlers.index.isin(removed_players)]
    
    print(f"Removed {len(removed_players)} players.")
    print(f"Final Pool: {len(batters_clean)} batters, {len(bowlers_clean)} bowlers.")
    
    batters_clean.to_csv(batters_path)
    bowlers_clean.to_csv(bowlers_path)
    print("Updated CSVs saved.")

if __name__ == "__main__":
    main()
