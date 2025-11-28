import pandas as pd
from pathlib import Path
import difflib

def normalize(name):
    return str(name).strip().lower()

def main():
    project_root = Path(__file__).parent.parent.parent
    pool_dir = project_root / 'results' / 'analysis' / 'auction_pool'
    batters_path = pool_dir / 'auction_pool_batters.csv'
    bowlers_path = pool_dir / 'auction_pool_bowlers.csv'
    
    batters = pd.read_csv(batters_path, index_col=0)
    bowlers = pd.read_csv(bowlers_path, index_col=0)
    
    # List of known retained players that might be leaking (based on unmatched list)
    # Format: (Retention Name, Likely Dataset Substring)
    suspects = [
        ("Syed Khaleel Ahmed", "Ahmed"),
        ("Mohd. Arshad Khan", "Arshad"),
        ("Yudhvir Charak", "Yudhvir"),
        ("Md Shami", "Shami"),
        ("Trent Boult", "Boult"),
        ("Deepak Chahar", "Chahar"),
        ("Bhuvneshwar Kumar", "Kumar"),
        ("Jasprit Bumrah", "Bumrah"), # Just in case
        ("Hardik Pandya", "Pandya"),
        ("Rishabh Pant", "Pant"),
        ("Axar Patel", "Patel"),
        ("Kuldeep Yadav", "Kuldeep"),
        ("Tristan Stubbs", "Stubbs"),
        ("Abishek Porel", "Porel"),
        ("Mukesh Kumar", "Mukesh")
    ]
    
    print("--- Checking for Leaking Retained Players ---")
    
    to_remove_batters = []
    to_remove_bowlers = []
    
    for r_name, substring in suspects:
        print(f"\nChecking for '{r_name}' (substring: '{substring}')...")
        
        # Check Batters
        for b_name in batters.index:
            if substring.lower() in b_name.lower():
                print(f"  [Batter Match?] '{b_name}'")
                # Heuristic: If it looks like a match, mark for removal
                # For safety, I'll just print for now, and we can hardcode the removal list.
        
        # Check Bowlers
        for bo_name in bowlers.index:
            if substring.lower() in bo_name.lower():
                print(f"  [Bowler Match?] '{bo_name}'")

if __name__ == "__main__":
    main()
