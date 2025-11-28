import pandas as pd
import difflib
from pathlib import Path

def normalize_name(name):
    if pd.isna(name):
        return ""
    return str(name).strip().lower()

def main():
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / 'results' / '06_context_adjustments' / 'ipl_with_raa.parquet'
    retention_path = project_root / 'data' / 'IPL_2026_retentions.csv'
    unmatched_path = project_root / 'results' / 'analysis' / 'auction_pool' / 'unmatched_retentions.txt'

    print("Loading data...")
    df = pd.read_parquet(data_path)
    df['season'] = df['season'].astype(str)
    
    # Filter for 2022-2025
    seasons = ['2022', '2023', '2024', '2025']
    df_4yr = df[df['season'].isin(seasons)]
    
    all_batters = df_4yr['batter_name'].unique()
    all_bowlers = df_4yr['bowler_name'].unique()
    dataset_names = list(set(all_batters) | set(all_bowlers))
    
    # Load unmatched list
    with open(unmatched_path, 'r') as f:
        lines = f.readlines()
    unmatched_names = [line.strip() for line in lines if line.strip() and not line.startswith("=") and not line.startswith("The following")]
    
    print(f"Loaded {len(unmatched_names)} unmatched players.")
    
    # Debug specific high-profile failures
    targets = ["Virat Kohli", "Jasprit Bumrah", "Rishabh Pant", "Hardik Pandya", "MS Dhoni"]
    
    print("\n--- Deep Dive Analysis ---")
    for target in targets:
        if target in unmatched_names:
            print(f"\nTarget: '{target}'")
            target_norm = normalize_name(target)
            
            # 1. Check for Last Name match
            target_parts = target_norm.split()
            last_name = target_parts[-1] if target_parts else ""
            
            potential_matches = []
            for name in dataset_names:
                name_norm = normalize_name(name)
                if last_name in name_norm:
                    potential_matches.append(name)
            
            print(f"  Dataset names containing '{last_name}': {potential_matches[:10]}...")
            
            # 2. Difflib closest matches
            closest = difflib.get_close_matches(target_norm, [normalize_name(n) for n in dataset_names], n=3, cutoff=0.4)
            print(f"  Difflib closest matches (cutoff=0.4): {closest}")

if __name__ == "__main__":
    main()
