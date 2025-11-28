import pandas as pd
import numpy as np
from pathlib import Path

def calculate_stats(df, batter_war_df, bowler_war_df, season_label):
    print(f"Calculating stats for {season_label}...")
    
    # Identify seasons in this slice
    seasons = df['season'].unique().astype(str)
    
    # --- BATTER ANALYSIS ---
    print("  Analyzing Batters...")
    batter_stats = df.groupby('batter_name').agg({
        'batter_runs': 'sum',
        'ball_in_over': 'count', 
        'is_wicket': lambda x: x.sum(), 
        'batter_RAA': 'sum'
    }).rename(columns={'ball_in_over': 'balls_faced', 'is_wicket': 'outs', 'batter_RAA': 'RAA_total'})
    
    batter_stats['SR'] = (batter_stats['batter_runs'] / batter_stats['balls_faced'] * 100).round(1)
    batter_stats['Avg'] = (batter_stats['batter_runs'] / batter_stats['outs'].replace(0, 1)).round(1)
    
    # Splits: Phase
    phases = ['powerplay', 'middle', 'death']
    for phase in phases:
        phase_df = df[df['phase'] == phase]
        phase_stats = phase_df.groupby('batter_name').agg({
            'batter_runs': 'sum',
            'ball_in_over': 'count',
            'batter_RAA': 'sum'
        }).rename(columns={
            'batter_runs': f'Runs_{phase}',
            'ball_in_over': f'Balls_{phase}',
            'batter_RAA': f'RAA_{phase}'
        })
        phase_stats[f'SR_{phase}'] = (phase_stats[f'Runs_{phase}'] / phase_stats[f'Balls_{phase}'] * 100).round(1)
        batter_stats = batter_stats.join(phase_stats, how='left').fillna(0)

    # Splits: Bowling Type
    for btype in ['pace', 'spin']:
        type_df = df[df['bowling_type'] == btype]
        type_stats = type_df.groupby('batter_name').agg({
            'batter_runs': 'sum',
            'ball_in_over': 'count',
            'batter_RAA': 'sum'
        }).rename(columns={
            'batter_runs': f'Runs_{btype}',
            'ball_in_over': f'Balls_{btype}',
            'batter_RAA': f'RAA_{btype}'
        })
        type_stats[f'SR_{btype}'] = (type_stats[f'Runs_{btype}'] / type_stats[f'Balls_{btype}'] * 100).round(1)
        batter_stats = batter_stats.join(type_stats, how='left').fillna(0)

    # WAR/VORP
    batter_war_df['season'] = batter_war_df['season'].astype(str)
    war_subset = batter_war_df[batter_war_df['season'].isin(seasons)]
    war_agg = war_subset.groupby('batter_name')[['VORP', 'WAR']].sum()
    batter_stats = batter_stats.join(war_agg, how='left').fillna(0)

    # --- BOWLER ANALYSIS ---
    print("  Analyzing Bowlers...")
    bowler_stats = df.groupby('bowler_name').agg({
        'is_wicket': 'sum',
        'total_runs': 'sum',
        'ball_in_over': 'count',
        'bowler_RAA': 'sum'
    }).rename(columns={'is_wicket': 'Wickets', 'total_runs': 'Runs_Conceded', 'ball_in_over': 'Balls_Bowled', 'bowler_RAA': 'RAA_total'})
    
    bowler_stats['Econ'] = (bowler_stats['Runs_Conceded'] / (bowler_stats['Balls_Bowled'] / 6)).round(2)
    bowler_stats['Avg'] = (bowler_stats['Runs_Conceded'] / bowler_stats['Wickets'].replace(0, 1)).round(1)
    
    # Splits: Phase
    for phase in phases:
        phase_df = df[df['phase'] == phase]
        phase_stats = phase_df.groupby('bowler_name').agg({
            'is_wicket': 'sum',
            'total_runs': 'sum',
            'ball_in_over': 'count',
            'bowler_RAA': 'sum'
        }).rename(columns={
            'is_wicket': f'Wickets_{phase}',
            'total_runs': f'Runs_{phase}',
            'ball_in_over': f'Balls_{phase}',
            'bowler_RAA': f'RAA_{phase}'
        })
        phase_stats[f'Econ_{phase}'] = (phase_stats[f'Runs_{phase}'] / (phase_stats[f'Balls_{phase}'] / 6)).round(2)
        bowler_stats = bowler_stats.join(phase_stats, how='left').fillna(0)

    # WAR/VORP
    bowler_war_df['season'] = bowler_war_df['season'].astype(str)
    war_subset_b = bowler_war_df[bowler_war_df['season'].isin(seasons)]
    war_agg_b = war_subset_b.groupby('bowler_name')[['VORP', 'WAR']].sum()
    bowler_stats = bowler_stats.join(war_agg_b, how='left').fillna(0)

    return batter_stats, bowler_stats

import difflib

def normalize_name(name):
    if pd.isna(name):
        return ""
    return str(name).strip().lower()

def get_best_match(retention_name, dataset_names, cutoff=0.6):
    """
    Find best match for retention_name in dataset_names using fuzzy matching.
    Returns (match_name, score) or (None, 0).
    """
    r_norm = normalize_name(retention_name)
    r_parts = r_norm.split()
    
    # 1. Exact match check (normalized)
    for d_name in dataset_names:
        if normalize_name(d_name) == r_norm:
            return d_name, 1.0
            
    # 2. Strong Heuristic: Last Name + First Initial
    # This handles "Virat Kohli" <-> "V Kohli", "Hardik Pandya" <-> "HH Pandya"
    best_heuristic_match = None
    best_heuristic_score = 0.0
    
    for d_name in dataset_names:
        d_norm = normalize_name(d_name)
        d_parts = d_norm.split()
        
        # Must have at least 2 parts (First Last) to be safe
        if len(d_parts) >= 2 and len(r_parts) >= 2:
            # Last name MUST match exactly
            if d_parts[-1] == r_parts[-1]:
                # First initial MUST match
                if d_parts[0][0] == r_parts[0][0]:
                    # This is a very strong candidate.
                    # Calculate fuzzy score to break ties (e.g. "H Pandya" vs "HH Pandya")
                    score = difflib.SequenceMatcher(None, r_norm, d_norm).ratio()
                    
                    # Boost score for this heuristic match because structural match is strong
                    # But keep it relative to other heuristic matches
                    if score > best_heuristic_score:
                        best_heuristic_score = score
                        best_heuristic_match = d_name
    
    if best_heuristic_match:
        return best_heuristic_match, 1.0 # Treat as confident match

    # 3. Fuzzy Match (Fallback)
    # Create a map of normalized -> original
    norm_map = {normalize_name(n): n for n in dataset_names}
    possibilities = list(norm_map.keys())
    
    matches = difflib.get_close_matches(r_norm, possibilities, n=1, cutoff=cutoff)
    
    if matches:
        best_match_norm = matches[0]
        return norm_map[best_match_norm], difflib.SequenceMatcher(None, r_norm, best_match_norm).ratio()
        
    return None, 0.0

def main():
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / 'results' / '06_context_adjustments' / 'ipl_with_raa.parquet'
    batter_war_path = project_root / 'results' / '09_vorp_war' / 'batter_war.csv'
    bowler_war_path = project_root / 'results' / '09_vorp_war' / 'bowler_war.csv'
    retention_path = project_root / 'data' / 'IPL_2026_retentions.csv'
    output_dir = project_root / 'results' / 'analysis' / 'auction_pool'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = pd.read_parquet(data_path)
    df['season'] = df['season'].astype(str)
    
    batter_war = pd.read_csv(batter_war_path)
    bowler_war = pd.read_csv(bowler_war_path)
    
    retentions = pd.read_csv(retention_path)
    retention_names = retentions['Player'].dropna().unique()
    print(f"Loaded {len(retention_names)} retained players from file.")

    # Filter for 2022-2025
    seasons = ['2022', '2023', '2024', '2025']
    df_4yr = df[df['season'].isin(seasons)]
    
    # Identify retained players in the dataset
    all_batters = df_4yr['batter_name'].unique()
    all_bowlers = df_4yr['bowler_name'].unique()
    all_dataset_players = list(set(all_batters) | set(all_bowlers))
    
    retained_in_dataset = set()
    unmatched_retentions = []
    
    print("Matching retained players to dataset names (Fuzzy Match)...")
    
    # Debug targets
    debug_targets = ["Tilak Verma", "Varun Chakaravarthy", "Tilak Varma", "CV Varun"]
    
    for r_name in retention_names:
        # Try to find r_name in dataset
        match, score = get_best_match(r_name, all_dataset_players, cutoff=0.8)
        
        # Special manual overrides if fuzzy fails or for known tricky ones
        if not match:
             # Manual checks for known issues
             r_norm = normalize_name(r_name)
             if "chakaravarthy" in r_norm: # Handle Varun Chakaravarthy -> CV Varun
                 # CV Varun is likely the name
                 if "CV Varun" in all_dataset_players:
                     match = "CV Varun"
                     score = 1.0
             elif "tilak verma" in r_norm:
                 if "Tilak Varma" in all_dataset_players:
                     match = "Tilak Varma"
                     score = 1.0
        
        if match:
            retained_in_dataset.add(match)
            print(f"Matched '{r_name}' -> '{match}' (Score: {score:.2f})")
            if r_name in debug_targets or match in debug_targets:
                print(f"  MATCH: '{r_name}' -> '{match}' (Score: {score:.2f})")
        else:
            unmatched_retentions.append(r_name)
            # print(f"  NO MATCH: '{r_name}'")

    print(f"Identified {len(retained_in_dataset)} retained players in the dataset.")
    print(f"Unmatched retained players: {len(unmatched_retentions)}")
    
    # Save unmatched report
    with open(output_dir / 'unmatched_retentions.txt', 'w') as f:
        f.write("The following retained players could not be found in the 2022-2025 dataset:\n")
        f.write("==========================================================================\n")
        for name in sorted(unmatched_retentions):
            f.write(f"{name}\n")
    print(f"Saved unmatched report to {output_dir / 'unmatched_retentions.txt'}")
    
    # Calculate stats for ALL players first
    print("Calculating aggregated stats for 2022-2025...")
    b_stats, bo_stats = calculate_stats(df_4yr, batter_war, bowler_war, "2022-2025")
    
    # Filter out retained players
    print("Filtering out retained players...")
    
    # For Batters
    auction_batters = b_stats[~b_stats.index.isin(retained_in_dataset)]
    print(f"Batters: {len(b_stats)} -> {len(auction_batters)} (Removed {len(b_stats) - len(auction_batters)})")
    
    # For Bowlers
    auction_bowlers = bo_stats[~bo_stats.index.isin(retained_in_dataset)]
    print(f"Bowlers: {len(bo_stats)} -> {len(auction_bowlers)} (Removed {len(bo_stats) - len(auction_bowlers)})")
    
    # Save
    auction_batters.sort_values('WAR', ascending=False).to_csv(output_dir / 'auction_pool_batters.csv')
    auction_bowlers.sort_values('WAR', ascending=False).to_csv(output_dir / 'auction_pool_bowlers.csv')
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()
