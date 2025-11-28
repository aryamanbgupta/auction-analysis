import pandas as pd
import numpy as np
from pathlib import Path

def analyze_dc_2025():
    # Paths
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / 'results' / '06_context_adjustments' / 'ipl_with_raa.parquet'
    batter_war_path = project_root / 'results' / '09_vorp_war' / 'batter_war.csv'
    bowler_war_path = project_root / 'results' / '09_vorp_war' / 'bowler_war.csv'
    output_dir = project_root / 'results' / 'analysis' / 'dc_2025'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = pd.read_parquet(data_path)
    
    # Filter for 2025 and Delhi Capitals
    df_2025 = df[df['season'] == '2025']
    
    # --- BATTER ANALYSIS ---
    print("Analyzing Batters...")
    dc_batting = df_2025[df_2025['batting_team'] == 'Delhi Capitals'].copy()
    
    # Basic Stats & RAA
    batter_stats = dc_batting.groupby('batter_name').agg({
        'batter_runs': 'sum',
        'ball_in_over': 'count', # Total balls faced
        'is_wicket': lambda x: x.sum(), # Total dismissals (approx for avg)
        'batter_RAA': 'sum'
    }).rename(columns={'ball_in_over': 'balls_faced', 'is_wicket': 'outs', 'batter_RAA': 'RAA_total'})
    
    batter_stats['SR'] = (batter_stats['batter_runs'] / batter_stats['balls_faced'] * 100).round(1)
    batter_stats['Avg'] = (batter_stats['batter_runs'] / batter_stats['outs'].replace(0, 1)).round(1)
    
    # Splits: Phase
    phases = ['powerplay', 'middle', 'death']
    for phase in phases:
        phase_df = dc_batting[dc_batting['phase'] == phase]
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

    # Splits: Bowling Type (Pace vs Spin)
    # Using 'bowling_type' column directly as it contains 'pace' and 'spin'
    for btype in ['pace', 'spin']:
        type_df = dc_batting[dc_batting['bowling_type'] == btype]
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

    # Merge WAR/VORP
    if batter_war_path.exists():
        war_df = pd.read_csv(batter_war_path)
        war_df_2025 = war_df[war_df['season'] == 2025][['batter_name', 'VORP', 'WAR']]
        batter_stats = batter_stats.join(war_df_2025.set_index('batter_name'), how='left')

    # --- BOWLER ANALYSIS ---
    print("Analyzing Bowlers...")
    dc_bowling = df_2025[df_2025['bowling_team'] == 'Delhi Capitals'].copy()
    
    bowler_stats = dc_bowling.groupby('bowler_name').agg({
        'is_wicket': 'sum',
        'total_runs': 'sum',
        'ball_in_over': 'count',
        'bowler_RAA': 'sum'
    }).rename(columns={'is_wicket': 'Wickets', 'total_runs': 'Runs_Conceded', 'ball_in_over': 'Balls_Bowled', 'bowler_RAA': 'RAA_total'})
    
    bowler_stats['Econ'] = (bowler_stats['Runs_Conceded'] / (bowler_stats['Balls_Bowled'] / 6)).round(2)
    bowler_stats['Avg'] = (bowler_stats['Runs_Conceded'] / bowler_stats['Wickets'].replace(0, 1)).round(1)
    
    # Splits: Phase
    for phase in phases:
        phase_df = dc_bowling[dc_bowling['phase'] == phase]
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

    # Merge WAR/VORP
    if bowler_war_path.exists():
        war_df = pd.read_csv(bowler_war_path)
        war_df_2025 = war_df[war_df['season'] == 2025][['bowler_name', 'VORP', 'WAR']]
        bowler_stats = bowler_stats.join(war_df_2025.set_index('bowler_name'), how='left')

    # Save Results
    print(f"Saving results to {output_dir}...")
    batter_stats.sort_values('WAR', ascending=False).to_csv(output_dir / 'dc_batters_2025.csv')
    bowler_stats.sort_values('WAR', ascending=False).to_csv(output_dir / 'dc_bowlers_2025.csv')
    
    # Print Summary
    print("\nTop DC Batters (by WAR):")
    print(batter_stats.sort_values('WAR', ascending=False)[['WAR', 'RAA_total', 'Runs_powerplay', 'Runs_middle', 'Runs_death']].head().to_string())
    
    print("\nTop DC Bowlers (by WAR):")
    print(bowler_stats.sort_values('WAR', ascending=False)[['WAR', 'RAA_total', 'Wickets_powerplay', 'Wickets_middle', 'Wickets_death']].head().to_string())

if __name__ == "__main__":
    analyze_dc_2025()
