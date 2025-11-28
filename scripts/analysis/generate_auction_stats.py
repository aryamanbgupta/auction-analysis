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

def main():
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / 'results' / '06_context_adjustments' / 'ipl_with_raa.parquet'
    batter_war_path = project_root / 'results' / '09_vorp_war' / 'batter_war.csv'
    bowler_war_path = project_root / 'results' / '09_vorp_war' / 'bowler_war.csv'
    output_dir = project_root / 'results' / 'analysis' / 'auction_stats'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = pd.read_parquet(data_path)
    df['season'] = df['season'].astype(str)
    
    batter_war = pd.read_csv(batter_war_path)
    bowler_war = pd.read_csv(bowler_war_path)
    
    # 2025
    df_2025 = df[df['season'] == '2025']
    b_2025, bo_2025 = calculate_stats(df_2025, batter_war, bowler_war, "2025")
    b_2025.sort_values('WAR', ascending=False).to_csv(output_dir / 'batters_2025.csv')
    bo_2025.sort_values('WAR', ascending=False).to_csv(output_dir / 'bowlers_2025.csv')
    
    # 2022-2025
    seasons = ['2022', '2023', '2024', '2025']
    df_4yr = df[df['season'].isin(seasons)]
    b_4yr, bo_4yr = calculate_stats(df_4yr, batter_war, bowler_war, "2022-2025")
    b_4yr.sort_values('WAR', ascending=False).to_csv(output_dir / 'batters_2022_2025.csv')
    bo_4yr.sort_values('WAR', ascending=False).to_csv(output_dir / 'bowlers_2022_2025.csv')
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()
