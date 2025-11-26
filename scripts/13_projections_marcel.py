"""
Calculate Marcel Projections for IPL 2026.

The Marcel projection system (named after the monkey from Friends) is the most basic
baseline forecasting system. It uses:
1. Weighted average of past 3 years (Weights: 5-4-3 for 2025-2024-2023)
2. Regression to the mean (based on sample size)
3. Age adjustment

Formula:
  Projected_Rate = (Weighted_Rate * Reliability) + (League_Avg_Rate * (1 - Reliability))
  Projected_WAR = Projected_Rate * Projected_Playing_Time * Age_Factor
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def load_data():
    """Load historical WAR data."""
    project_root = Path(__file__).parent.parent
    war_dir = project_root / 'results' / '09_vorp_war'
    meta_file = project_root / 'data' / 'player_metadata.csv'
    
    batter_war = pd.read_csv(war_dir / 'batter_war.csv')
    bowler_war = pd.read_csv(war_dir / 'bowler_war.csv')
    metadata = pd.read_csv(meta_file)
    
    return batter_war, bowler_war, metadata

def calculate_weighted_stats(df, metric_col, usage_col, weights):
    """
    Calculate weighted average of a rate statistic.
    
    Args:
        df: DataFrame with season, player_id, metric, usage
        metric_col: Column name for the rate stat (e.g., WAR_per_ball)
        usage_col: Column name for usage (e.g., balls_faced)
        weights: Dictionary of season weights
    """
    # Filter for relevant seasons
    df = df[df['season'].isin(weights.keys())].copy()
    
    # Apply weights
    df['weight'] = df['season'].map(weights)
    
    # Calculate weighted sums
    df['weighted_metric_num'] = df[metric_col] * df[usage_col] * df['weight']
    df['weighted_usage_denom'] = df[usage_col] * df['weight']
    
    # Group by player
    grouped = df.groupby(['player_id', 'player_name']).agg({
        'weighted_metric_num': 'sum',
        'weighted_usage_denom': 'sum',
        usage_col: 'sum',  # Total actual usage (unweighted)
        'season': 'max'    # Most recent season
    }).reset_index()
    
    # Calculate weighted rate
    grouped['weighted_rate'] = grouped['weighted_metric_num'] / grouped['weighted_usage_denom']
    
    # Calculate weighted usage (for regression reliability)
    # Marcel typically projects playing time as: 0.5 * Year_T + 0.1 * Year_T-1
    # Here we'll just use a weighted average of usage for the projection baseline
    grouped['projected_usage'] = grouped['weighted_usage_denom'] / sum(weights.values())
    
    return grouped

def apply_regression_to_mean(df, rate_col, usage_col, league_avg_rate, regression_constant=100):
    """
    Regress rates to the mean based on sample size.
    
    Reliability = Usage / (Usage + Constant)
    Regressed_Rate = (Rate * Reliability) + (Avg * (1 - Reliability))
    """
    df['reliability'] = df[usage_col] / (df[usage_col] + regression_constant)
    df['regressed_rate'] = (df[rate_col] * df['reliability']) + (league_avg_rate * (1 - df['reliability']))
    return df

def apply_aging_adjustment(df, metadata, target_year):
    """
    Apply aging curve adjustment.
    
    Standard Marcel:
    Age < 29: +0.6% per year of age difference
    Age > 29: -0.3% per year of age difference
    """
    # Merge DOB
    df = df.merge(metadata[['player_name', 'dob']], left_on='player_name', right_on='player_name', how='left')
    
    # Calculate Age in target_year
    df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
    df[f'age_{target_year}'] = target_year - df['dob'].dt.year
    
    # Fill missing ages with median (approx 28)
    df[f'age_{target_year}'].fillna(28, inplace=True)
    
    # Calculate Age Factor
    # Using a simplified curve: Peak at 28
    # < 28: Improve
    # > 28: Decline
    
    def get_age_factor(age):
        peak_age = 28
        if age <= peak_age:
            return 1 + (peak_age - age) * 0.006
        else:
            return 1 - (age - peak_age) * 0.003
            
    df['age_factor'] = df[f'age_{target_year}'].apply(get_age_factor)
    
    return df

def run_projections(target_year, weights, output_subdir):
    """Run projections for a specific target year."""
    print(f"\n" + "="*70)
    print(f"MARCEL PROJECTIONS {target_year}")
    print("="*70)
    
    # 1. Load Data
    print("\nLoading data...")
    batter_war, bowler_war, metadata = load_data()
    
    # Rename ID columns for consistency
    batter_war.rename(columns={'batter_id': 'player_id', 'batter_name': 'player_name'}, inplace=True)
    bowler_war.rename(columns={'bowler_id': 'player_id', 'bowler_name': 'player_name'}, inplace=True)
    
    # 2. Calculate League Averages (for regression)
    # Use the previous season as the baseline environment
    baseline_year = target_year - 1
    avg_batter_war_per_ball = batter_war[batter_war['season'] == baseline_year]['WAR_per_ball'].mean()
    avg_bowler_war_per_ball = bowler_war[bowler_war['season'] == baseline_year]['WAR_per_ball'].mean()
    
    print(f"Baseline Year: {baseline_year}")
    print(f"League Avg Batter WAR/ball: {avg_batter_war_per_ball:.5f}")
    print(f"League Avg Bowler WAR/ball: {avg_bowler_war_per_ball:.5f}")
    
    # 3. Process Batters
    print("\nProcessing Batters...")
    bat_proj = calculate_weighted_stats(batter_war, 'WAR_per_ball', 'balls_faced', weights)
    bat_proj = apply_regression_to_mean(bat_proj, 'weighted_rate', 'projected_usage', 
                                      avg_batter_war_per_ball, regression_constant=100)
    bat_proj = apply_aging_adjustment(bat_proj, metadata, target_year)
    
    # Final Projection
    bat_proj[f'projected_war_{target_year}'] = bat_proj['regressed_rate'] * bat_proj['projected_usage'] * bat_proj['age_factor']
    
    # 4. Process Bowlers
    print("\nProcessing Bowlers...")
    bowl_proj = calculate_weighted_stats(bowler_war, 'WAR_per_ball', 'balls_bowled', weights)
    bowl_proj = apply_regression_to_mean(bowl_proj, 'weighted_rate', 'projected_usage', 
                                       avg_bowler_war_per_ball, regression_constant=100)
    bowl_proj = apply_aging_adjustment(bowl_proj, metadata, target_year)
    
    # Final Projection
    bowl_proj[f'projected_war_{target_year}'] = bowl_proj['regressed_rate'] * bowl_proj['projected_usage'] * bowl_proj['age_factor']
    
    # 5. Display Results
    print("\n" + "-"*70)
    print(f"TOP 10 PROJECTED BATTERS ({target_year})")
    print("-"*70)
    cols = ['player_name', f'age_{target_year}', 'projected_usage', 'weighted_rate', 'regressed_rate', f'projected_war_{target_year}']
    print(bat_proj.sort_values(f'projected_war_{target_year}', ascending=False).head(10)[cols].to_string(index=False))
    
    print("\n" + "-"*70)
    print(f"TOP 10 PROJECTED BOWLERS ({target_year})")
    print("-"*70)
    print(bowl_proj.sort_values(f'projected_war_{target_year}', ascending=False).head(10)[cols].to_string(index=False))
    
    # 6. Save Results
    output_dir = Path(__file__).parent.parent / 'results' / '13_projections' / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    bat_proj.to_csv(output_dir / f'batter_projections_{target_year}.csv', index=False)
    bowl_proj.to_csv(output_dir / f'bowler_projections_{target_year}.csv', index=False)
    
    print(f"\n✓ Projections saved to {output_dir}")

def main():
    # 1. Backtest: Predict 2025 using 2022-2024
    # Weights: 2024 (5), 2023 (4), 2022 (3)
    weights_2025 = {2024: 5, 2023: 4, 2022: 3}
    run_projections(target_year=2025, weights=weights_2025, output_subdir='backtest_2025')
    
    # 2. Forecast: Predict 2026 using 2023-2025
    # Weights: 2025 (5), 2024 (4), 2023 (3)
    weights_2026 = {2025: 5, 2024: 4, 2023: 3}
    run_projections(target_year=2026, weights=weights_2026, output_subdir='forecast_2026')

if __name__ == "__main__":
    main()
