"""
Calculate Marcel Projections for IPL 2026 using Full History Data.
Adapted from scripts/13_projections_marcel.py to use WARprojections data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def load_data():
    """Load historical WAR data from WARprojections pipeline."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    
    # Load full history WAR
    batter_war = pd.read_csv(data_dir / 'batter_war_full_history.csv')
    bowler_war = pd.read_csv(data_dir / 'bowler_war_full_history.csv')
    
    # Load metadata
    meta_path = data_dir / 'player_metadata_full.csv'
    if not meta_path.exists():
        meta_path = data_dir / 'player_metadata.csv'
    metadata = pd.read_csv(meta_path)
    
    return batter_war, bowler_war, metadata

def calculate_weighted_stats(df, metric_col, usage_col, weights):
    """Calculate weighted average of a rate statistic."""
    # Filter for relevant seasons
    df = df[df['season'].isin(weights.keys())].copy()
    
    # Apply weights
    df['weight'] = df['season'].map(weights)
    
    # Calculate weighted sums
    df['weighted_metric_num'] = df[metric_col] * df[usage_col] * df['weight']
    df['weighted_usage_denom'] = df[usage_col] * df['weight']
    
    # Group by player
    # Use id and name
    id_col = 'batter_id' if 'batter_id' in df.columns else 'bowler_id'
    name_col = 'batter_name' if 'batter_name' in df.columns else 'bowler_name'
    
    grouped = df.groupby([id_col, name_col]).agg({
        'weighted_metric_num': 'sum',
        'weighted_usage_denom': 'sum',
        usage_col: 'sum',  # Total actual usage (unweighted)
        'season': 'max'    # Most recent season
    }).reset_index()
    
    # Rename to standard
    grouped.rename(columns={id_col: 'player_id', name_col: 'player_name'}, inplace=True)
    
    # Calculate weighted rate
    grouped['weighted_rate'] = grouped['weighted_metric_num'] / grouped['weighted_usage_denom']
    
    # Calculate weighted usage (for regression reliability)
    grouped['projected_usage'] = grouped['weighted_usage_denom'] / sum(weights.values())
    
    return grouped

def apply_regression_to_mean(df, rate_col, usage_col, league_avg_rate, regression_constant=100):
    """Regress rates to the mean based on sample size."""
    df['reliability'] = df[usage_col] / (df[usage_col] + regression_constant)
    df['regressed_rate'] = (df[rate_col] * df['reliability']) + (league_avg_rate * (1 - df['reliability']))
    return df

def apply_aging_adjustment(df, metadata, target_year):
    """Apply aging curve adjustment."""
    # Merge DOB
    # Metadata usually has 'player_id' or 'id'
    meta_id = 'player_id' if 'player_id' in metadata.columns else 'id'
    
    # Try merging on ID first
    if 'player_id' in df.columns:
        df = df.merge(metadata[[meta_id, 'dob']], left_on='player_id', right_on=meta_id, how='left')
    else:
        # Fallback to name
        df = df.merge(metadata[['player_name', 'dob']], left_on='player_name', right_on='player_name', how='left')
    
    # Calculate Age in target_year
    df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
    df[f'age_{target_year}'] = target_year - df['dob'].dt.year
    
    # Fill missing ages with median (approx 28)
    df[f'age_{target_year}'].fillna(28, inplace=True)
    
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
    print(f"MARCEL PROJECTIONS {target_year} (FULL HISTORY DATA)")
    print("="*70)
    
    # 1. Load Data
    batter_war, bowler_war, metadata = load_data()
    
    # 2. Calculate League Averages (for regression)
    baseline_year = target_year - 1
    
    # Check if baseline year exists in data
    if baseline_year not in batter_war['season'].unique():
        print(f"Warning: Baseline year {baseline_year} not in data. Using mean of all years.")
        avg_batter_war_per_ball = batter_war['RAA_per_ball'].mean() # Using RAA_per_ball as proxy for rate? 
        # Wait, Marcel usually projects the rate stat. WAR is a counting stat.
        # We should project RAA_per_ball (or WAR_per_ball) and then multiply by usage.
        # In 02_calculate_metrics, we have 'RAA_per_ball'. Let's use that.
        # WAR = (RAA - Rep) / RPW. 
        # Simpler to project RAA_per_ball, then convert to WAR?
        # Or just project WAR_per_ball?
        # Let's project WAR directly if we have it? No, WAR depends on replacement level which varies.
        # RAA_per_ball is more intrinsic.
        # Let's project RAA_per_ball.
        pass
    else:
        avg_batter_war_per_ball = batter_war[batter_war['season'] == baseline_year]['RAA_per_ball'].mean()
        avg_bowler_war_per_ball = bowler_war[bowler_war['season'] == baseline_year]['RAA_per_ball'].mean()
    
    print(f"Baseline Year: {baseline_year}")
    print(f"League Avg Batter RAA/ball: {avg_batter_war_per_ball:.5f}")
    
    # 3. Process Batters
    print("\nProcessing Batters...")
    # Using 'RAA_per_ball' as the rate stat
    bat_proj = calculate_weighted_stats(batter_war, 'RAA_per_ball', 'balls_faced', weights)
    bat_proj = apply_regression_to_mean(bat_proj, 'weighted_rate', 'projected_usage', 
                                      avg_batter_war_per_ball, regression_constant=100)
    bat_proj = apply_aging_adjustment(bat_proj, metadata, target_year)
    
    # Final Projection (RAA)
    bat_proj['projected_raa'] = bat_proj['regressed_rate'] * bat_proj['projected_usage'] * bat_proj['age_factor']
    
    # Convert to WAR
    # WAR = (RAA - (RepLevel * Usage)) / RPW
    # We need a replacement level. Let's assume 0 for simplicity or use the one from 02 script?
    # In 02 script, Rep Level was calculated per season.
    # For projection, we can assume Rep Level is similar to last year?
    # Or just project WAR directly? 
    # Let's project WAR directly to be comparable with ML model which predicts WAR.
    # Re-doing with WAR column.
    
    # Actually, ML model predicts 'WAR'.
    # So let's project 'WAR' directly.
    # But WAR is a counting stat. We should project WAR/ball.
    
    # Add WAR_per_ball to input dfs
    batter_war['WAR_per_ball'] = batter_war['WAR'] / batter_war['balls_faced']
    bowler_war['WAR_per_ball'] = bowler_war['WAR'] / bowler_war['balls_bowled']
    
    # Recalculate averages
    if baseline_year in batter_war['season'].unique():
        avg_batter_war_per_ball = batter_war[batter_war['season'] == baseline_year]['WAR_per_ball'].mean()
        avg_bowler_war_per_ball = bowler_war[bowler_war['season'] == baseline_year]['WAR_per_ball'].mean()
    else:
        avg_batter_war_per_ball = batter_war['WAR_per_ball'].mean()
        avg_bowler_war_per_ball = bowler_war['WAR_per_ball'].mean()
        
    # Process Batters (WAR)
    bat_proj = calculate_weighted_stats(batter_war, 'WAR_per_ball', 'balls_faced', weights)
    bat_proj = apply_regression_to_mean(bat_proj, 'weighted_rate', 'projected_usage', 
                                      avg_batter_war_per_ball, regression_constant=100)
    bat_proj = apply_aging_adjustment(bat_proj, metadata, target_year)
    bat_proj[f'projected_war_{target_year}'] = bat_proj['regressed_rate'] * bat_proj['projected_usage'] * bat_proj['age_factor']
    
    # Process Bowlers (WAR)
    print("Processing Bowlers...")
    bowl_proj = calculate_weighted_stats(bowler_war, 'WAR_per_ball', 'balls_bowled', weights)
    bowl_proj = apply_regression_to_mean(bowl_proj, 'weighted_rate', 'projected_usage', 
                                       avg_bowler_war_per_ball, regression_constant=100)
    bowl_proj = apply_aging_adjustment(bowl_proj, metadata, target_year)
    bowl_proj[f'projected_war_{target_year}'] = bowl_proj['regressed_rate'] * bowl_proj['projected_usage'] * bowl_proj['age_factor']
    
    # Save Results
    output_dir = Path(__file__).parent.parent / 'results' / 'WARprojections' / 'marcel'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    bat_proj.to_csv(output_dir / f'batter_projections_{target_year}.csv', index=False)
    bowl_proj.to_csv(output_dir / f'bowler_projections_{target_year}.csv', index=False)
    
    print(f"✓ Saved to {output_dir}")

def main():
    # 1. Backtest: Predict 2025 using 2022-2024
    weights_2025 = {2024: 5, 2023: 4, 2022: 3}
    run_projections(target_year=2025, weights=weights_2025, output_subdir='backtest_2025')
    
    # 2. Forecast: Predict 2026 using 2023-2025
    weights_2026 = {2025: 5, 2024: 4, 2023: 3}
    run_projections(target_year=2026, weights=weights_2026, output_subdir='forecast_2026')

if __name__ == "__main__":
    main()
