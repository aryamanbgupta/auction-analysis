"""
Generate Global Features for IPL Projections.
1. Calculate Adjusted RAA (RAA * League Factor).
2. Aggregate 'Recent Form' (last 1 year).
3. Merge with existing IPL features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

def load_data():
    """Load necessary datasets."""
    data_dir = Path(__file__).parent.parent / 'data'
    
    # Global Matches
    global_df = pd.read_parquet(data_dir / 'global_t20_matches.parquet')
    
    # League Factors
    factors_df = pd.read_csv(data_dir / 'league_factors.csv')
    
    # Existing IPL Features
    bat_features = pd.read_csv(data_dir / 'ml_features' / 'batter_features_full.csv')
    bowl_features = pd.read_csv(data_dir / 'ml_features' / 'bowler_features_full.csv')
    
    return global_df, factors_df, bat_features, bowl_features

def calculate_adjusted_raa(df, factors_df):
    """Calculate Adjusted RAA using league factors."""
    print("Calculating Adjusted RAA...")
    
    # Merge factors
    # Handle negative factors: Clamp to 0.1 (assume at least some positive correlation)
    factors_df['factor'] = factors_df['factor'].apply(lambda x: max(0.1, x))
    
    # Create dictionary for fast mapping
    factor_map = dict(zip(factors_df['league'], factors_df['factor']))
    
    # Calculate Raw RAA (same simplified logic as before)
    # RAA = Total Runs - League Avg
    league_avgs = df.groupby(['league', 'season'])['total_runs'].mean().to_dict()
    
    # Vectorized RAA calculation
    # We need to map (league, season) to avg
    # Create a tuple index for mapping
    df['league_season'] = list(zip(df['league'], df['season']))
    df['league_avg'] = df['league_season'].map(league_avgs)
    
    df['raw_raa'] = df['total_runs'] - df['league_avg']
    
    # Apply Factor
    df['league_factor'] = df['league'].map(factor_map).fillna(0.1) # Default to 0.1 if unknown
    df['adjusted_raa'] = df['raw_raa'] * df['league_factor']
    
    return df

def generate_recent_form(global_df, ipl_features, role='batter'):
    """
    Generate 'Recent Form' features.
    For each IPL season in ipl_features, calculate weighted avg of Adjusted RAA 
    in the previous calendar year (or previous season).
    """
    print(f"Generating Recent Form for {role}s...")
    
    # Filter global data for this role
    id_col = f'{role}_id'
    if id_col not in global_df.columns:
        return ipl_features # Should not happen if we extracted correctly
        
    # We need to iterate through IPL seasons to define "Recent"
    # For IPL 2024, "Recent" is 2023 season of other leagues?
    # Or strictly date based? Global data has 'match_date'.
    # IPL usually starts in March/April.
    # So "Recent" = Jan 1st of (Year-1) to March 1st of (Year).
    # Let's approximate: "Recent" = Previous Season (Year-1).
    
    # Aggregate global stats by Player-Season
    # We use 'season' column in global_df.
    # Note: BBL 2023/24 is usually labeled '2023' in our extraction.
    # So '2023' global data is good predictor for '2024' IPL.
    
    global_stats = global_df.groupby(['season', id_col]).agg({
        'adjusted_raa': 'sum',
        'match_id': 'count'
    }).rename(columns={'match_id': 'global_balls', 'adjusted_raa': 'global_raa_sum'}).reset_index()
    
    global_stats['global_raa_per_ball'] = global_stats['global_raa_sum'] / global_stats['global_balls']
    
    # We want to merge this into IPL features where ipl_season = global_season + 1
    # e.g. Global 2023 predicts IPL 2024.
    
    global_stats['target_ipl_season'] = global_stats['season'] + 1
    
    # Merge
    # We join on (id_col, season=target_ipl_season)
    merged = ipl_features.merge(
        global_stats[[id_col, 'target_ipl_season', 'global_raa_per_ball', 'global_balls']],
        left_on=[id_col, 'season'],
        right_on=[id_col, 'target_ipl_season'],
        how='left'
    )
    
    # Fill NaNs (No global play in previous year)
    merged['global_raa_per_ball'] = merged['global_raa_per_ball'].fillna(0)
    merged['global_balls'] = merged['global_balls'].fillna(0)
    
    # Drop temp col
    merged = merged.drop(columns=['target_ipl_season'])
    
    return merged

def main():
    global_df, factors_df, bat_features, bowl_features = load_data()
    
    # Calculate Adjusted RAA
    global_df = calculate_adjusted_raa(global_df, factors_df)
    
    # Generate Features
    bat_features_aug = generate_recent_form(global_df, bat_features, 'batter')
    bowl_features_aug = generate_recent_form(global_df, bowl_features, 'bowler')
    
    # Save
    output_dir = Path(__file__).parent.parent / 'data' / 'ml_features'
    bat_features_aug.to_csv(output_dir / 'batter_features_global.csv', index=False)
    bowl_features_aug.to_csv(output_dir / 'bowler_features_global.csv', index=False)
    
    print(f"\nSaved augmented features to {output_dir}")

if __name__ == "__main__":
    main()
