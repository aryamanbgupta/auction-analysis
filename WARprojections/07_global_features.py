"""
Generate Global Features for IPL Projections (v2).

IMPROVEMENTS:
1. Uses role-specific league factors (batter vs bowler)
2. Handles T20I tier stratification
3. Uses conservative prior (0.3) for unknown leagues
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Same strong teams as in 06_league_strength.py
STRONG_TEAMS = {
    'India', 'Australia', 'England', 'Pakistan', 'South Africa',
    'New Zealand', 'West Indies', 'Sri Lanka', 'Bangladesh', 'Afghanistan'
}

# Conservative prior for unknown leagues
CONSERVATIVE_PRIOR = 0.3


def load_data():
    """Load necessary datasets."""
    data_dir = Path(__file__).parent.parent / 'data'
    
    # Global Matches
    global_df = pd.read_parquet(data_dir / 'global_t20_matches.parquet')
    
    # League Factors (now with 'role' column)
    factors_df = pd.read_csv(data_dir / 'league_factors.csv')
    
    # Existing IPL Features
    bat_features = pd.read_csv(data_dir / 'ml_features' / 'batter_features_full.csv')
    bowl_features = pd.read_csv(data_dir / 'ml_features' / 'bowler_features_full.csv')
    
    return global_df, factors_df, bat_features, bowl_features


def get_t20i_tier(team1, team2):
    """Classify T20I match by opponent strength."""
    if pd.isna(team1) or pd.isna(team2):
        return 'T20I'
    
    teams = {str(team1).strip(), str(team2).strip()}
    strong_count = len(teams & STRONG_TEAMS)
    
    if strong_count == 2:
        return 'T20I_Elite'
    elif strong_count == 1:
        return 'T20I_Mixed'
    else:
        return 'T20I_Associate'


def stratify_t20i(df):
    """Split T20I matches into tiers based on opponent strength."""
    t20i_mask = df['league'].str.upper().str.contains('T20I|T20 INTERNATIONAL', na=False, regex=True)
    
    if t20i_mask.sum() == 0:
        return df
    
    if 'team1' not in df.columns or 'team2' not in df.columns:
        return df
    
    df.loc[t20i_mask, 'league'] = df.loc[t20i_mask].apply(
        lambda row: get_t20i_tier(row.get('team1'), row.get('team2')), axis=1
    )
    
    return df


def calculate_adjusted_raa(df, factors_df, role='batter'):
    """
    Calculate Adjusted RAA using role-specific league factors.
    
    Args:
        df: Global matches dataframe
        factors_df: League factors with 'role' column
        role: 'batter' or 'bowler'
    """
    print(f"Calculating Adjusted RAA for {role}s...")
    
    # Filter factors for this role
    role_factors = factors_df[factors_df['role'] == role].copy()
    
    # Clamp factors to minimum 0.1
    role_factors['factor'] = role_factors['factor'].clip(lower=0.1)
    
    # Create mapping
    factor_map = dict(zip(role_factors['league'], role_factors['factor']))
    
    # Calculate Raw RAA
    league_avgs = df.groupby(['league', 'season'])['total_runs'].mean().to_dict()
    df['league_season'] = list(zip(df['league'], df['season']))
    df['league_avg'] = df['league_season'].map(league_avgs)
    df['raw_raa'] = df['total_runs'] - df['league_avg']
    
    # Apply role-specific factor (default to conservative prior)
    df['league_factor'] = df['league'].map(factor_map).fillna(CONSERVATIVE_PRIOR)
    df['adjusted_raa'] = df['raw_raa'] * df['league_factor']
    
    # For bowlers, negate (lower runs = better)
    if role == 'bowler':
        df['adjusted_raa'] = -df['adjusted_raa']
    
    return df


def generate_recent_form(global_df, ipl_features, role='batter'):
    """
    Generate 'Recent Form' features.
    Global season N predicts IPL season N+1.
    """
    print(f"Generating Recent Form for {role}s...")
    
    id_col = f'{role}_id'
    if id_col not in global_df.columns:
        print(f"  Warning: {id_col} not in global_df. Skipping.")
        return ipl_features
    
    # Aggregate global stats by Player-Season
    global_stats = global_df.groupby(['season', id_col]).agg({
        'adjusted_raa': 'sum',
        'match_id': 'count'
    }).rename(columns={'match_id': 'global_balls', 'adjusted_raa': 'global_raa_sum'}).reset_index()
    
    global_stats['global_raa_per_ball'] = global_stats['global_raa_sum'] / global_stats['global_balls']
    
    # Global season N predicts IPL season N+1
    global_stats['target_ipl_season'] = global_stats['season'] + 1
    
    # Merge
    merged = ipl_features.merge(
        global_stats[[id_col, 'target_ipl_season', 'global_raa_per_ball', 'global_balls']],
        left_on=[id_col, 'season'],
        right_on=[id_col, 'target_ipl_season'],
        how='left'
    )
    
    # Fill NaNs
    merged['global_raa_per_ball'] = merged['global_raa_per_ball'].fillna(0)
    merged['global_balls'] = merged['global_balls'].fillna(0)
    
    # Drop temp col
    if 'target_ipl_season' in merged.columns:
        merged = merged.drop(columns=['target_ipl_season'])
    
    return merged


def main():
    global_df, factors_df, bat_features, bowl_features = load_data()
    
    # Check if factors has 'role' column (new format)
    if 'role' not in factors_df.columns:
        print("Warning: League factors missing 'role' column. Using single factor for all roles.")
        factors_df['role'] = 'batter'  # Fallback
    
    # Stratify T20I matches
    global_df = stratify_t20i(global_df)
    
    # Calculate adjusted RAA for batters
    global_df_bat = calculate_adjusted_raa(global_df.copy(), factors_df, role='batter')
    bat_features_aug = generate_recent_form(global_df_bat, bat_features, 'batter')
    
    # Calculate adjusted RAA for bowlers (using bowler-specific factors)
    global_df_bowl = calculate_adjusted_raa(global_df.copy(), factors_df, role='bowler')
    bowl_features_aug = generate_recent_form(global_df_bowl, bowl_features, 'bowler')
    
    # Save
    output_dir = Path(__file__).parent.parent / 'data' / 'ml_features'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    bat_features_aug.to_csv(output_dir / 'batter_features_global.csv', index=False)
    bowl_features_aug.to_csv(output_dir / 'bowler_features_global.csv', index=False)
    
    print(f"\n✓ Saved augmented features to {output_dir}")
    print(f"  Batters: {len(bat_features_aug)} rows, {bat_features_aug.columns.tolist()[:5]}...")
    print(f"  Bowlers: {len(bowl_features_aug)} rows")


if __name__ == "__main__":
    main()
