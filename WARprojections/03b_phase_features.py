"""
Generate Phase-Specific Features for WAR Projections (v3).

NEW FEATURES:
1. Phase-specific RAA (powerplay, middle, death overs)
2. Batting position metrics
3. Last N matches form
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Phase definitions
POWERPLAY_OVERS = (0, 5)   # Overs 0-5
MIDDLE_OVERS = (6, 15)      # Overs 6-15
DEATH_OVERS = (16, 20)      # Overs 16-20


def load_data():
    """Load necessary datasets."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    
    # Ball-by-ball IPL data
    ipl_df = pd.read_parquet(data_dir / 'ipl_matches_all.parquet')
    
    # Existing features
    bat_features = pd.read_csv(data_dir / 'ml_features' / 'batter_features_full.csv')
    bowl_features = pd.read_csv(data_dir / 'ml_features' / 'bowler_features_full.csv')
    
    return ipl_df, bat_features, bowl_features


def calculate_phase_raa(df, role='batter'):
    """Calculate phase-specific RAA for batters or bowlers."""
    print(f"Calculating phase-specific features for {role}s...")
    
    id_col = f'{role}_id'
    name_col = f'{role}_name'
    
    # Define phase based on over number
    df = df.copy()
    df['phase'] = pd.cut(
        df['over'], 
        bins=[-1, 5, 15, 20], 
        labels=['powerplay', 'middle', 'death']
    )
    
    # Calculate league average runs per ball by phase and season
    phase_avgs = df.groupby(['season', 'phase'], observed=True)['total_runs'].mean().reset_index()
    phase_avgs.rename(columns={'total_runs': 'phase_avg_runs'}, inplace=True)
    
    df = df.merge(phase_avgs, on=['season', 'phase'], how='left')
    
    # RAA = Actual - Expected
    df['phase_raa'] = df['total_runs'] - df['phase_avg_runs']
    
    # For bowlers, negate (fewer runs is better)
    if role == 'bowler':
        df['phase_raa'] = -df['phase_raa']
    
    # Aggregate by player-season-phase
    agg_cols = {
        'phase_raa': 'sum',
        'match_id': 'count'
    }
    
    phase_stats = df.groupby(['season', id_col, name_col, 'phase'], observed=True).agg(agg_cols)
    phase_stats = phase_stats.rename(columns={'match_id': 'phase_balls'}).reset_index()
    
    phase_stats['phase_raa_per_ball'] = phase_stats['phase_raa'] / phase_stats['phase_balls']
    
    # Pivot to get one row per player-season
    pivoted = phase_stats.pivot_table(
        index=['season', id_col, name_col],
        columns='phase',
        values=['phase_raa', 'phase_balls', 'phase_raa_per_ball'],
        aggfunc='first'
    )
    
    # Flatten column names
    pivoted.columns = [f'{val}_{phase}' for val, phase in pivoted.columns]
    pivoted = pivoted.reset_index()
    
    return pivoted


def calculate_last_n_matches_form(df, n_matches=5, role='batter'):
    """Calculate form from last N IPL matches of each season."""
    print(f"Calculating last {n_matches} matches form for {role}s...")
    
    id_col = f'{role}_id'
    name_col = f'{role}_name'
    
    # Get unique matches per season, ordered by date
    match_order = df.groupby(['season', 'match_id'])['match_date'].first().reset_index()
    match_order = match_order.sort_values(['season', 'match_date'])
    
    # Calculate match-level RAA
    df = df.copy()
    league_avg = df.groupby(['season'])['total_runs'].mean().to_dict()
    df['expected_runs'] = df['season'].map(league_avg)
    df['ball_raa'] = df['total_runs'] - df['expected_runs']
    
    if role == 'bowler':
        df['ball_raa'] = -df['ball_raa']
    
    # Aggregate by player-match
    match_stats = df.groupby(['season', 'match_id', id_col, name_col]).agg({
        'ball_raa': 'sum',
        'match_date': 'first'
    }).reset_index()
    
    match_stats = match_stats.sort_values(['season', id_col, 'match_date'])
    
    # For each player-season, get last N matches' avg RAA
    form_data = []
    
    for (season, player_id), group in match_stats.groupby(['season', id_col]):
        # Sort by date and take last N
        group = group.sort_values('match_date')
        last_n = group.tail(n_matches)
        
        form_data.append({
            'season': season,
            id_col: player_id,
            name_col: group[name_col].iloc[0],
            f'last_{n_matches}_matches_raa': last_n['ball_raa'].mean(),
            f'last_{n_matches}_matches_count': len(last_n)
        })
    
    return pd.DataFrame(form_data)


def merge_features(base_features, phase_features, form_features, role='batter'):
    """Merge all features together."""
    id_col = f'{role}_id'
    name_col = f'{role}_name'
    
    # Start with base features
    merged = base_features.copy()
    
    # Merge phase features
    phase_cols = [c for c in phase_features.columns if c not in ['season', id_col, name_col]]
    merged = merged.merge(
        phase_features[['season', id_col] + phase_cols],
        on=['season', id_col],
        how='left'
    )
    
    # Merge form features (lagged by 1 season - form from previous season)
    form_features = form_features.copy()
    form_features['season'] = form_features['season'] + 1  # Lag by 1 year
    
    form_cols = [c for c in form_features.columns if c not in ['season', id_col, name_col]]
    merged = merged.merge(
        form_features[['season', id_col] + form_cols],
        on=['season', id_col],
        how='left',
        suffixes=('', '_lag')
    )
    
    # Fill NaNs
    for col in phase_cols + form_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)
    
    return merged


def main():
    ipl_df, bat_features, bowl_features = load_data()
    
    # Calculate phase-specific features
    bat_phase = calculate_phase_raa(ipl_df, 'batter')
    bowl_phase = calculate_phase_raa(ipl_df, 'bowler')
    
    # Calculate form features
    bat_form = calculate_last_n_matches_form(ipl_df, n_matches=5, role='batter')
    bowl_form = calculate_last_n_matches_form(ipl_df, n_matches=5, role='bowler')
    
    # Merge with base features
    bat_v3 = merge_features(bat_features, bat_phase, bat_form, 'batter')
    bowl_v3 = merge_features(bowl_features, bowl_phase, bowl_form, 'bowler')
    
    # Save
    output_dir = Path(__file__).parent.parent / 'data' / 'ml_features'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    bat_v3.to_csv(output_dir / 'batter_features_v3.csv', index=False)
    bowl_v3.to_csv(output_dir / 'bowler_features_v3.csv', index=False)
    
    print(f"\n✓ Saved v3 features to {output_dir}")
    print(f"  Batters: {len(bat_v3)} rows, new features: {[c for c in bat_v3.columns if 'phase' in c or 'last_5' in c][:6]}...")
    print(f"  Bowlers: {len(bowl_v3)} rows")
    
    # Show sample of new features
    print("\nSample of new batter features:")
    sample_cols = ['batter_name', 'season', 'phase_raa_powerplay', 'phase_raa_death', 'last_5_matches_raa']
    sample_cols = [c for c in sample_cols if c in bat_v3.columns]
    print(bat_v3[bat_v3['season'] == 2024][sample_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
