import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Venue Keywords for Home/Away mapping
VENUE_KEYWORDS = {
    'Chennai Super Kings': ['Chidambaram'],
    'Royal Challengers Bengaluru': ['Chinnaswamy'],
    'Royal Challengers Bangalore': ['Chinnaswamy'],
    'Mumbai Indians': ['Wankhede', 'Brabourne', 'DY Patil'],
    'Kolkata Knight Riders': ['Eden Gardens'],
    'Delhi Capitals': ['Arun Jaitley', 'Feroz Shah Kotla'],
    'Punjab Kings': ['Mohali', 'Bindra', 'Mullanpur', 'Dharamshala'],
    'Rajasthan Royals': ['Sawai Mansingh', 'Guwahati'],
    'Sunrisers Hyderabad': ['Rajiv Gandhi'],
    'Lucknow Super Giants': ['Ekana'],
    'Gujarat Titans': ['Narendra Modi', 'Motera']
}

def load_data():
    """Load WAR data, ball-by-ball data, and player metadata."""
    project_root = Path(__file__).parent.parent
    results_dir = project_root / 'results' / '09_vorp_war'
    data_dir = project_root / 'data'
    context_dir = project_root / 'results' / '06_context_adjustments'
    
    batter_war = pd.read_csv(results_dir / 'batter_war.csv')
    bowler_war = pd.read_csv(results_dir / 'bowler_war.csv')
    
    # Load ball-by-ball data for granular metrics
    ball_data = pd.read_parquet(context_dir / 'ipl_with_raa.parquet')
    
    # Try to load full metadata first
    full_meta_path = data_dir / 'player_metadata_full.csv'
    if full_meta_path.exists():
        metadata = pd.read_csv(full_meta_path)
        print(f"Loaded metadata from {full_meta_path}")
    else:
        metadata = pd.read_csv(data_dir / 'player_metadata.csv')
        print(f"Loaded metadata from {data_dir / 'player_metadata.csv'}")
    
    # Ensure season is int
    batter_war['season'] = batter_war['season'].astype(int)
    bowler_war['season'] = bowler_war['season'].astype(int)
    ball_data['season'] = ball_data['season'].astype(int)
    
    return batter_war, bowler_war, ball_data, metadata

def calculate_age(dob_str, target_year):
    """Calculate age in years for a target season."""
    if pd.isna(dob_str):
        return np.nan
    try:
        dob = pd.to_datetime(dob_str)
        # Approximate age at start of IPL (April 1st)
        return target_year - dob.year
    except:
        return np.nan

def is_home_match(row, role):
    """Determine if the match is at home for the player's team."""
    venue = str(row['venue'])
    
    if role == 'batter':
        team = row['batting_team']
    else:
        team = row['bowling_team']
        
    keywords = VENUE_KEYWORDS.get(team, [])
    for kw in keywords:
        if kw in venue:
            return 1
    return 0

def calculate_advanced_metrics(ball_data, role):
    """
    Calculate Consistency (Std Dev of RAA) and Home/Away Splits.
    
    Args:
        ball_data: Ball-by-ball DataFrame
        role: 'batter' or 'bowler'
        
    Returns:
        DataFrame with advanced metrics per player-season
    """
    print(f"Calculating advanced metrics for {role}s...")
    
    # Define ID and RAA column
    id_col = 'batter_id' if role == 'batter' else 'bowler_id'
    raa_col = 'batter_RAA' if role == 'batter' else 'bowler_RAA'
    
    # 1. Consistency Score (Std Dev of Match RAA)
    # First, aggregate RAA by match
    match_raa = ball_data.groupby(['season', id_col, 'match_id'])[raa_col].sum().reset_index()
    
    # Calculate Std Dev per season
    consistency = match_raa.groupby(['season', id_col])[raa_col].std().reset_index()
    consistency.rename(columns={raa_col: 'consistency_score'}, inplace=True)
    
    # 2. Home/Away Split
    # Mark home matches
    ball_data['is_home'] = ball_data.apply(lambda x: is_home_match(x, role), axis=1)
    
    # Aggregate RAA and Balls by Home/Away
    ha_stats = ball_data.groupby(['season', id_col, 'is_home']).agg({
        raa_col: 'sum',
        'match_id': 'count' 
    }).rename(columns={'match_id': 'balls', raa_col: 'RAA'}).reset_index()
    
    # Pivot to get Home and Away columns
    ha_pivot = ha_stats.pivot_table(
        index=['season', id_col], 
        columns='is_home', 
        values=['RAA', 'balls'], 
        fill_value=0
    ).reset_index()
    
    # Flatten columns
    ha_pivot.columns = ['season', id_col, 'balls_away', 'balls_home', 'RAA_away', 'RAA_home']
    
    # Calculate RAA per ball
    ha_pivot['RAA_per_ball_home'] = ha_pivot['RAA_home'] / ha_pivot['balls_home'].replace(0, np.nan)
    ha_pivot['RAA_per_ball_away'] = ha_pivot['RAA_away'] / ha_pivot['balls_away'].replace(0, np.nan)
    
    # Calculate Diff (Home - Away)
    # Fill NaN with 0 (if no home or no away games, diff is 0)
    ha_pivot['home_advantage'] = (ha_pivot['RAA_per_ball_home'] - ha_pivot['RAA_per_ball_away']).fillna(0)
    
    # Merge Consistency and Home/Away
    advanced_features = pd.merge(consistency, ha_pivot, on=['season', id_col], how='outer')
    
    return advanced_features

def create_lagged_features(df, id_col, name_col, metric_cols, lags=[1, 2]):
    """Create lagged features for time-series data."""
    df = df.copy()
    df = df.sort_values(['season', id_col])
    
    feature_df = df.copy()
    
    for lag in lags:
        lag_df = df[['season', id_col] + metric_cols].copy()
        lag_df['season'] = lag_df['season'] + lag
        
        rename_dict = {col: f'{col}_lag{lag}' for col in metric_cols}
        lag_df = lag_df.rename(columns=rename_dict)
        
        feature_df = pd.merge(feature_df, lag_df, on=['season', id_col], how='left')
        
    return feature_df

def add_metadata_features(df, metadata, id_col):
    """Add age and other metadata features."""
    cols_to_use = ['player_id']
    if 'dob' in metadata.columns:
        cols_to_use.append('dob')
    if 'batting_hand' in metadata.columns:
        cols_to_use.append('batting_hand')
    if 'bowling_type' in metadata.columns:
        cols_to_use.append('bowling_type')
        
    df = pd.merge(df, metadata[cols_to_use], 
                  left_on=id_col, right_on='player_id', how='left')
    
    if 'dob' in df.columns:
        df['age'] = df.apply(lambda x: calculate_age(x['dob'], x['season']), axis=1)
    else:
        df['age'] = np.nan
    
    if df['age'].isnull().any():
        if df['age'].isnull().all():
            df['age'] = 27
        else:
            df['age'] = df['age'].fillna(df['age'].median())
        
    return df

def main():
    print("Loading data...")
    batter_war, bowler_war, ball_data, metadata = load_data()
    
    # --- Process Batters ---
    print("Processing Batters...")
    
    # Calculate Advanced Metrics
    bat_advanced = calculate_advanced_metrics(ball_data, 'batter')
    
    # Merge Advanced Metrics into WAR data
    batter_war = pd.merge(batter_war, bat_advanced, on=['season', 'batter_id'], how='left')
    
    # Metrics to lag (Added new metrics)
    bat_metrics = ['WAR', 'RAA', 'balls_faced', 'WAR_per_ball', 'consistency_score', 'home_advantage']
    
    bat_features = create_lagged_features(
        batter_war, 
        id_col='batter_id', 
        name_col='batter_name',
        metric_cols=bat_metrics,
        lags=[1, 2]
    )
    
    bat_features = add_metadata_features(bat_features, metadata, 'batter_id')
    bat_features['recent_balls'] = bat_features['balls_faced_lag1'].fillna(0) + bat_features['balls_faced_lag2'].fillna(0)
    
    output_dir = Path(__file__).parent.parent / 'data' / 'ml_features'
    output_dir.mkdir(exist_ok=True)
    
    bat_features.to_csv(output_dir / 'batter_features.csv', index=False)
    print(f"Saved batter features to {output_dir / 'batter_features.csv'} ({len(bat_features)} rows)")
    
    # --- Process Bowlers ---
    print("Processing Bowlers...")
    
    # Calculate Advanced Metrics
    bowl_advanced = calculate_advanced_metrics(ball_data, 'bowler')
    
    # Merge Advanced Metrics into WAR data
    bowler_war = pd.merge(bowler_war, bowl_advanced, on=['season', 'bowler_id'], how='left')
    
    bowl_metrics = ['WAR', 'RAA', 'balls_bowled', 'WAR_per_ball', 'consistency_score', 'home_advantage']
    
    bowl_features = create_lagged_features(
        bowler_war, 
        id_col='bowler_id', 
        name_col='bowler_name',
        metric_cols=bowl_metrics,
        lags=[1, 2]
    )
    
    bowl_features = add_metadata_features(bowl_features, metadata, 'bowler_id')
    bowl_features['recent_balls'] = bowl_features['balls_bowled_lag1'].fillna(0) + bowl_features['balls_bowled_lag2'].fillna(0)
    
    bowl_features.to_csv(output_dir / 'bowler_features.csv', index=False)
    print(f"Saved bowler features to {output_dir / 'bowler_features.csv'} ({len(bowl_features)} rows)")

if __name__ == "__main__":
    main()
