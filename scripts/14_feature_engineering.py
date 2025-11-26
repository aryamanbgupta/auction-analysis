import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def load_data():
    """Load WAR data and player metadata."""
    results_dir = Path(__file__).parent.parent / 'results' / '09_vorp_war'
    data_dir = Path(__file__).parent.parent / 'data'
    
    batter_war = pd.read_csv(results_dir / 'batter_war.csv')
    bowler_war = pd.read_csv(results_dir / 'bowler_war.csv')
    
    # Try to load full metadata first
    full_meta_path = data_dir / 'player_metadata_full.csv'
    if full_meta_path.exists():
        metadata = pd.read_csv(full_meta_path)
        print(f"Loaded metadata from {full_meta_path}")
    else:
        metadata = pd.read_csv(data_dir / 'player_metadata.csv')
        print(f"Loaded metadata from {data_dir / 'player_metadata.csv'}")
    
    return batter_war, bowler_war, metadata

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

def create_lagged_features(df, id_col, name_col, metric_cols, lags=[1, 2]):
    """
    Create lagged features for time-series data.
    
    Args:
        df: DataFrame with 'season', id_col, and metric_cols
        id_col: Column name for player ID
        metric_cols: List of columns to create lags for
        lags: List of previous years to look back
        
    Returns:
        DataFrame with original columns plus lagged features
    """
    df = df.copy()
    df = df.sort_values(['season', id_col])
    
    # Create a master list of all player-seasons we want to predict
    # We want to predict for every player who played in a season, 
    # PLUS potentially players who played in previous seasons (but that's for inference)
    # For training, we only have labels (actual WAR) for players who actually played.
    # So we stick to the rows present in df for the target.
    
    # However, to get lagged values, we need to look up past data.
    # Self-join is the easiest way.
    
    feature_df = df.copy()
    
    for lag in lags:
        # Create a temporary dataframe for the lagged year
        lag_df = df[['season', id_col] + metric_cols].copy()
        lag_df['season'] = lag_df['season'] + lag  # Shift season forward to match target
        
        # Rename columns
        rename_dict = {col: f'{col}_lag{lag}' for col in metric_cols}
        lag_df = lag_df.rename(columns=rename_dict)
        
        # Merge back to original
        feature_df = pd.merge(
            feature_df, 
            lag_df, 
            on=['season', id_col], 
            how='left'
        )
        
    return feature_df

def add_metadata_features(df, metadata, id_col):
    """Add age and other metadata features."""
    # Merge metadata
    cols_to_use = ['player_id']
    if 'dob' in metadata.columns:
        cols_to_use.append('dob')
    if 'batting_hand' in metadata.columns:
        cols_to_use.append('batting_hand')
    if 'bowling_type' in metadata.columns:
        cols_to_use.append('bowling_type')
        
    df = pd.merge(df, metadata[cols_to_use], 
                  left_on=id_col, right_on='player_id', how='left')
    
    # Calculate Age
    if 'dob' in df.columns:
        df['age'] = df.apply(lambda x: calculate_age(x['dob'], x['season']), axis=1)
    else:
        df['age'] = np.nan
    
    # Fill missing age with mean (or median)
    if df['age'].isnull().any():
        # If all are NaN (e.g. dob missing), fill with default age (e.g. 27)
        if df['age'].isnull().all():
            df['age'] = 27
        else:
            df['age'] = df['age'].fillna(df['age'].median())
        
    return df

def main():
    print("Loading data...")
    batter_war, bowler_war, metadata = load_data()
    
    # --- Process Batters ---
    print("Processing Batters...")
    # Metrics to lag
    bat_metrics = ['WAR', 'RAA', 'balls_faced', 'WAR_per_ball']
    
    bat_features = create_lagged_features(
        batter_war, 
        id_col='batter_id', 
        name_col='batter_name',
        metric_cols=bat_metrics,
        lags=[1, 2]
    )
    
    bat_features = add_metadata_features(bat_features, metadata, 'batter_id')
    
    # Add cumulative experience (balls faced prior to this season)
    # This is a bit trickier with just lags. 
    # Let's simple sum of lag1 and lag2 balls for now as a proxy for "recent experience"
    bat_features['recent_balls'] = bat_features['balls_faced_lag1'].fillna(0) + bat_features['balls_faced_lag2'].fillna(0)
    
    # Save
    output_dir = Path(__file__).parent.parent / 'data' / 'ml_features'
    output_dir.mkdir(exist_ok=True)
    
    bat_features.to_csv(output_dir / 'batter_features.csv', index=False)
    print(f"Saved batter features to {output_dir / 'batter_features.csv'} ({len(bat_features)} rows)")
    
    # --- Process Bowlers ---
    print("Processing Bowlers...")
    bowl_metrics = ['WAR', 'RAA', 'balls_bowled', 'WAR_per_ball']
    
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
