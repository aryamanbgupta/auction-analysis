"""
Global-to-IPL Translation Model.

PURPOSE: Predict IPL performance for players who have NEVER played IPL,
based solely on their global T20 performance (T20I, BBL, PSL, CPL, etc.)

APPROACH:
1. For players with BOTH global + IPL data: Learn the relationship
2. Apply to players with ONLY global data to predict their IPL potential

OUTPUT: results/WARprojections/global_only/
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from pathlib import Path


def load_data():
    """Load IPL and global T20 data."""
    data_dir = Path(__file__).parent.parent / 'data'
    
    global_df = pd.read_parquet(data_dir / 'global_t20_matches.parquet')
    ipl_df = pd.read_parquet(data_dir / 'ipl_matches_all.parquet')
    
    # Load league factors
    league_factors = pd.read_csv(data_dir / 'league_factors.csv')
    
    return global_df, ipl_df, league_factors


def calculate_global_features(global_df, league_factors, role='batter'):
    """Calculate global T20 features for each player."""
    print(f"Calculating global features for {role}s...")
    
    id_col = f'{role}_id'
    name_col = f'{role}_name'
    
    # Get league factors (format: league, role, raw_factor, factor, ...)
    role_factors = league_factors[league_factors['role'] == role]
    factors = role_factors.set_index('league')['factor'].to_dict()
    
    # Calculate RAA per ball
    global_df = global_df.copy()
    global_df['league_avg'] = global_df.groupby('league')['total_runs'].transform('mean')
    global_df['raw_raa'] = global_df['total_runs'] - global_df['league_avg']
    
    if role == 'bowler':
        global_df['raw_raa'] = -global_df['raw_raa']  # Fewer runs is better for bowlers
    
    # Apply league adjustment factor
    global_df['league_factor'] = global_df['league'].map(factors).fillna(0.3)
    global_df['adjusted_raa'] = global_df['raw_raa'] * global_df['league_factor']
    
    # Aggregate by player
    agg_funcs = {
        'adjusted_raa': 'sum',
        'raw_raa': 'sum',
        'match_id': 'count',  # balls
    }
    
    player_stats = global_df.groupby([id_col, name_col]).agg(agg_funcs)
    player_stats = player_stats.rename(columns={'match_id': 'global_balls'})
    player_stats = player_stats.reset_index()
    
    # Calculate derived features
    player_stats['global_raa_per_ball'] = player_stats['adjusted_raa'] / player_stats['global_balls']
    player_stats['global_raw_raa_per_ball'] = player_stats['raw_raa'] / player_stats['global_balls']
    
    # League diversity (how many leagues played)
    leagues_played = global_df.groupby(id_col)['league'].nunique().reset_index()
    leagues_played.columns = [id_col, 'leagues_played']
    player_stats = player_stats.merge(leagues_played, on=id_col, how='left')
    
    # T20I experience
    t20i_data = global_df[global_df['league'].str.contains('T20I', na=False)]
    t20i_balls = t20i_data.groupby(id_col).size().reset_index(name='t20i_balls')
    player_stats = player_stats.merge(t20i_balls, on=id_col, how='left')
    player_stats['t20i_balls'] = player_stats['t20i_balls'].fillna(0)
    
    # Has T20I experience flag
    player_stats['has_t20i'] = (player_stats['t20i_balls'] > 0).astype(int)
    
    # Major franchise leagues (BBL, PSL, CPL, etc.)
    franchise_leagues = ['BBL', 'PSL', 'CPL', 'BPL', 'SA20', 'MLC', 'ILT20']
    franchise_data = global_df[global_df['league'].isin(franchise_leagues)]
    franchise_balls = franchise_data.groupby(id_col).size().reset_index(name='franchise_balls')
    player_stats = player_stats.merge(franchise_balls, on=id_col, how='left')
    player_stats['franchise_balls'] = player_stats['franchise_balls'].fillna(0)
    
    return player_stats


def calculate_ipl_target(ipl_df, role='batter'):
    """Calculate average IPL WAR per season for each player."""
    print(f"Calculating IPL targets for {role}s...")
    
    id_col = f'{role}_id'
    name_col = f'{role}_name'
    
    # Load WAR data
    data_dir = Path(__file__).parent.parent / 'data'
    war_file = data_dir / f'{role}_war_full_history.csv'
    
    if war_file.exists():
        war_df = pd.read_csv(war_file)
        
        # Calculate average WAR and total WAR
        player_war = war_df.groupby(name_col).agg({
            'WAR': ['mean', 'sum', 'count', 'max']
        })
        player_war.columns = ['avg_war', 'total_war', 'seasons_played', 'peak_war']
        player_war = player_war.reset_index()
        
        return player_war
    
    return None


def prepare_training_data(global_features, ipl_targets, role='batter'):
    """Merge global features with IPL targets to create training data."""
    name_col = f'{role}_name'
    
    # Merge
    df = global_features.merge(ipl_targets, on=name_col, how='inner')
    
    print(f"Training data: {len(df)} {role}s with both global + IPL data")
    
    return df


def train_global_to_ipl_model(df, role='batter'):
    """Train model to predict IPL WAR from global features."""
    print(f"\n{'='*60}")
    print(f"Training Global-to-IPL Model for {role.upper()}s")
    print('='*60)
    
    # Features
    features = [
        'global_balls',
        'global_raa_per_ball',
        'leagues_played',
        't20i_balls',
        'has_t20i',
        'franchise_balls',
    ]
    
    features = [f for f in features if f in df.columns]
    print(f"Features: {features}")
    
    # Target: Average IPL WAR
    target = 'avg_war'
    
    # Remove NaN
    df_clean = df.dropna(subset=features + [target])
    
    X = df_clean[features]
    y = df_clean[target]
    
    print(f"Training samples: {len(X)}")
    
    # Cross-validation
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1
    )
    
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"Cross-validation R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Train final model
    model.fit(X, y)
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nFeature Importance:")
    print(importance.to_string(index=False))
    
    # Training performance
    y_pred = model.predict(X)
    train_r2 = r2_score(y, y_pred)
    train_mae = mean_absolute_error(y, y_pred)
    print(f"\nTraining R²: {train_r2:.4f}")
    print(f"Training MAE: {train_mae:.4f}")
    
    return model, features


def predict_for_new_players(model, features, global_features, ipl_players, role='batter'):
    """Predict IPL WAR for players with NO IPL history."""
    name_col = f'{role}_name'
    
    # Filter to players NOT in IPL
    ipl_names = set(ipl_players)
    global_only = global_features[~global_features[name_col].isin(ipl_names)]
    
    print(f"\n{role.capitalize()}s with global data but NO IPL history: {len(global_only)}")
    
    # Make predictions
    X = global_only[features].fillna(0)
    predictions = model.predict(X)
    
    result = global_only[[name_col, 'global_balls', 'global_raa_per_ball', 'leagues_played']].copy()
    result['predicted_ipl_war'] = predictions
    result = result.sort_values('predicted_ipl_war', ascending=False)
    
    return result


def main():
    global_df, ipl_df, league_factors = load_data()
    
    results_dir = Path(__file__).parent.parent / 'results' / 'WARprojections' / 'global_only'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for role in ['batter', 'bowler']:
        print(f"\n\n{'='*60}")
        print(f"PROCESSING {role.upper()}S")
        print('='*60)
        
        # Calculate global features
        global_features = calculate_global_features(global_df, league_factors, role)
        
        # Calculate IPL targets
        ipl_targets = calculate_ipl_target(ipl_df, role)
        
        if ipl_targets is None:
            print(f"Could not load IPL WAR data for {role}s")
            continue
        
        # Prepare training data
        train_df = prepare_training_data(global_features, ipl_targets, role)
        
        # Train model
        model, features = train_global_to_ipl_model(train_df, role)
        
        # Get IPL player names
        name_col = f'{role}_name'
        ipl_players = set(ipl_df[name_col].unique())
        
        # Predict for new players
        predictions = predict_for_new_players(model, features, global_features, ipl_players, role)
        
        # Filter to players with meaningful global experience (min 100 balls)
        predictions = predictions[predictions['global_balls'] >= 100]
        
        # Save predictions
        predictions.to_csv(results_dir / f'{role}_global_only_predictions.csv', index=False)
        
        print(f"\n--- TOP 20 {role.upper()} PROSPECTS (No IPL History) ---")
        print(predictions.head(20).to_string(index=False))
        
        # Also save the training validation for reference
        train_df['predicted_war'] = model.predict(train_df[features].fillna(0))
        train_df['error'] = train_df['predicted_war'] - train_df['avg_war']
        train_df[['batter_name' if role == 'batter' else 'bowler_name', 'avg_war', 'predicted_war', 'error']].to_csv(
            results_dir / f'{role}_training_validation.csv', index=False
        )
    
    print("\n" + "="*60)
    print("✓ Global-to-IPL Translation Model Complete!")
    print(f"  Results saved to: {results_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
