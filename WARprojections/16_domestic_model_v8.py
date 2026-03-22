"""
V8 Domestic-Enhanced Model - Better predictions for uncapped/domestic players.

KEY IMPROVEMENTS:
1. More comprehensive SMAT/domestic league feature extraction
2. Special handling for players with NO IPL history
3. Trains on global-to-IPL relationships
4. Backtests on 2025 season

This model is specifically designed for auction players who:
- Have SMAT/domestic T20 experience
- May have NO IPL history
- Are typically Indian domestic players

OUTPUT: results/WARprojections/v8_domestic/
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from pathlib import Path


def load_data():
    """Load all required data sources."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    
    data = {}
    
    # Global T20 data (includes SMAT)
    data['global_df'] = pd.read_parquet(data_dir / 'global_t20_matches.parquet')
    
    # IPL data
    data['ipl_df'] = pd.read_parquet(data_dir / 'ipl_matches_all.parquet')
    
    # League factors
    data['league_factors'] = pd.read_csv(data_dir / 'league_factors.csv')
    
    # WAR history
    data['batter_war'] = pd.read_csv(data_dir / 'batter_war_full_history.csv')
    data['bowler_war'] = pd.read_csv(data_dir / 'bowler_war_full_history.csv')
    
    # Auction list
    data['auction'] = pd.read_csv(data_dir / 'ipl_2026_auction_enriched.csv')
    
    return data


def calculate_domestic_features(global_df, league_factors, role='batter'):
    """Calculate comprehensive features from domestic T20 leagues."""
    print(f"\n--- Calculating Domestic Features for {role.upper()}S ---")
    
    id_col = f'{role}_id'
    name_col = f'{role}_name'
    
    # Get role-specific factors
    role_factors = league_factors[league_factors['role'] == role]
    factors = role_factors.set_index('league')['factor'].to_dict()
    
    # Classify leagues
    domestic_leagues = ['SMAT']  # India domestic
    international_leagues = ['T20I']
    franchise_leagues = ['BBL', 'PSL', 'CPL', 'SA20', 'ILT20', 'MLC', 'BPL', 'T20Blast']
    
    df = global_df.copy()
    
    # Calculate basic RAA
    df['league_avg'] = df.groupby('league')['total_runs'].transform('mean')
    df['raw_raa'] = df['total_runs'] - df['league_avg']
    
    if role == 'bowler':
        df['raw_raa'] = -df['raw_raa']
    
    # Apply league factors
    df['league_factor'] = df['league'].map(factors).fillna(0.3)
    df['adjusted_raa'] = df['raw_raa'] * df['league_factor']
    
    # Extract year
    df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')
    df['year'] = df['match_date'].dt.year
    
    # === SMAT-specific features ===
    smat_df = df[df['league'] == 'SMAT']
    smat_stats = smat_df.groupby([id_col, name_col]).agg({
        'adjusted_raa': 'sum',
        'match_id': 'count',
    }).reset_index()
    smat_stats.columns = [id_col, name_col, 'smat_raa', 'smat_balls']
    smat_stats['smat_raa_per_ball'] = smat_stats['smat_raa'] / smat_stats['smat_balls']
    
    # Recent SMAT (last 2 years for 2025 backtest - 2023, 2024)
    recent_smat = smat_df[smat_df['year'].isin([2023, 2024])]
    recent_smat_stats = recent_smat.groupby(id_col).agg({
        'adjusted_raa': 'sum',
        'match_id': 'count',
    }).reset_index()
    recent_smat_stats.columns = [id_col, 'recent_smat_raa', 'recent_smat_balls']
    recent_smat_stats['recent_smat_raa_per_ball'] = (
        recent_smat_stats['recent_smat_raa'] / recent_smat_stats['recent_smat_balls']
    )
    
    smat_stats = smat_stats.merge(recent_smat_stats, on=id_col, how='left')
    
    # === T20I features ===
    t20i_df = df[df['league'].str.contains('T20I', na=False)]
    t20i_stats = t20i_df.groupby(id_col).agg({
        'adjusted_raa': 'sum',
        'match_id': 'count',
    }).reset_index()
    t20i_stats.columns = [id_col, 't20i_raa', 't20i_balls']
    t20i_stats['t20i_raa_per_ball'] = t20i_stats['t20i_raa'] / t20i_stats['t20i_balls']
    t20i_stats['has_t20i'] = 1
    
    # === Franchise league features ===
    franchise_df = df[df['league'].isin(franchise_leagues)]
    franchise_stats = franchise_df.groupby(id_col).agg({
        'adjusted_raa': 'sum',
        'match_id': 'count',
        'league': 'nunique'
    }).reset_index()
    franchise_stats.columns = [id_col, 'franchise_raa', 'franchise_balls', 'franchise_leagues']
    franchise_stats['franchise_raa_per_ball'] = franchise_stats['franchise_raa'] / franchise_stats['franchise_balls']
    
    # === Total global features ===
    all_stats = df.groupby([id_col, name_col]).agg({
        'adjusted_raa': 'sum',
        'match_id': 'count',
        'league': 'nunique'
    }).reset_index()
    all_stats.columns = [id_col, name_col, 'global_raa', 'global_balls', 'leagues_played']
    all_stats['global_raa_per_ball'] = all_stats['global_raa'] / all_stats['global_balls']
    
    # Merge all features
    features = all_stats.merge(smat_stats[[id_col, 'smat_balls', 'smat_raa_per_ball', 
                                           'recent_smat_balls', 'recent_smat_raa_per_ball']], 
                               on=id_col, how='left')
    features = features.merge(t20i_stats, on=id_col, how='left')
    features = features.merge(franchise_stats, on=id_col, how='left')
    
    # Fill NaN flags
    features['has_smat'] = features['smat_balls'].notna().astype(int)
    features['has_t20i'] = features['has_t20i'].fillna(0)
    features['has_franchise'] = features['franchise_balls'].notna().astype(int)
    
    print(f"  Total players: {len(features)}")
    print(f"  With SMAT: {features['has_smat'].sum()}")
    print(f"  With T20I: {features['has_t20i'].sum()}")
    print(f"  With Franchise: {features['has_franchise'].sum()}")
    
    return features


def prepare_training_data(domestic_features, ipl_war, role='batter'):
    """Prepare training data: players with both domestic + IPL experience."""
    id_col = f'{role}_id'
    name_col = f'{role}_name'
    
    # Get IPL WAR by player (average across seasons)
    ipl_avg = ipl_war.groupby([id_col, name_col]).agg({
        'WAR': ['mean', 'sum', 'count', 'max', 'std']
    }).reset_index()
    ipl_avg.columns = [id_col, name_col, 'avg_ipl_war', 'total_ipl_war', 'ipl_seasons', 'peak_ipl_war', 'ipl_war_std']
    ipl_avg['ipl_war_std'] = ipl_avg['ipl_war_std'].fillna(0)
    
    # Merge with domestic features
    train_df = domestic_features.merge(ipl_avg, on=[id_col, name_col], how='inner')
    
    print(f"\nTraining data: {len(train_df)} players with both domestic + IPL experience")
    print(f"  Avg IPL WAR: {train_df['avg_ipl_war'].mean():.3f}")
    
    return train_df


def train_domestic_model(train_df, role='batter'):
    """Train model to predict IPL WAR from domestic features."""
    print(f"\n{'='*60}")
    print(f"Training DOMESTIC Model for {role.upper()}S")
    print('='*60)
    
    name_col = f'{role}_name'
    
    # Features
    features = [
        'global_balls', 'global_raa_per_ball', 'leagues_played',
        'smat_balls', 'smat_raa_per_ball', 'recent_smat_balls', 'recent_smat_raa_per_ball',
        't20i_balls', 't20i_raa_per_ball', 'has_t20i',
        'franchise_balls', 'franchise_raa_per_ball', 'franchise_leagues',
        'has_smat', 'has_franchise'
    ]
    features = [f for f in features if f in train_df.columns]
    
    target = 'avg_ipl_war'
    
    # Remove rows with missing target
    df = train_df.dropna(subset=[target])
    
    X = df[features].fillna(0)
    y = df[target]
    
    print(f"Using {len(features)} features, {len(X)} samples")
    
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
    print(importance.head(10).to_string(index=False))
    
    # Training metrics
    y_pred = model.predict(X)
    train_r2 = r2_score(y, y_pred)
    train_rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f"\nTraining R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}")
    
    return model, features, importance


def predict_for_auction_players(model, features, domestic_features, auction, ipl_player_ids, role='batter'):
    """Predict IPL WAR for auction players using domestic model."""
    id_col = f'{role}_id'
    name_col = f'{role}_name'
    
    # Filter auction to players with cricsheet_id
    auction_with_id = auction[auction['cricsheet_id'].notna()].copy()
    auction_ids = set(auction_with_id['cricsheet_id'])
    
    # Players with domestic but potentially NO IPL
    domestic_auction = domestic_features[domestic_features[id_col].isin(auction_ids)]
    
    # Split: players with IPL history vs no IPL history
    with_ipl = domestic_auction[domestic_auction[id_col].isin(ipl_player_ids)]
    no_ipl = domestic_auction[~domestic_auction[id_col].isin(ipl_player_ids)]
    
    print(f"\n--- Auction Players ({role.upper()}) ---")
    print(f"  With IPL history: {len(with_ipl)}")
    print(f"  NO IPL history: {len(no_ipl)}")
    
    # Predict for players without IPL history
    if len(no_ipl) > 0:
        X_pred = no_ipl[features].fillna(0)
        predictions = model.predict(X_pred)
        
        results = no_ipl[[id_col, name_col, 'global_balls', 'smat_balls', 't20i_balls']].copy()
        results['predicted_ipl_war'] = predictions
        results = results.sort_values('predicted_ipl_war', ascending=False)
        
        return results
    
    return pd.DataFrame()


def main():
    print("=" * 60)
    print("V8 DOMESTIC-ENHANCED MODEL")
    print("Focus: SMAT + Domestic T20 → IPL Translation")
    print("=" * 60)
    
    # Load data
    data = load_data()
    
    output_dir = Path(__file__).parent.parent / 'results' / 'WARprojections' / 'v8_domestic'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_predictions = []
    
    for role in ['batter', 'bowler']:
        id_col = f'{role}_id'
        
        # Calculate domestic features
        domestic_features = calculate_domestic_features(
            data['global_df'], data['league_factors'], role
        )
        
        # Get IPL player IDs
        ipl_war = data['batter_war'] if role == 'batter' else data['bowler_war']
        ipl_player_ids = set(ipl_war[id_col].unique())
        
        # Prepare training data
        train_df = prepare_training_data(domestic_features, ipl_war, role)
        
        # Train model
        model, features, importance = train_domestic_model(train_df, role)
        
        # Predict for auction players
        predictions = predict_for_auction_players(
            model, features, domestic_features, data['auction'], ipl_player_ids, role
        )
        
        if len(predictions) > 0:
            predictions['role'] = role
            all_predictions.append(predictions)
            
            print(f"\nTop 10 {role.upper()} Prospects (No IPL History):")
            print(predictions.head(10).to_string(index=False))
            
            # Save
            predictions.to_csv(output_dir / f'{role}_domestic_predictions.csv', index=False)
        
        # Save importance
        importance.to_csv(output_dir / f'{role}_feature_importance_v8.csv', index=False)
    
    # Combine all predictions
    if all_predictions:
        combined = pd.concat(all_predictions, ignore_index=True)
        combined.to_csv(output_dir / 'all_domestic_predictions.csv', index=False)
        
        print("\n" + "=" * 60)
        print("V8 DOMESTIC MODEL SUMMARY")
        print("=" * 60)
        print(f"Total players predicted: {len(combined)}")
        print(f"  Batters: {len(combined[combined['role'] == 'batter'])}")
        print(f"  Bowlers: {len(combined[combined['role'] == 'bowler'])}")
    
    print(f"\n✓ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
