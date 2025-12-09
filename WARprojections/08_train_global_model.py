"""
Train XGBoost model with Global Features and generate 2026 projections (v2).

IMPROVEMENTS:
1. Role-specific hyperparameters (shallower trees + more regularization for bowlers)
2. Higher global_balls threshold for bowlers (100 balls minimum)
3. Conditional global feature usage
4. Reduced feature set to prevent overfitting
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

# Minimum global balls to trust the global signal
MIN_GLOBAL_BALLS_BATTER = 30
MIN_GLOBAL_BALLS_BOWLER = 100  # Higher threshold for bowlers


def load_features():
    """Load feature data with global stats."""
    data_dir = Path(__file__).parent.parent / 'data' / 'ml_features'
    bat_features = pd.read_csv(data_dir / 'batter_features_global.csv')
    bowl_features = pd.read_csv(data_dir / 'bowler_features_global.csv')
    return bat_features, bowl_features


def get_model_config(role):
    """
    Return role-specific model hyperparameters.
    
    Bowlers: More regularization, shallower trees (less overfitting)
    Batters: Standard configuration
    """
    if role == 'bowler':
        return {
            'objective': 'reg:squarederror',
            'n_estimators': 150,      # Fewer trees
            'learning_rate': 0.03,    # Slower learning
            'max_depth': 3,           # Shallower trees
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'reg_alpha': 1.0,         # L1 regularization
            'reg_lambda': 2.0,        # L2 regularization
            'n_jobs': -1,
            'random_state': 42
        }
    else:  # batter
        return {
            'objective': 'reg:squarederror',
            'n_estimators': 100,
            'learning_rate': 0.03,
            'max_depth': 3,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'reg_alpha': 1.0,
            'reg_lambda': 2.0,
            'n_jobs': -1,
            'random_state': 42
        }


def train_and_evaluate(df, role, target_col='target_WAR_next'):
    """Train XGBoost model and evaluate."""
    print(f"\n{'='*60}")
    print(f"Training {role.upper()} Model (Global)")
    print(f"{'='*60}")
    
    # Apply minimum balls threshold
    min_balls = MIN_GLOBAL_BALLS_BOWLER if role == 'bowler' else MIN_GLOBAL_BALLS_BATTER
    
    # Zero out global features for players with insufficient global exposure
    df = df.copy()
    insufficient_mask = df['global_balls'] < min_balls
    df.loc[insufficient_mask, 'global_raa_per_ball'] = 0
    print(f"Zeroed global features for {insufficient_mask.sum()} rows with <{min_balls} global balls")
    
    # REDUCED feature set (same as IPL-only, plus global)
    base_features = [
        'WAR_weighted',    # Best Marcel-style predictor
        'consistency',     # Stability measure
        'career_war',      # Cumulative experience
        'years_played',    # Tenure in IPL
    ]
    
    # Global features - more conservative for bowlers
    if role == 'bowler':
        # Only use global_balls as a signal
        global_features = ['global_balls']
    else:
        global_features = ['global_raa_per_ball', 'global_balls']
    
    if role == 'batter':
        role_features = ['balls_faced']
    else:
        role_features = ['balls_bowled']
    
    features = base_features + global_features + role_features
    
    # Filter to columns that exist
    features = [f for f in features if f in df.columns]
    print(f"Using {len(features)} features: {features}")
    
    # Split Data
    train_mask = (df['season'] < 2024) & (df[target_col].notna())
    backtest_mask = (df['season'] == 2024) & (df[target_col].notna())
    forecast_mask = (df['season'] == 2025)
    
    X_train = df.loc[train_mask, features].fillna(0)
    y_train = df.loc[train_mask, target_col]
    
    X_backtest = df.loc[backtest_mask, features].fillna(0)
    y_backtest = df.loc[backtest_mask, target_col]
    
    X_forecast = df.loc[forecast_mask, features].fillna(0)
    
    print(f"Train: {len(X_train)}, Backtest: {len(X_backtest)}, Forecast: {len(X_forecast)}")
    
    # Get role-specific config
    config = get_model_config(role)
    print(f"Config: max_depth={config['max_depth']}, reg_alpha={config['reg_alpha']}, reg_lambda={config['reg_lambda']}")
    
    model = xgb.XGBRegressor(**config)
    model.fit(X_train, y_train)
    
    # Evaluate on Backtest (2025 Actuals)
    results_dir = Path(__file__).parent.parent / 'results' / 'WARprojections'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if len(X_backtest) > 0:
        preds_backtest = model.predict(X_backtest)
        mae = mean_absolute_error(y_backtest, preds_backtest)
        rmse = np.sqrt(mean_squared_error(y_backtest, preds_backtest))
        r2 = r2_score(y_backtest, preds_backtest)
        
        print(f"\nBacktest Results (Predicting 2025):")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R2:   {r2:.4f}")
        
        # Feature Importance
        importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 5 Features:")
        print(importance.head(5).to_string(index=False))
        
        # Save backtest results
        backtest_df = df.loc[backtest_mask, ['season', f'{role}_name', target_col]].copy()
        backtest_df['predicted_war'] = preds_backtest
        backtest_df.to_csv(results_dir / f'{role}_backtest_2025_global.csv', index=False)
        
    # Forecast 2026
    if len(X_forecast) > 0:
        preds_forecast = model.predict(X_forecast)
        
        forecast_df = df.loc[forecast_mask, ['season', f'{role}_name']].copy()
        forecast_df['projected_war_2026'] = preds_forecast
        forecast_df.to_csv(results_dir / f'{role}_projections_2026_global.csv', index=False)
        
        print(f"\nTop 10 Projected {role.capitalize()}s (2026):")
        print(forecast_df.sort_values('projected_war_2026', ascending=False).head(10).to_string(index=False))

    return model


def main():
    bat_features, bowl_features = load_features()
    
    train_and_evaluate(bat_features, 'batter')
    train_and_evaluate(bowl_features, 'bowler')
    
    print("\n" + "="*60)
    print("✓ Global model training complete")
    print("="*60)


if __name__ == "__main__":
    main()
