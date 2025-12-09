"""
Train XGBoost model on full history and generate 2026 projections (v2).

IMPROVEMENTS:
1. Reduced feature set to avoid overfitting
2. Role-specific hyperparameters with more regularization
3. Better handling of feature selection
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path


def load_features():
    """Load feature data."""
    data_dir = Path(__file__).parent.parent / 'data' / 'ml_features'
    bat_features = pd.read_csv(data_dir / 'batter_features_full.csv')
    bowl_features = pd.read_csv(data_dir / 'bowler_features_full.csv')
    return bat_features, bowl_features


def get_config(role):
    """Role-specific model configuration."""
    # Both roles need more regularization to prevent overfitting
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
    print(f"Training {role.upper()} Model (IPL-only)")
    print('='*60)
    
    # REDUCED feature set - avoid highly correlated features
    # Focus on the most predictive, least correlated features
    base_features = [
        'WAR_weighted',    # Best Marcel-style predictor (combines lags)
        'consistency',     # Stability measure
        'career_war',      # Cumulative experience
        'years_played',    # Tenure in IPL
    ]
    
    if role == 'batter':
        features = base_features + ['balls_faced']
    else:
        features = base_features + ['balls_bowled']
    
    # Filter to existing columns
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
    
    # Train with regularized config
    config = get_config(role)
    print(f"Config: max_depth={config['max_depth']}, reg_alpha={config['reg_alpha']}, reg_lambda={config['reg_lambda']}")
    
    model = xgb.XGBRegressor(**config)
    model.fit(X_train, y_train)
    
    # Evaluate on training set (to check overfitting)
    train_preds = model.predict(X_train)
    train_r2 = r2_score(y_train, train_preds)
    
    results_dir = Path(__file__).parent.parent / 'results' / 'WARprojections'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate on Backtest (2025 Actuals)
    if len(X_backtest) > 0:
        preds_backtest = model.predict(X_backtest)
        mae = mean_absolute_error(y_backtest, preds_backtest)
        rmse = np.sqrt(mean_squared_error(y_backtest, preds_backtest))
        r2 = r2_score(y_backtest, preds_backtest)
        
        print(f"\nResults:")
        print(f"  Train R2: {train_r2:.4f}")
        print(f"  Test R2:  {r2:.4f} (gap: {train_r2 - r2:.4f})")
        print(f"  MAE:      {mae:.4f}")
        print(f"  RMSE:     {rmse:.4f}")
        
        # Feature Importance
        print("\nFeature Importance:")
        for f, imp in sorted(zip(features, model.feature_importances_), key=lambda x: -x[1]):
            print(f"  {f:20s}: {imp:.4f}")
        
        # Save backtest results
        backtest_df = df.loc[backtest_mask, ['season', f'{role}_name', target_col]].copy()
        backtest_df['predicted_war'] = preds_backtest
        backtest_df['error'] = backtest_df['predicted_war'] - backtest_df[target_col]
        backtest_df.to_csv(results_dir / f'{role}_backtest_2025.csv', index=False)
        
    # Forecast 2026
    if len(X_forecast) > 0:
        preds_forecast = model.predict(X_forecast)
        
        forecast_df = df.loc[forecast_mask, ['season', f'{role}_name']].copy()
        forecast_df['projected_war_2026'] = preds_forecast
        forecast_df.to_csv(results_dir / f'{role}_projections_2026.csv', index=False)
        
        print(f"\nTop 10 Projected {role.capitalize()}s (2026):")
        print(forecast_df.sort_values('projected_war_2026', ascending=False).head(10).to_string(index=False))

    return model


def main():
    bat_features, bowl_features = load_features()
    
    train_and_evaluate(bat_features, 'batter')
    train_and_evaluate(bowl_features, 'bowler')
    
    print("\n" + "="*60)
    print("✓ IPL-only model training complete")
    print("="*60)


if __name__ == "__main__":
    main()
