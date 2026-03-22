"""
Train v3 Model with High-Impact Improvements.

IMPROVEMENTS:
1. Regression-to-mean targeting: Predict deviation from expected WAR, not raw WAR
2. Phase-specific features (powerplay/middle/death RAA)
3. Last N matches form features
4. Same regularization as v2

OUTPUT: results/WARprojections/v3/ (separate from v2 for comparison)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from pathlib import Path


def load_features():
    """Load v3 feature data."""
    data_dir = Path(__file__).parent.parent / 'data' / 'ml_features'
    
    bat_features = pd.read_csv(data_dir / 'batter_features_v3.csv')
    bowl_features = pd.read_csv(data_dir / 'bowler_features_v3.csv')
    
    return bat_features, bowl_features


def calculate_regression_target(df, target_col='target_WAR_next'):
    """
    Instead of predicting raw WAR, predict deviation from regression line.
    
    This explicitly models that:
    - High performers tend to regress down
    - Low performers tend to regress up
    """
    # Fit simple regression: expected_WAR = f(WAR_weighted)
    valid_mask = df[target_col].notna() & df['WAR_weighted'].notna()
    valid_df = df[valid_mask]
    
    X = valid_df[['WAR_weighted']].fillna(0)
    y = valid_df[target_col]
    
    reg = LinearRegression()
    reg.fit(X, y)
    
    # Calculate expected WAR for everyone
    df = df.copy()
    df['expected_war'] = reg.predict(df[['WAR_weighted']].fillna(0))
    
    # New target: deviation from expected
    df['target_deviation'] = df[target_col] - df['expected_war']
    
    print(f"Regression coefficients: intercept={reg.intercept_:.4f}, slope={reg.coef_[0]:.4f}")
    print(f"R² of regression: {reg.score(X, y):.4f}")
    
    return df, reg


def train_and_evaluate_v3(df, role, target_col='target_WAR_next'):
    """Train v3 model with regression-to-mean targeting."""
    print(f"\n{'='*60}")
    print(f"Training {role.upper()} Model v3 (Regression-to-Mean)")
    print('='*60)
    
    # Calculate regression target
    df, reg_model = calculate_regression_target(df, target_col)
    
    # Features: base v2 features + new phase/form features
    base_features = [
        'WAR_weighted',
        'consistency', 
        'career_war',
        'years_played',
    ]
    
    if role == 'batter':
        base_features.append('balls_faced')
    else:
        base_features.append('balls_bowled')
    
    # NEW v3 features
    phase_features = [
        'phase_raa_per_ball_powerplay',
        'phase_raa_per_ball_middle', 
        'phase_raa_per_ball_death',
    ]
    
    form_features = [
        'last_5_matches_raa',
    ]
    
    all_features = base_features + phase_features + form_features
    
    # Filter to existing columns
    features = [f for f in all_features if f in df.columns]
    print(f"Using {len(features)} features: {features}")
    
    # Split Data
    train_mask = (df['season'] < 2024) & (df['target_deviation'].notna())
    backtest_mask = (df['season'] == 2024) & (df['target_deviation'].notna())
    forecast_mask = (df['season'] == 2025)
    
    X_train = df.loc[train_mask, features].fillna(0)
    y_train_dev = df.loc[train_mask, 'target_deviation']
    y_train_raw = df.loc[train_mask, target_col]
    
    X_backtest = df.loc[backtest_mask, features].fillna(0)
    y_backtest_dev = df.loc[backtest_mask, 'target_deviation']
    y_backtest_raw = df.loc[backtest_mask, target_col]
    expected_backtest = df.loc[backtest_mask, 'expected_war']
    
    X_forecast = df.loc[forecast_mask, features].fillna(0)
    expected_forecast = df.loc[forecast_mask, 'expected_war']
    
    print(f"Train: {len(X_train)}, Backtest: {len(X_backtest)}, Forecast: {len(X_forecast)}")
    
    # Train model to predict DEVIATION
    config = {
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
    
    model = xgb.XGBRegressor(**config)
    model.fit(X_train, y_train_dev)
    
    # Evaluate on Backtest
    results_dir = Path(__file__).parent.parent / 'results' / 'WARprojections' / 'v3'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if len(X_backtest) > 0:
        # Predict deviation
        pred_deviation = model.predict(X_backtest)
        
        # Convert back to WAR: predicted_WAR = expected_WAR + predicted_deviation
        pred_war = expected_backtest.values + pred_deviation
        
        # Calculate metrics on RAW WAR (not deviation)
        mae = mean_absolute_error(y_backtest_raw, pred_war)
        rmse = np.sqrt(mean_squared_error(y_backtest_raw, pred_war))
        r2 = r2_score(y_backtest_raw, pred_war)
        
        # Also calculate baseline metrics (just using regression)
        r2_baseline = r2_score(y_backtest_raw, expected_backtest)
        
        print(f"\nBacktest Results (Predicting 2025):")
        print(f"  Baseline R² (regression only): {r2_baseline:.4f}")
        print(f"  V3 Model R²:                   {r2:.4f} ({r2 - r2_baseline:+.4f} improvement)")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        
        # Feature Importance
        importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop Features:")
        print(importance.head(8).to_string(index=False))
        
        # Save backtest results
        backtest_df = df.loc[backtest_mask, ['season', f'{role}_name', target_col]].copy()
        backtest_df['predicted_war'] = pred_war
        backtest_df['predicted_deviation'] = pred_deviation
        backtest_df['expected_war'] = expected_backtest.values
        backtest_df['error'] = pred_war - y_backtest_raw.values
        backtest_df.to_csv(results_dir / f'{role}_backtest_2025_v3.csv', index=False)
        
    # Forecast 2026
    if len(X_forecast) > 0:
        pred_deviation = model.predict(X_forecast)
        pred_war = expected_forecast.values + pred_deviation
        
        forecast_df = df.loc[forecast_mask, ['season', f'{role}_name']].copy()
        forecast_df['projected_war_2026'] = pred_war
        forecast_df['expected_war'] = expected_forecast.values
        forecast_df['predicted_deviation'] = pred_deviation
        forecast_df.to_csv(results_dir / f'{role}_projections_2026_v3.csv', index=False)
        
        print(f"\nTop 10 Projected {role.capitalize()}s (2026):")
        top10 = forecast_df.sort_values('projected_war_2026', ascending=False).head(10)
        print(top10[['season', f'{role}_name', 'projected_war_2026', 'expected_war', 'predicted_deviation']].to_string(index=False))

    return model, r2


def main():
    bat_features, bowl_features = load_features()
    
    bat_model, bat_r2 = train_and_evaluate_v3(bat_features, 'batter')
    bowl_model, bowl_r2 = train_and_evaluate_v3(bowl_features, 'bowler')
    
    print("\n" + "="*60)
    print("V3 MODEL SUMMARY")
    print("="*60)
    print(f"  Batter R²: {bat_r2:.4f}")
    print(f"  Bowler R²: {bowl_r2:.4f}")
    print("\n✓ Results saved to results/WARprojections/v3/")


if __name__ == "__main__":
    main()
