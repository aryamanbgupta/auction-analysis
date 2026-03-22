"""
Train v4 Model with All Improvements.

IMPROVEMENTS (cumulative):
- V3: Regression-to-mean targeting, phase RAA, last-5-matches form
- V4: Situational RAA (chasing/setting), batting position, team strength

OUTPUT: results/WARprojections/v4/
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from pathlib import Path


def load_features():
    """Load v4 feature data."""
    data_dir = Path(__file__).parent.parent / 'data' / 'ml_features'
    bat_features = pd.read_csv(data_dir / 'batter_features_v4.csv')
    bowl_features = pd.read_csv(data_dir / 'bowler_features_v4.csv')
    return bat_features, bowl_features


def calculate_regression_target(df, target_col='target_WAR_next'):
    """Predict deviation from regression line."""
    valid_mask = df[target_col].notna() & df['WAR_weighted'].notna()
    valid_df = df[valid_mask]
    
    X = valid_df[['WAR_weighted']].fillna(0)
    y = valid_df[target_col]
    
    reg = LinearRegression()
    reg.fit(X, y)
    
    df = df.copy()
    df['expected_war'] = reg.predict(df[['WAR_weighted']].fillna(0))
    df['target_deviation'] = df[target_col] - df['expected_war']
    
    print(f"Regression: intercept={reg.intercept_:.3f}, slope={reg.coef_[0]:.3f}, R²={reg.score(X, y):.3f}")
    
    return df, reg


def train_and_evaluate_v4(df, role, target_col='target_WAR_next'):
    """Train v4 model."""
    print(f"\n{'='*60}")
    print(f"Training {role.upper()} Model v4")
    print('='*60)
    
    df, _ = calculate_regression_target(df, target_col)
    
    # V4 feature set: V3 + situational + opportunity
    base_features = ['WAR_weighted', 'consistency', 'career_war', 'years_played']
    
    if role == 'batter':
        base_features.extend(['balls_faced', 'bat_position'])
    else:
        base_features.append('balls_bowled')
    
    phase_features = [
        'phase_raa_per_ball_powerplay',
        'phase_raa_per_ball_middle', 
        'phase_raa_per_ball_death',
    ]
    
    form_features = ['last_5_matches_raa']
    
    situational_features = [
        'sit_raa_per_ball_chasing',
        'sit_raa_per_ball_setting',
    ]
    
    opportunity_features = ['win_rate']
    
    all_features = base_features + phase_features + form_features + situational_features + opportunity_features
    features = [f for f in all_features if f in df.columns]
    
    print(f"Using {len(features)} features")
    
    # Split
    train_mask = (df['season'] < 2024) & (df['target_deviation'].notna())
    backtest_mask = (df['season'] == 2024) & (df['target_deviation'].notna())
    forecast_mask = (df['season'] == 2025)
    
    X_train = df.loc[train_mask, features].fillna(0)
    y_train = df.loc[train_mask, 'target_deviation']
    
    X_backtest = df.loc[backtest_mask, features].fillna(0)
    y_backtest_raw = df.loc[backtest_mask, target_col]
    expected_backtest = df.loc[backtest_mask, 'expected_war']
    
    X_forecast = df.loc[forecast_mask, features].fillna(0)
    expected_forecast = df.loc[forecast_mask, 'expected_war']
    
    print(f"Train: {len(X_train)}, Backtest: {len(X_backtest)}, Forecast: {len(X_forecast)}")
    
    # Train
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
    model.fit(X_train, y_train)
    
    # Evaluate
    results_dir = Path(__file__).parent.parent / 'results' / 'WARprojections' / 'v4'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    r2 = None
    if len(X_backtest) > 0:
        pred_deviation = model.predict(X_backtest)
        pred_war = expected_backtest.values + pred_deviation
        
        mae = mean_absolute_error(y_backtest_raw, pred_war)
        rmse = np.sqrt(mean_squared_error(y_backtest_raw, pred_war))
        r2 = r2_score(y_backtest_raw, pred_war)
        r2_baseline = r2_score(y_backtest_raw, expected_backtest)
        
        print(f"\nResults:")
        print(f"  Baseline R²: {r2_baseline:.4f}")
        print(f"  V4 R²:       {r2:.4f} ({r2 - r2_baseline:+.4f})")
        print(f"  MAE:         {mae:.4f}")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nTop Features:")
        print(importance.head(8).to_string(index=False))
        
        # Save
        backtest_df = df.loc[backtest_mask, ['season', f'{role}_name', target_col]].copy()
        backtest_df['predicted_war'] = pred_war
        backtest_df.to_csv(results_dir / f'{role}_backtest_2025_v4.csv', index=False)
    
    # Forecast
    if len(X_forecast) > 0:
        pred_deviation = model.predict(X_forecast)
        pred_war = expected_forecast.values + pred_deviation
        
        forecast_df = df.loc[forecast_mask, ['season', f'{role}_name']].copy()
        forecast_df['projected_war_2026'] = pred_war
        forecast_df.to_csv(results_dir / f'{role}_projections_2026_v4.csv', index=False)
        
        print(f"\nTop 10 Projected {role.capitalize()}s (2026):")
        print(forecast_df.nlargest(10, 'projected_war_2026')[['season', f'{role}_name', 'projected_war_2026']].to_string(index=False))
    
    return r2


def main():
    bat_features, bowl_features = load_features()
    
    bat_r2 = train_and_evaluate_v4(bat_features, 'batter')
    bowl_r2 = train_and_evaluate_v4(bowl_features, 'bowler')
    
    print("\n" + "="*60)
    print("V4 MODEL SUMMARY")
    print("="*60)
    print(f"  Batter R²: {bat_r2:.4f}")
    print(f"  Bowler R²: {bowl_r2:.4f}")
    print("\n✓ Results saved to results/WARprojections/v4/")


if __name__ == "__main__":
    main()
