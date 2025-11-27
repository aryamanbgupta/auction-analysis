"""
Train XGBoost model on full history and generate 2026 projections.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import matplotlib.pyplot as plt
import shap

def load_features():
    """Load feature data."""
    data_dir = Path(__file__).parent.parent / 'data' / 'ml_features'
    bat_features = pd.read_csv(data_dir / 'batter_features_full.csv')
    bowl_features = pd.read_csv(data_dir / 'bowler_features_full.csv')
    return bat_features, bowl_features

def train_and_evaluate(df, role, target_col='target_WAR_next'):
    """Train XGBoost model and evaluate."""
    print(f"\n--- Training {role} Model ---")
    
    # Define features
    features = [
        'age', 'career_war', 'career_raa', 'career_balls', 'years_played',
        'WAR', 'RAA', 'RAA_per_ball', 'consistency',
        'WAR_lag1', 'RAA_lag1', 'consistency_lag1',
        'WAR_lag2', 'RAA_lag2',
        'WAR_weighted', 'RAA_weighted'
    ]
    
    if role == 'batter':
        features.extend(['balls_faced', 'balls_faced_lag1'])
    else:
        features.extend(['balls_bowled', 'balls_bowled_lag1'])
        
    # Filter for rows with enough history (e.g., at least 1 year of lag)
    # df = df.dropna(subset=['WAR_lag1'])
    
    # Split Data
    # Train: < 2024 (Predicting up to 2024)
    # Backtest: 2024 (Predicting 2025)
    # Forecast: 2025 (Predicting 2026)
    
    train_mask = (df['season'] < 2024) & (df[target_col].notna())
    backtest_mask = (df['season'] == 2024) & (df[target_col].notna())
    forecast_mask = (df['season'] == 2025)
    
    X_train = df.loc[train_mask, features]
    y_train = df.loc[train_mask, target_col]
    
    X_backtest = df.loc[backtest_mask, features]
    y_backtest = df.loc[backtest_mask, target_col]
    
    X_forecast = df.loc[forecast_mask, features]
    
    print(f"Train size: {len(X_train)}")
    print(f"Backtest size: {len(X_backtest)}")
    print(f"Forecast candidates: {len(X_forecast)}")
    
    # Train
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate on Backtest (2025 Actuals)
    if len(X_backtest) > 0:
        preds_backtest = model.predict(X_backtest)
        mae = mean_absolute_error(y_backtest, preds_backtest)
        rmse = np.sqrt(mean_squared_error(y_backtest, preds_backtest))
        r2 = r2_score(y_backtest, preds_backtest)
        
        print(f"\nBacktest Results (Predicting 2025):")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R2:   {r2:.4f}")
        
        # Save backtest results
        backtest_df = df.loc[backtest_mask, ['season', f'{role}_name', target_col]].copy()
        backtest_df['predicted_war'] = preds_backtest
        backtest_df['error'] = backtest_df['predicted_war'] - backtest_df[target_col]
        
        results_dir = Path(__file__).parent.parent / 'results' / 'WARprojections'
        results_dir.mkdir(parents=True, exist_ok=True)
        backtest_df.to_csv(results_dir / f'{role}_backtest_2025.csv', index=False)
        
    # Forecast 2026
    if len(X_forecast) > 0:
        preds_forecast = model.predict(X_forecast)
        
        forecast_df = df.loc[forecast_mask, ['season', f'{role}_name']].copy()
        forecast_df['projected_war_2026'] = preds_forecast
        
        results_dir = Path(__file__).parent.parent / 'results' / 'WARprojections'
        forecast_df.to_csv(results_dir / f'{role}_projections_2026.csv', index=False)
        print(f"\nSaved 2026 projections to {results_dir}")
        
        # Top 10 Projections
        print(f"\nTop 10 Projected {role.capitalize()}s (2026):")
        print(forecast_df.sort_values('projected_war_2026', ascending=False).head(10))

    return model, features

def main():
    bat_features, bowl_features = load_features()
    
    train_and_evaluate(bat_features, 'batter')
    train_and_evaluate(bowl_features, 'bowler')

if __name__ == "__main__":
    main()
