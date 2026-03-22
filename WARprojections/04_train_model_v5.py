"""
Train v5 Ensemble Model.

APPROACH: Combine multiple models intelligently
1. Marcel (baseline, conservative)
2. V4 XGBoost (best ML model)
3. Dynamic weighting based on player characteristics

OUTPUT: results/WARprojections/v5/
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from pathlib import Path


def load_all_data():
    """Load all required data."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    results_dir = project_root / 'results' / 'WARprojections'
    
    # V4 features
    bat_features = pd.read_csv(data_dir / 'ml_features' / 'batter_features_v4.csv')
    bowl_features = pd.read_csv(data_dir / 'ml_features' / 'bowler_features_v4.csv')
    
    # Marcel predictions (2025 backtest)
    marcel_bat_25 = pd.read_csv(results_dir / 'marcel' / 'batter_projections_2025.csv')
    marcel_bowl_25 = pd.read_csv(results_dir / 'marcel' / 'bowler_projections_2025.csv')
    
    # Marcel predictions (2026 forecast)
    marcel_bat_26 = pd.read_csv(results_dir / 'marcel' / 'batter_projections_2026.csv')
    marcel_bowl_26 = pd.read_csv(results_dir / 'marcel' / 'bowler_projections_2026.csv')
    
    return {
        'bat_features': bat_features,
        'bowl_features': bowl_features,
        'marcel_bat_25': marcel_bat_25,
        'marcel_bowl_25': marcel_bowl_25,
        'marcel_bat_26': marcel_bat_26,
        'marcel_bowl_26': marcel_bowl_26,
    }


def train_xgb_model(df, role, target_col='target_WAR_next'):
    """Train XGBoost component (same as V4)."""
    # Calculate regression target
    valid_mask = df[target_col].notna() & df['WAR_weighted'].notna()
    valid_df = df[valid_mask]
    
    X_reg = valid_df[['WAR_weighted']].fillna(0)
    y_reg = valid_df[target_col]
    
    reg = LinearRegression()
    reg.fit(X_reg, y_reg)
    
    df = df.copy()
    df['expected_war'] = reg.predict(df[['WAR_weighted']].fillna(0))
    df['target_deviation'] = df[target_col] - df['expected_war']
    
    # Features
    base = ['WAR_weighted', 'consistency', 'career_war', 'years_played']
    if role == 'batter':
        base.extend(['balls_faced', 'bat_position'])
    else:
        base.append('balls_bowled')
    
    phase = ['phase_raa_per_ball_powerplay', 'phase_raa_per_ball_middle', 'phase_raa_per_ball_death']
    form = ['last_5_matches_raa']
    sit = ['sit_raa_per_ball_chasing', 'sit_raa_per_ball_setting']
    opp = ['win_rate']
    
    all_features = base + phase + form + sit + opp
    features = [f for f in all_features if f in df.columns]
    
    # Train
    train_mask = (df['season'] < 2024) & (df['target_deviation'].notna())
    X_train = df.loc[train_mask, features].fillna(0)
    y_train = df.loc[train_mask, 'target_deviation']
    
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
    
    return model, reg, features, df


def train_ensemble_weights(df, xgb_preds, marcel_preds, actual, role):
    """Learn optimal weights for combining predictions."""
    # Stack predictions
    X_stack = np.column_stack([xgb_preds, marcel_preds])
    
    # Use Ridge regression to find optimal weights
    stacker = Ridge(alpha=1.0)
    stacker.fit(X_stack, actual)
    
    weights = stacker.coef_ / stacker.coef_.sum()  # Normalize
    print(f"  Learned weights: XGB={weights[0]:.2f}, Marcel={weights[1]:.2f}")
    
    return stacker


def train_and_evaluate_v5(data, role):
    """Train v5 ensemble model."""
    print(f"\n{'='*60}")
    print(f"Training {role.upper()} Model v5 (Ensemble)")
    print('='*60)
    
    # Fixed key lookup
    df = data['bat_features'] if role == 'batter' else data['bowl_features']
    marcel_25 = data['marcel_bat_25'] if role == 'batter' else data['marcel_bowl_25']
    marcel_26 = data['marcel_bat_26'] if role == 'batter' else data['marcel_bowl_26']
    
    # Train XGBoost model
    xgb_model, reg_model, features, df = train_xgb_model(df, role)
    
    # Get backtest predictions
    backtest_mask = (df['season'] == 2024) & (df['target_WAR_next'].notna())
    forecast_mask = (df['season'] == 2025)
    
    X_backtest = df.loc[backtest_mask, features].fillna(0)
    expected_backtest = df.loc[backtest_mask, 'expected_war']
    actual = df.loc[backtest_mask, 'target_WAR_next']
    
    # XGB predictions
    xgb_pred_dev = xgb_model.predict(X_backtest)
    xgb_pred_war = expected_backtest.values + xgb_pred_dev
    
    # Marcel predictions (merge)
    name_col = f'{role}_name'
    backtest_names = df.loc[backtest_mask, name_col].values
    marcel_map = marcel_25.set_index('player_name')['projected_war_2025'].to_dict()
    marcel_preds = np.array([marcel_map.get(n, np.nan) for n in backtest_names])
    
    # Handle missing Marcel predictions (use XGB)
    valid_mask = ~np.isnan(marcel_preds)
    
    print(f"  Valid ensemble samples: {valid_mask.sum()}/{len(valid_mask)}")
    
    # Learn ensemble weights
    stacker = train_ensemble_weights(
        df.loc[backtest_mask][valid_mask],
        xgb_pred_war[valid_mask],
        marcel_preds[valid_mask],
        actual.values[valid_mask],
        role
    )
    
    # Ensemble prediction
    ensemble_pred = np.zeros_like(xgb_pred_war)
    ensemble_pred[valid_mask] = stacker.predict(
        np.column_stack([xgb_pred_war[valid_mask], marcel_preds[valid_mask]])
    )
    ensemble_pred[~valid_mask] = xgb_pred_war[~valid_mask]  # Fallback to XGB
    
    # Evaluate
    r2_xgb = r2_score(actual, xgb_pred_war)
    r2_marcel = r2_score(actual.values[valid_mask], marcel_preds[valid_mask])
    r2_ensemble = r2_score(actual, ensemble_pred)
    
    print(f"\nBacktest Results:")
    print(f"  XGB Only R²:    {r2_xgb:.4f}")
    print(f"  Marcel Only R²: {r2_marcel:.4f}")
    print(f"  Ensemble R²:    {r2_ensemble:.4f}")
    
    # Save results
    results_dir = Path(__file__).parent.parent / 'results' / 'WARprojections' / 'v5'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    backtest_df = df.loc[backtest_mask, ['season', name_col, 'target_WAR_next']].copy()
    backtest_df['xgb_pred'] = xgb_pred_war
    backtest_df['marcel_pred'] = marcel_preds
    backtest_df['ensemble_pred'] = ensemble_pred
    backtest_df.to_csv(results_dir / f'{role}_backtest_2025_v5.csv', index=False)
    
    # Forecast 2026
    X_forecast = df.loc[forecast_mask, features].fillna(0)
    expected_forecast = df.loc[forecast_mask, 'expected_war']
    
    xgb_pred_dev_26 = xgb_model.predict(X_forecast)
    xgb_pred_war_26 = expected_forecast.values + xgb_pred_dev_26
    
    forecast_names = df.loc[forecast_mask, name_col].values
    marcel_map_26 = marcel_26.set_index('player_name')['projected_war_2026'].to_dict()
    marcel_preds_26 = np.array([marcel_map_26.get(n, np.nan) for n in forecast_names])
    
    valid_26 = ~np.isnan(marcel_preds_26)
    ensemble_pred_26 = np.zeros_like(xgb_pred_war_26)
    ensemble_pred_26[valid_26] = stacker.predict(
        np.column_stack([xgb_pred_war_26[valid_26], marcel_preds_26[valid_26]])
    )
    ensemble_pred_26[~valid_26] = xgb_pred_war_26[~valid_26]
    
    forecast_df = df.loc[forecast_mask, ['season', name_col]].copy()
    forecast_df['projected_war_2026'] = ensemble_pred_26
    forecast_df.to_csv(results_dir / f'{role}_projections_2026_v5.csv', index=False)
    
    print(f"\nTop 10 Projected (2026):")
    print(forecast_df.nlargest(10, 'projected_war_2026').to_string(index=False))
    
    return r2_ensemble


def main():
    data = load_all_data()
    
    bat_r2 = train_and_evaluate_v5(data, 'batter')
    bowl_r2 = train_and_evaluate_v5(data, 'bowler')
    
    print("\n" + "="*60)
    print("V5 ENSEMBLE SUMMARY")
    print("="*60)
    print(f"  Batter R²: {bat_r2:.4f}")
    print(f"  Bowler R²: {bowl_r2:.4f}")
    print("\n✓ Results saved to results/WARprojections/v5/")


if __name__ == "__main__":
    main()
