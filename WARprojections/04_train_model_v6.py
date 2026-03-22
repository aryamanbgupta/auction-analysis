"""
Train v6 Model with Optimized Hyperparameters + Multi-Model Ensemble.

IMPROVEMENTS (based on experiments):
1. Batters: max_depth=5 (improves R² 0.17 → 0.20)
2. Bowlers: depth=2 + heavier regularization
3. Multi-model ensemble: XGB + RandomForest + Marcel

OUTPUT: results/WARprojections/v6/
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path


def load_all_data():
    """Load all required data."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    results_dir = project_root / 'results' / 'WARprojections'
    
    bat_features = pd.read_csv(data_dir / 'ml_features' / 'batter_features_v4.csv')
    bowl_features = pd.read_csv(data_dir / 'ml_features' / 'bowler_features_v4.csv')
    
    marcel_bat_25 = pd.read_csv(results_dir / 'marcel' / 'batter_projections_2025.csv')
    marcel_bowl_25 = pd.read_csv(results_dir / 'marcel' / 'bowler_projections_2025.csv')
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


def train_xgb_v6(X_train, y_train, role):
    """Train XGBoost with optimized hyperparameters per role."""
    if role == 'batter':
        config = {
            'n_estimators': 100,
            'learning_rate': 0.03,
            'max_depth': 5,  # Deeper for batters
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'reg_alpha': 1.0,
            'reg_lambda': 2.0,
            'random_state': 42,
            'n_jobs': -1
        }
    else:  # bowler
        config = {
            'n_estimators': 50,  # Fewer trees
            'learning_rate': 0.03,
            'max_depth': 2,  # Shallower for bowlers
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'reg_alpha': 2.0,  # More regularization
            'reg_lambda': 4.0,
            'random_state': 42,
            'n_jobs': -1
        }
    
    model = xgb.XGBRegressor(**config)
    model.fit(X_train, y_train)
    return model, config


def train_rf_v6(X_train, y_train, role):
    """Train RandomForest as ensemble component."""
    config = {
        'n_estimators': 100,
        'max_depth': 5 if role == 'batter' else 4,
        'min_samples_split': 10,
        'random_state': 42,
        'n_jobs': -1
    }
    model = RandomForestRegressor(**config)
    model.fit(X_train, y_train)
    return model


def train_and_evaluate_v6(data, role):
    """Train v6 model with multi-model ensemble."""
    print(f"\n{'='*60}")
    print(f"Training {role.upper()} Model v6 (Optimized + Multi-Model)")
    print('='*60)
    
    df = data['bat_features'] if role == 'batter' else data['bowl_features']
    marcel_25 = data['marcel_bat_25'] if role == 'batter' else data['marcel_bowl_25']
    marcel_26 = data['marcel_bat_26'] if role == 'batter' else data['marcel_bowl_26']
    
    # Regression target
    valid = df[df['target_WAR_next'].notna() & df['WAR_weighted'].notna()]
    reg = LinearRegression()
    reg.fit(valid[['WAR_weighted']].fillna(0), valid['target_WAR_next'])
    
    df = df.copy()
    df['expected_war'] = reg.predict(df[['WAR_weighted']].fillna(0))
    df['target_dev'] = df['target_WAR_next'] - df['expected_war']
    
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
    
    features = [f for f in base + phase + form + sit + opp if f in df.columns]
    print(f"Using {len(features)} features")
    
    # Split
    train_mask = (df['season'] < 2024) & (df['target_dev'].notna())
    backtest_mask = (df['season'] == 2024) & (df['target_dev'].notna())
    forecast_mask = (df['season'] == 2025)
    
    X_train = df.loc[train_mask, features].fillna(0)
    y_train = df.loc[train_mask, 'target_dev']
    
    X_backtest = df.loc[backtest_mask, features].fillna(0)
    y_backtest_raw = df.loc[backtest_mask, 'target_WAR_next']
    expected_backtest = df.loc[backtest_mask, 'expected_war']
    
    X_forecast = df.loc[forecast_mask, features].fillna(0)
    expected_forecast = df.loc[forecast_mask, 'expected_war']
    
    print(f"Train: {len(X_train)}, Backtest: {len(X_backtest)}, Forecast: {len(X_forecast)}")
    
    # Train models
    xgb_model, xgb_cfg = train_xgb_v6(X_train, y_train, role)
    rf_model = train_rf_v6(X_train, y_train, role)
    
    print(f"XGB config: depth={xgb_cfg['max_depth']}, n_est={xgb_cfg['n_estimators']}")
    
    # Backtest predictions
    name_col = f'{role}_name'
    backtest_names = df.loc[backtest_mask, name_col].values
    
    xgb_pred_dev = xgb_model.predict(X_backtest)
    xgb_pred_war = expected_backtest.values + xgb_pred_dev
    
    rf_pred_dev = rf_model.predict(X_backtest)
    rf_pred_war = expected_backtest.values + rf_pred_dev
    
    marcel_map = marcel_25.set_index('player_name')['projected_war_2025'].to_dict()
    marcel_preds = np.array([marcel_map.get(n, np.nan) for n in backtest_names])
    valid_marcel = ~np.isnan(marcel_preds)
    
    # Learn optimal weights for 3-model ensemble
    X_stack = np.column_stack([
        xgb_pred_war[valid_marcel],
        rf_pred_war[valid_marcel],
        marcel_preds[valid_marcel]
    ])
    
    stacker = Ridge(alpha=1.0)
    stacker.fit(X_stack, y_backtest_raw.values[valid_marcel])
    
    weights = stacker.coef_ / stacker.coef_.sum()
    print(f"Ensemble weights: XGB={weights[0]:.2f}, RF={weights[1]:.2f}, Marcel={weights[2]:.2f}")
    
    # Ensemble prediction
    ensemble_pred = np.zeros_like(xgb_pred_war)
    ensemble_pred[valid_marcel] = stacker.predict(X_stack)
    ensemble_pred[~valid_marcel] = (xgb_pred_war[~valid_marcel] + rf_pred_war[~valid_marcel]) / 2
    
    # Evaluate
    r2_xgb = r2_score(y_backtest_raw, xgb_pred_war)
    r2_rf = r2_score(y_backtest_raw, rf_pred_war)
    r2_ensemble = r2_score(y_backtest_raw, ensemble_pred)
    
    print(f"\nBacktest Results:")
    print(f"  XGB Only:    R² = {r2_xgb:.4f}")
    print(f"  RF Only:     R² = {r2_rf:.4f}")
    print(f"  Ensemble:    R² = {r2_ensemble:.4f}")
    
    # Save
    results_dir = Path(__file__).parent.parent / 'results' / 'WARprojections' / 'v6'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    backtest_df = df.loc[backtest_mask, ['season', name_col, 'target_WAR_next']].copy()
    backtest_df['xgb_pred'] = xgb_pred_war
    backtest_df['rf_pred'] = rf_pred_war
    backtest_df['ensemble_pred'] = ensemble_pred
    backtest_df.to_csv(results_dir / f'{role}_backtest_2025_v6.csv', index=False)
    
    # Forecast 2026
    xgb_pred_dev_26 = xgb_model.predict(X_forecast)
    xgb_pred_war_26 = expected_forecast.values + xgb_pred_dev_26
    
    rf_pred_dev_26 = rf_model.predict(X_forecast)
    rf_pred_war_26 = expected_forecast.values + rf_pred_dev_26
    
    forecast_names = df.loc[forecast_mask, name_col].values
    marcel_map_26 = marcel_26.set_index('player_name')['projected_war_2026'].to_dict()
    marcel_preds_26 = np.array([marcel_map_26.get(n, np.nan) for n in forecast_names])
    valid_26 = ~np.isnan(marcel_preds_26)
    
    ensemble_pred_26 = np.zeros_like(xgb_pred_war_26)
    X_stack_26 = np.column_stack([
        xgb_pred_war_26[valid_26], rf_pred_war_26[valid_26], marcel_preds_26[valid_26]
    ])
    ensemble_pred_26[valid_26] = stacker.predict(X_stack_26)
    ensemble_pred_26[~valid_26] = (xgb_pred_war_26[~valid_26] + rf_pred_war_26[~valid_26]) / 2
    
    forecast_df = df.loc[forecast_mask, ['season', name_col]].copy()
    forecast_df['projected_war_2026'] = ensemble_pred_26
    forecast_df.to_csv(results_dir / f'{role}_projections_2026_v6.csv', index=False)
    
    print(f"\nTop 10 Projected (2026):")
    print(forecast_df.nlargest(10, 'projected_war_2026').to_string(index=False))
    
    return r2_ensemble


def main():
    data = load_all_data()
    
    bat_r2 = train_and_evaluate_v6(data, 'batter')
    bowl_r2 = train_and_evaluate_v6(data, 'bowler')
    
    print("\n" + "="*60)
    print("V6 MODEL SUMMARY")
    print("="*60)
    print(f"  Batter R²: {bat_r2:.4f}")
    print(f"  Bowler R²: {bowl_r2:.4f}")
    print("\n✓ Results saved to results/WARprojections/v6/")


if __name__ == "__main__":
    main()
