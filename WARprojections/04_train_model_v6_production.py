"""
V6 Production Model - Trained on ALL available data (2008-2025).

DIFFERENCE FROM V6:
- V6 (backtest): Train 2008-2023, Validate on 2024→2025, Forecast 2026
- V6-prod: Train 2008-2024 (includes 2024→2025), NO validation, Forecast 2026

This gives the most up-to-date model for actual 2026 predictions.

OUTPUT: results/WARprojections/v6_production/
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score
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
    
    marcel_bat_26 = pd.read_csv(results_dir / 'marcel' / 'batter_projections_2026.csv')
    marcel_bowl_26 = pd.read_csv(results_dir / 'marcel' / 'bowler_projections_2026.csv')
    
    return {
        'bat_features': bat_features,
        'bowl_features': bowl_features,
        'marcel_bat_26': marcel_bat_26,
        'marcel_bowl_26': marcel_bowl_26,
    }


def train_production_model(data, role):
    """Train production model on ALL data including 2024→2025."""
    print(f"\n{'='*60}")
    print(f"Training {role.upper()} PRODUCTION Model")
    print(f"(Trained on 2008-2024, forecasting 2026)")
    print('='*60)
    
    df = data['bat_features'] if role == 'batter' else data['bowl_features']
    marcel_26 = data['marcel_bat_26'] if role == 'batter' else data['marcel_bowl_26']
    
    # Regression target - now fit on ALL available data
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
    
    # PRODUCTION: Train on 2008-2024 (one more year than backtest version)
    train_mask = (df['season'] <= 2024) & (df['target_dev'].notna())
    forecast_mask = (df['season'] == 2025)
    
    X_train = df.loc[train_mask, features].fillna(0)
    y_train = df.loc[train_mask, 'target_dev']
    
    X_forecast = df.loc[forecast_mask, features].fillna(0)
    expected_forecast = df.loc[forecast_mask, 'expected_war']
    
    print(f"Train samples: {len(X_train)} (includes 2024→2025)")
    print(f"Forecast samples: {len(X_forecast)}")
    
    # Train models with V6 optimized configs
    if role == 'batter':
        xgb_config = {'n_estimators': 100, 'max_depth': 5, 'reg_alpha': 1.0, 'reg_lambda': 2.0}
        rf_config = {'n_estimators': 100, 'max_depth': 5}
    else:
        xgb_config = {'n_estimators': 50, 'max_depth': 2, 'reg_alpha': 2.0, 'reg_lambda': 4.0}
        rf_config = {'n_estimators': 100, 'max_depth': 4}
    
    xgb_model = xgb.XGBRegressor(
        learning_rate=0.03, subsample=0.7, colsample_bytree=0.7,
        random_state=42, n_jobs=-1, **xgb_config
    )
    xgb_model.fit(X_train, y_train)
    
    rf_model = RandomForestRegressor(min_samples_split=10, random_state=42, n_jobs=-1, **rf_config)
    rf_model.fit(X_train, y_train)
    
    # Forecast 2026
    name_col = f'{role}_name'
    forecast_names = df.loc[forecast_mask, name_col].values
    
    xgb_pred_dev = xgb_model.predict(X_forecast)
    xgb_pred_war = expected_forecast.values + xgb_pred_dev
    
    rf_pred_dev = rf_model.predict(X_forecast)
    rf_pred_war = expected_forecast.values + rf_pred_dev
    
    marcel_map = marcel_26.set_index('player_name')['projected_war_2026'].to_dict()
    marcel_preds = np.array([marcel_map.get(n, np.nan) for n in forecast_names])
    valid_marcel = ~np.isnan(marcel_preds)
    
    # Use V6 learned weights (from backtest)
    if role == 'batter':
        weights = np.array([1.18, -0.43, 0.25])  # XGB, RF, Marcel
    else:
        weights = np.array([-0.69, 1.16, 0.53])  # XGB, RF, Marcel
    
    weights = weights / weights.sum()  # Normalize
    print(f"Using V6 weights: XGB={weights[0]:.2f}, RF={weights[1]:.2f}, Marcel={weights[2]:.2f}")
    
    # Ensemble prediction
    ensemble_pred = np.zeros_like(xgb_pred_war)
    ensemble_pred[valid_marcel] = (
        weights[0] * xgb_pred_war[valid_marcel] +
        weights[1] * rf_pred_war[valid_marcel] +
        weights[2] * marcel_preds[valid_marcel]
    )
    ensemble_pred[~valid_marcel] = (xgb_pred_war[~valid_marcel] + rf_pred_war[~valid_marcel]) / 2
    
    # Save
    results_dir = Path(__file__).parent.parent / 'results' / 'WARprojections' / 'v6_production'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    forecast_df = df.loc[forecast_mask, ['season', name_col]].copy()
    forecast_df['projected_war_2026'] = ensemble_pred
    forecast_df['xgb_pred'] = xgb_pred_war
    forecast_df['rf_pred'] = rf_pred_war
    forecast_df['marcel_pred'] = marcel_preds
    forecast_df.to_csv(results_dir / f'{role}_projections_2026_prod.csv', index=False)
    
    print(f"\nTop 10 Projected (2026 - PRODUCTION):")
    print(forecast_df.nlargest(10, 'projected_war_2026')[[name_col, 'projected_war_2026']].to_string(index=False))
    
    return forecast_df


def compare_models():
    """Compare V6 backtest vs V6 production forecasts."""
    results_dir = Path(__file__).parent.parent / 'results' / 'WARprojections'
    
    print("\n" + "="*60)
    print("COMPARISON: V6 (Backtest) vs V6 (Production)")
    print("="*60)
    
    for role in ['batter', 'bowler']:
        print(f"\n--- {role.upper()}S ---")
        
        v6_bt = pd.read_csv(results_dir / 'v6' / f'{role}_projections_2026_v6.csv')
        v6_prod = pd.read_csv(results_dir / 'v6_production' / f'{role}_projections_2026_prod.csv')
        
        name_col = f'{role}_name'
        
        # Merge
        merged = v6_bt.merge(
            v6_prod[[name_col, 'projected_war_2026']],
            on=name_col,
            suffixes=('_backtest', '_production')
        )
        
        merged['diff'] = merged['projected_war_2026_production'] - merged['projected_war_2026_backtest']
        
        # Stats
        print(f"Players compared: {len(merged)}")
        print(f"Average diff (prod - backtest): {merged['diff'].mean():.3f}")
        print(f"Std of diff: {merged['diff'].std():.3f}")
        print(f"Correlation: {merged['projected_war_2026_backtest'].corr(merged['projected_war_2026_production']):.4f}")
        
        print(f"\nTop 10 by Production model:")
        top10 = merged.nlargest(10, 'projected_war_2026_production')[[
            name_col, 'projected_war_2026_backtest', 'projected_war_2026_production', 'diff'
        ]]
        top10.columns = [name_col, 'V6_Backtest', 'V6_Prod', 'Diff']
        print(top10.to_string(index=False))
        
        # Biggest changes
        print(f"\nBiggest increases (prod vs backtest):")
        print(merged.nlargest(5, 'diff')[[name_col, 'projected_war_2026_backtest', 'projected_war_2026_production', 'diff']].to_string(index=False))


def main():
    data = load_all_data()
    
    bat_forecast = train_production_model(data, 'batter')
    bowl_forecast = train_production_model(data, 'bowler')
    
    compare_models()
    
    print("\n" + "="*60)
    print("✓ Production model complete!")
    print("  Results: results/WARprojections/v6_production/")
    print("="*60)


if __name__ == "__main__":
    main()
