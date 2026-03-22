"""
V9 Production Model - Trained on ALL data through 2025.

DIFFERENCE FROM V9 (backtest):
- V9 (backtest): Train 2008-2023, Validate on 2024→2025
- V9-Prod: Train 2008-2024 (includes 2024→2025), NO validation, Forecast 2026

This gives the most up-to-date model with enhanced features for 2026 predictions.

OUTPUT: results/WARprojections/v9_production/
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from pathlib import Path


def load_data():
    """Load V9 enhanced features and Marcel projections."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    results_dir = project_root / 'results' / 'WARprojections'
    
    # V9 enhanced features
    bat_features = pd.read_csv(data_dir / 'ml_features' / 'batter_features_v9.csv')
    bowl_features = pd.read_csv(data_dir / 'ml_features' / 'bowler_features_v9.csv')
    
    # Marcel projections for ensemble
    marcel_bat_26 = pd.read_csv(results_dir / 'marcel' / 'batter_projections_2026.csv')
    marcel_bowl_26 = pd.read_csv(results_dir / 'marcel' / 'bowler_projections_2026.csv')
    
    print(f"V9 Features: {len(bat_features)} batters, {len(bowl_features)} bowlers")
    
    return {
        'bat_features': bat_features,
        'bowl_features': bowl_features,
        'marcel_bat_26': marcel_bat_26,
        'marcel_bowl_26': marcel_bowl_26,
    }


def train_production_model(data, role='batter'):
    """Train production model on ALL data including 2024→2025."""
    print(f"\n{'='*60}")
    print(f"Training V9 PRODUCTION Model for {role.upper()}S")
    print(f"(Trained on 2008-2024, forecasting 2026)")
    print('='*60)
    
    df = data['bat_features'] if role == 'batter' else data['bowl_features']
    marcel_26 = data['marcel_bat_26'] if role == 'batter' else data['marcel_bowl_26']
    
    name_col = f'{role}_name'
    id_col = f'{role}_id'
    
    # Regression target - fit on ALL available data
    valid = df[df['target_WAR_next'].notna() & df['WAR_weighted'].notna()]
    reg = LinearRegression()
    reg.fit(valid[['WAR_weighted']].fillna(0), valid['target_WAR_next'])
    
    df = df.copy()
    df['expected_war'] = reg.predict(df[['WAR_weighted']].fillna(0))
    df['target_dev'] = df['target_WAR_next'] - df['expected_war']
    
    # V9 feature set (same as backtest version)
    base_features = ['WAR_weighted', 'consistency', 'career_war', 'years_played']
    if role == 'batter':
        base_features.extend(['balls_faced', 'bat_position'])
    else:
        base_features.append('balls_bowled')
    
    phase_features = ['phase_raa_per_ball_powerplay', 'phase_raa_per_ball_middle', 'phase_raa_per_ball_death']
    sit_features = ['sit_raa_per_ball_chasing', 'sit_raa_per_ball_setting']
    context_features = ['win_rate']
    opponent_features = ['opponent_adj_raa_per_ball']
    form_features = ['last_5_form', 'last_10_form', 'decay_weighted_form', 
                     'form_volatility', 'form_trend']
    legacy_form = ['last_5_matches_raa']
    
    all_features = (base_features + phase_features + sit_features + context_features +
                    opponent_features + form_features + legacy_form)
    features = [f for f in all_features if f in df.columns]
    
    print(f"Using {len(features)} features")
    
    # PRODUCTION: Train on 2008-2024 (one more year than backtest)
    train_mask = (df['season'] <= 2024) & (df['target_dev'].notna())
    forecast_mask = (df['season'] == 2025)
    
    X_train = df.loc[train_mask, features].copy()
    y_train = df.loc[train_mask, 'target_dev'].copy()
    
    X_forecast = df.loc[forecast_mask, features].copy()
    expected_forecast = df.loc[forecast_mask, 'expected_war'].values
    
    print(f"Train samples: {len(X_train)} (includes 2024→2025)")
    print(f"Forecast samples: {len(X_forecast)}")
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train), columns=features, index=X_train.index
    )
    X_forecast_imputed = pd.DataFrame(
        imputer.transform(X_forecast), columns=features, index=X_forecast.index
    ) if len(X_forecast) > 0 else pd.DataFrame()
    
    # Role-specific hyperparameters
    if role == 'batter':
        xgb_config = {
            'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.03,
            'reg_alpha': 1.0, 'reg_lambda': 2.0, 'subsample': 0.7, 'colsample_bytree': 0.7
        }
        rf_config = {'n_estimators': 100, 'max_depth': 5, 'min_samples_split': 10}
    else:
        xgb_config = {
            'n_estimators': 50, 'max_depth': 2, 'learning_rate': 0.03,
            'reg_alpha': 2.0, 'reg_lambda': 4.0, 'subsample': 0.7, 'colsample_bytree': 0.7
        }
        rf_config = {'n_estimators': 100, 'max_depth': 4, 'min_samples_split': 10}
    
    # Train models
    xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1, **xgb_config)
    xgb_model.fit(X_train_imputed, y_train)
    
    rf_model = RandomForestRegressor(random_state=42, n_jobs=-1, **rf_config)
    rf_model.fit(X_train_imputed, y_train)
    
    # Use V9 learned weights from backtest
    if role == 'batter':
        weights = np.array([0.195, 0.555, 0.250])  # XGB, RF, Marcel
    else:
        weights = np.array([-0.279, 0.556, 0.165])
    
    weights = weights / np.abs(weights).sum()
    print(f"Using V9 weights: XGB={weights[0]:.3f}, RF={weights[1]:.3f}, Marcel={weights[2]:.3f}")
    
    # Forecast 2026
    forecast_names = df.loc[forecast_mask, name_col].values
    forecast_ids = df.loc[forecast_mask, id_col].values
    
    xgb_pred_dev = xgb_model.predict(X_forecast_imputed)
    xgb_pred_war = expected_forecast + xgb_pred_dev
    
    rf_pred_dev = rf_model.predict(X_forecast_imputed)
    rf_pred_war = expected_forecast + rf_pred_dev
    
    marcel_map = marcel_26.set_index('player_name')['projected_war_2026'].to_dict()
    marcel_preds = np.array([marcel_map.get(n, np.nan) for n in forecast_names])
    valid_marcel = ~np.isnan(marcel_preds)
    
    # Ensemble prediction
    ensemble_pred = np.zeros_like(xgb_pred_war)
    ensemble_pred[valid_marcel] = (
        weights[0] * xgb_pred_war[valid_marcel] +
        weights[1] * rf_pred_war[valid_marcel] +
        weights[2] * marcel_preds[valid_marcel]
    )
    ensemble_pred[~valid_marcel] = (xgb_pred_war[~valid_marcel] + rf_pred_war[~valid_marcel]) / 2
    
    # Save
    results_dir = Path(__file__).parent.parent / 'results' / 'WARprojections' / 'v9_production'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    forecast_df = df.loc[forecast_mask, ['season', id_col, name_col]].copy()
    forecast_df['projected_war_2026'] = ensemble_pred
    forecast_df['xgb_pred'] = xgb_pred_war
    forecast_df['rf_pred'] = rf_pred_war
    forecast_df['marcel_pred'] = marcel_preds
    forecast_df.to_csv(results_dir / f'{role}_projections_2026_v9prod.csv', index=False)
    
    print(f"\nTop 15 Projected (2026 - V9 PRODUCTION):")
    print(forecast_df.nlargest(15, 'projected_war_2026')[[name_col, 'projected_war_2026']].to_string(index=False))
    
    return forecast_df


def main():
    print("=" * 60)
    print("V9 PRODUCTION MODEL")
    print("Training on ALL data (2008-2024) for 2026 forecasts")
    print("=" * 60)
    
    data = load_data()
    
    bat_forecast = train_production_model(data, 'batter')
    bowl_forecast = train_production_model(data, 'bowler')
    
    output_dir = Path(__file__).parent.parent / 'results' / 'WARprojections' / 'v9_production'
    
    print("\n" + "=" * 60)
    print("✓ V9 Production model complete!")
    print(f"  Results: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
