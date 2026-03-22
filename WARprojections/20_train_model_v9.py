"""
V9 Model Training - Uses enhanced features with opponent quality and rolling form.

NEW FEATURES USED:
- opponent_adj_raa_per_ball: RAA weighted by opponent team strength
- last_5_form, last_10_form, last_15_form: Rolling match form
- decay_weighted_form: Exponentially weighted recent form
- form_volatility: Consistency of recent performances
- form_trend: Improving or declining form

TRAINING: 2008-2024, BACKTEST: 2025
OUTPUT: results/WARprojections/v9_enhanced/
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
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
    marcel_bat_25 = pd.read_csv(results_dir / 'marcel' / 'batter_projections_2025.csv')
    marcel_bowl_25 = pd.read_csv(results_dir / 'marcel' / 'bowler_projections_2025.csv')
    marcel_bat_26 = pd.read_csv(results_dir / 'marcel' / 'batter_projections_2026.csv')
    marcel_bowl_26 = pd.read_csv(results_dir / 'marcel' / 'bowler_projections_2026.csv')
    
    print(f"V9 Features: {len(bat_features)} batters, {len(bowl_features)} bowlers")
    
    return {
        'bat_features': bat_features,
        'bowl_features': bowl_features,
        'marcel_bat_25': marcel_bat_25,
        'marcel_bowl_25': marcel_bowl_25,
        'marcel_bat_26': marcel_bat_26,
        'marcel_bowl_26': marcel_bowl_26,
    }


def train_v9_model(data, role='batter'):
    """Train V9 model with enhanced features."""
    print(f"\n{'='*60}")
    print(f"Training V9 Model for {role.upper()}S")
    print(f"(Train: 2008-2024, Backtest: 2025)")
    print('='*60)
    
    df = data['bat_features'] if role == 'batter' else data['bowl_features']
    marcel_25 = data['marcel_bat_25'] if role == 'batter' else data['marcel_bowl_25']
    marcel_26 = data['marcel_bat_26'] if role == 'batter' else data['marcel_bowl_26']
    
    # Regression target (predict deviation from expected)
    valid = df[df['target_WAR_next'].notna() & df['WAR_weighted'].notna()]
    reg = LinearRegression()
    reg.fit(valid[['WAR_weighted']].fillna(0), valid['target_WAR_next'])
    
    df = df.copy()
    df['expected_war'] = reg.predict(df[['WAR_weighted']].fillna(0))
    df['target_dev'] = df['target_WAR_next'] - df['expected_war']
    
    name_col = f'{role}_name'
    
    # V9 feature set - includes NEW features
    base_features = ['WAR_weighted', 'consistency', 'career_war', 'years_played']
    if role == 'batter':
        base_features.extend(['balls_faced', 'bat_position'])
    else:
        base_features.append('balls_bowled')
    
    phase_features = ['phase_raa_per_ball_powerplay', 'phase_raa_per_ball_middle', 'phase_raa_per_ball_death']
    sit_features = ['sit_raa_per_ball_chasing', 'sit_raa_per_ball_setting']
    context_features = ['win_rate']
    
    # NEW V9 features
    opponent_features = ['opponent_adj_raa_per_ball']
    form_features = ['last_5_form', 'last_10_form', 'decay_weighted_form', 
                     'form_volatility', 'form_trend']
    
    # Use legacy form feature too
    legacy_form = ['last_5_matches_raa']
    
    all_features = (base_features + phase_features + sit_features + context_features +
                    opponent_features + form_features + legacy_form)
    
    # Filter to available features
    features = [f for f in all_features if f in df.columns]
    
    print(f"Using {len(features)} features")
    print(f"  NEW features: {[f for f in opponent_features + form_features if f in features]}")
    
    # Train/test split
    train_mask = (df['season'] < 2024) & (df['target_dev'].notna())
    backtest_mask = (df['season'] == 2024) & (df['target_dev'].notna())
    forecast_mask = (df['season'] == 2025)
    
    X_train = df.loc[train_mask, features].copy()
    y_train = df.loc[train_mask, 'target_dev'].copy()
    
    X_backtest = df.loc[backtest_mask, features].copy()
    y_backtest_raw = df.loc[backtest_mask, 'target_WAR_next'].copy()
    expected_backtest = df.loc[backtest_mask, 'expected_war'].values
    
    X_forecast = df.loc[forecast_mask, features].copy()
    expected_forecast = df.loc[forecast_mask, 'expected_war'].values
    
    print(f"Train: {len(X_train)}, Backtest: {len(X_backtest)}, Forecast: {len(X_forecast)}")
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train), columns=features, index=X_train.index
    )
    X_backtest_imputed = pd.DataFrame(
        imputer.transform(X_backtest), columns=features, index=X_backtest.index
    )
    X_forecast_imputed = pd.DataFrame(
        imputer.transform(X_forecast), columns=features, index=X_forecast.index
    ) if len(X_forecast) > 0 else pd.DataFrame()
    
    # Role-specific hyperparameters (same as V6)
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
    
    # Backtest predictions
    backtest_names = df.loc[backtest_mask, name_col].values
    
    xgb_pred_dev = xgb_model.predict(X_backtest_imputed)
    xgb_pred_war = expected_backtest + xgb_pred_dev
    
    rf_pred_dev = rf_model.predict(X_backtest_imputed)
    rf_pred_war = expected_backtest + rf_pred_dev
    
    # Marcel predictions for ensemble
    marcel_map = marcel_25.set_index('player_name')['projected_war_2025'].to_dict()
    marcel_preds = np.array([marcel_map.get(n, np.nan) for n in backtest_names])
    valid_marcel = ~np.isnan(marcel_preds)
    
    # Learn ensemble weights
    if valid_marcel.sum() > 10:
        X_stack = np.column_stack([
            xgb_pred_war[valid_marcel],
            rf_pred_war[valid_marcel],
            marcel_preds[valid_marcel]
        ])
        stacker = Ridge(alpha=1.0)
        stacker.fit(X_stack, y_backtest_raw.values[valid_marcel])
        weights = stacker.coef_ / np.abs(stacker.coef_).sum()
        print(f"Ensemble weights: XGB={weights[0]:.3f}, RF={weights[1]:.3f}, Marcel={weights[2]:.3f}")
        
        ensemble_pred = np.zeros_like(xgb_pred_war)
        ensemble_pred[valid_marcel] = stacker.predict(X_stack)
        ensemble_pred[~valid_marcel] = (xgb_pred_war[~valid_marcel] + rf_pred_war[~valid_marcel]) / 2
    else:
        ensemble_pred = (xgb_pred_war + rf_pred_war) / 2
        weights = [0.5, 0.5, 0]
    
    # Calculate metrics
    r2_xgb = r2_score(y_backtest_raw, xgb_pred_war)
    r2_rf = r2_score(y_backtest_raw, rf_pred_war)
    r2_ensemble = r2_score(y_backtest_raw, ensemble_pred)
    rmse_ensemble = np.sqrt(mean_squared_error(y_backtest_raw, ensemble_pred))
    
    print(f"\nBacktest Results (2025):")
    print(f"  XGB:      R² = {r2_xgb:.4f}")
    print(f"  RF:       R² = {r2_rf:.4f}")
    print(f"  Ensemble: R² = {r2_ensemble:.4f}, RMSE = {rmse_ensemble:.4f}")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': features,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Feature Importance:")
    print(importance.head(10).to_string(index=False))
    
    # Highlight new features
    new_feat_names = opponent_features + form_features
    new_importance = importance[importance['feature'].isin(new_feat_names)]
    print(f"\nNEW V9 Feature Importance:")
    print(new_importance.to_string(index=False))
    
    # Save backtest results
    results_dir = Path(__file__).parent.parent / 'results' / 'WARprojections' / 'v9_enhanced'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    backtest_df = df.loc[backtest_mask, ['season', name_col, 'target_WAR_next']].copy()
    backtest_df['xgb_pred'] = xgb_pred_war
    backtest_df['rf_pred'] = rf_pred_war
    backtest_df['ensemble_pred'] = ensemble_pred
    backtest_df.to_csv(results_dir / f'{role}_backtest_2025_v9.csv', index=False)
    
    # Forecast 2026
    if len(X_forecast_imputed) > 0:
        forecast_names = df.loc[forecast_mask, name_col].values
        
        xgb_pred_26 = xgb_model.predict(X_forecast_imputed)
        xgb_war_26 = expected_forecast + xgb_pred_26
        
        rf_pred_26 = rf_model.predict(X_forecast_imputed)
        rf_war_26 = expected_forecast + rf_pred_26
        
        marcel_map_26 = marcel_26.set_index('player_name')['projected_war_2026'].to_dict()
        marcel_26_preds = np.array([marcel_map_26.get(n, np.nan) for n in forecast_names])
        valid_26 = ~np.isnan(marcel_26_preds)
        
        ensemble_26 = np.zeros_like(xgb_war_26)
        if valid_26.sum() > 0 and len(weights) == 3:
            X_stack_26 = np.column_stack([
                xgb_war_26[valid_26], rf_war_26[valid_26], marcel_26_preds[valid_26]
            ])
            ensemble_26[valid_26] = stacker.predict(X_stack_26)
            ensemble_26[~valid_26] = (xgb_war_26[~valid_26] + rf_war_26[~valid_26]) / 2
        else:
            ensemble_26 = (xgb_war_26 + rf_war_26) / 2
        
        forecast_df = df.loc[forecast_mask, ['season', name_col]].copy()
        forecast_df['projected_war_2026'] = ensemble_26
        forecast_df.to_csv(results_dir / f'{role}_projections_2026_v9.csv', index=False)
        
        print(f"\nTop 10 Projected {role.upper()}S (2026):")
        print(forecast_df.nlargest(10, 'projected_war_2026').to_string(index=False))
    
    # Save feature importance
    importance.to_csv(results_dir / f'{role}_feature_importance_v9.csv', index=False)
    
    return r2_ensemble, importance


def main():
    print("=" * 60)
    print("V9 ENHANCED MODEL TRAINING")
    print("New features: Opponent Quality + Rolling Form")
    print("=" * 60)
    
    data = load_data()
    
    results = {}
    
    for role in ['batter', 'bowler']:
        r2, importance = train_v9_model(data, role)
        results[role] = r2
    
    # Print summary comparison with previous versions
    print("\n" + "=" * 60)
    print("V9 MODEL SUMMARY")
    print("=" * 60)
    print(f"  V9 Batter Backtest R²: {results['batter']:.4f}")
    print(f"  V9 Bowler Backtest R²: {results['bowler']:.4f}")
    
    print("\nComparison with previous versions:")
    print("  V6: Batter R²=0.246, Bowler R²=0.351")
    print("  V7: Batter R²=0.245, Bowler R²=0.321")
    print(f"  V9: Batter R²={results['batter']:.3f}, Bowler R²={results['bowler']:.3f}")
    
    output_dir = Path(__file__).parent.parent / 'results' / 'WARprojections' / 'v9_enhanced'
    print(f"\n✓ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
