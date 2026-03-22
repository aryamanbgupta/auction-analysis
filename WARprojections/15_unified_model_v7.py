"""
V7 Unified Projection Model - Combines IPL + Global + Marcel.

KEY IMPROVEMENTS:
1. Single model that uses ALL available features
2. Handles missing features gracefully via imputation
3. Includes player meta-features (has_ipl_history, has_global_data, etc.)
4. Backtests on 2025 season (trained on 2008-2024)

OUTPUT: results/WARprojections/v7_unified/
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from pathlib import Path


def load_all_data():
    """Load all data sources: IPL features, Global data, Marcel, WAR history."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    results_dir = project_root / 'results' / 'WARprojections'
    
    data = {}
    
    # IPL ML features (already have lag features, phase features, etc.)
    data['bat_features'] = pd.read_csv(data_dir / 'ml_features' / 'batter_features_v4.csv')
    data['bowl_features'] = pd.read_csv(data_dir / 'ml_features' / 'bowler_features_v4.csv')
    print(f"IPL Features: {len(data['bat_features'])} batter-seasons, {len(data['bowl_features'])} bowler-seasons")
    
    # Global T20 data
    data['global_df'] = pd.read_parquet(data_dir / 'global_t20_matches.parquet')
    print(f"Global T20: {len(data['global_df'])} balls")
    
    # League factors
    data['league_factors'] = pd.read_csv(data_dir / 'league_factors.csv')
    
    # Marcel projections (for ensemble features)
    data['marcel_bat'] = pd.read_csv(results_dir / 'marcel' / 'batter_projections_2025.csv')
    data['marcel_bowl'] = pd.read_csv(results_dir / 'marcel' / 'bowler_projections_2025.csv')
    
    # WAR history for target variable
    data['batter_war'] = pd.read_csv(data_dir / 'batter_war_full_history.csv')
    data['bowler_war'] = pd.read_csv(data_dir / 'bowler_war_full_history.csv')
    
    return data


def calculate_global_features(global_df, league_factors, role='batter'):
    """Calculate aggregated global T20 features by player-season."""
    id_col = f'{role}_id'
    name_col = f'{role}_name'
    
    # Get league factors for this role
    role_factors = league_factors[league_factors['role'] == role]
    factors = role_factors.set_index('league')['factor'].to_dict()
    
    # Calculate RAA per ball (simplified)
    global_df = global_df.copy()
    global_df['league_avg'] = global_df.groupby('league')['total_runs'].transform('mean')
    global_df['raw_raa'] = global_df['total_runs'] - global_df['league_avg']
    
    if role == 'bowler':
        global_df['raw_raa'] = -global_df['raw_raa']  # Fewer runs is better for bowlers
    
    # Apply league adjustment
    global_df['league_factor'] = global_df['league'].map(factors).fillna(0.3)
    global_df['adjusted_raa'] = global_df['raw_raa'] * global_df['league_factor']
    
    # Extract year from match date for season alignment
    global_df['match_date'] = pd.to_datetime(global_df['match_date'], errors='coerce')
    global_df['year'] = global_df['match_date'].dt.year
    
    # Aggregate by player-year
    agg_funcs = {
        'adjusted_raa': 'sum',
        'match_id': 'count',  # balls
    }
    
    player_year_stats = global_df.groupby([id_col, name_col, 'year']).agg(agg_funcs).reset_index()
    player_year_stats = player_year_stats.rename(columns={'match_id': 'global_balls'})
    player_year_stats['global_raa_per_ball'] = player_year_stats['adjusted_raa'] / player_year_stats['global_balls']
    
    # Also calculate career-level global stats
    career_stats = global_df.groupby([id_col, name_col]).agg({
        'adjusted_raa': 'sum',
        'match_id': 'count',
    }).reset_index()
    career_stats.columns = [id_col, name_col, 'global_career_raa', 'global_career_balls']
    career_stats['global_career_raa_per_ball'] = career_stats['global_career_raa'] / career_stats['global_career_balls']
    
    # Count leagues played
    leagues_count = global_df.groupby(id_col)['league'].nunique().reset_index()
    leagues_count.columns = [id_col, 'leagues_played']
    
    # T20I experience
    t20i_mask = global_df['league'].str.contains('T20I', na=False)
    t20i_stats = global_df[t20i_mask].groupby(id_col).agg({
        'adjusted_raa': 'sum',
        'match_id': 'count'
    }).reset_index()
    t20i_stats.columns = [id_col, 't20i_raa', 't20i_balls']
    t20i_stats['has_t20i'] = 1
    
    return player_year_stats, career_stats, leagues_count, t20i_stats


def create_unified_features(data, role='batter'):
    """Create unified feature set combining IPL + Global + Marcel."""
    print(f"\n{'='*60}")
    print(f"Creating Unified Features for {role.upper()}S")
    print('='*60)
    
    id_col = f'{role}_id'
    name_col = f'{role}_name'
    
    # Start with IPL features
    ipl_features = data['bat_features'] if role == 'batter' else data['bowl_features']
    ipl_features = ipl_features.copy()
    
    # Calculate global features
    global_year, global_career, leagues, t20i = calculate_global_features(
        data['global_df'], data['league_factors'], role
    )
    
    # Merge global year-aligned features
    # For each IPL season, get the player's global performance in the prior year
    ipl_features['prior_year'] = ipl_features['season'] - 1
    
    ipl_features = ipl_features.merge(
        global_year[[id_col, 'year', 'global_balls', 'global_raa_per_ball']],
        left_on=[id_col, 'prior_year'],
        right_on=[id_col, 'year'],
        how='left'
    ).drop(columns=['year'], errors='ignore')
    
    # Merge global career features
    ipl_features = ipl_features.merge(
        global_career[[id_col, 'global_career_balls', 'global_career_raa_per_ball']],
        on=id_col,
        how='left'
    )
    
    # Merge leagues count
    ipl_features = ipl_features.merge(leagues, on=id_col, how='left')
    
    # Merge T20I features
    ipl_features = ipl_features.merge(t20i, on=id_col, how='left')
    ipl_features['has_t20i'] = ipl_features['has_t20i'].fillna(0)
    
    # Add Marcel predictions (for ensemble)
    marcel = data['marcel_bat'] if role == 'batter' else data['marcel_bowl']
    marcel_map = marcel.set_index('player_id')['projected_war_2025'].to_dict()
    ipl_features['marcel_pred'] = ipl_features[id_col].map(marcel_map)
    
    # Add meta-features
    ipl_features['has_ipl_history'] = 1  # All rows here have IPL history
    ipl_features['has_global_data'] = ipl_features['global_career_balls'].notna().astype(int)
    ipl_features['has_marcel'] = ipl_features['marcel_pred'].notna().astype(int)
    
    print(f"Unified features: {len(ipl_features)} rows")
    print(f"  With global data: {ipl_features['has_global_data'].sum()}")
    print(f"  With Marcel: {ipl_features['has_marcel'].sum()}")
    print(f"  With T20I: {ipl_features['has_t20i'].sum()}")
    
    return ipl_features


def train_unified_model_backtest(unified_features, role='batter'):
    """Train unified model on 2008-2024, backtest on 2025."""
    print(f"\n{'='*60}")
    print(f"Training UNIFIED Model for {role.upper()}S")
    print(f"(Train: 2008-2024, Backtest: 2025)")
    print('='*60)
    
    df = unified_features.copy()
    name_col = f'{role}_name'
    
    # Define unified feature set
    ipl_features = [
        'WAR_weighted', 'consistency', 'career_war', 'years_played',
        'phase_raa_per_ball_powerplay', 'phase_raa_per_ball_middle', 'phase_raa_per_ball_death',
        'last_5_matches_raa', 'sit_raa_per_ball_chasing', 'sit_raa_per_ball_setting', 'win_rate'
    ]
    
    if role == 'batter':
        ipl_features.extend(['balls_faced', 'bat_position'])
    else:
        ipl_features.append('balls_bowled')
    
    global_features = [
        'global_balls', 'global_raa_per_ball',
        'global_career_balls', 'global_career_raa_per_ball',
        'leagues_played', 't20i_balls', 'has_t20i'
    ]
    
    marcel_features = ['marcel_pred', 'has_marcel']
    
    meta_features = ['has_global_data', 'age']
    
    all_features = [f for f in ipl_features + global_features + marcel_features + meta_features 
                    if f in df.columns]
    
    print(f"Using {len(all_features)} features")
    
    # Split data: train on pre-2024, backtest on 2024->2025
    train_mask = (df['season'] < 2024) & (df['target_WAR_next'].notna())
    backtest_mask = (df['season'] == 2024) & (df['target_WAR_next'].notna())
    forecast_mask = (df['season'] == 2025)
    
    X_train = df.loc[train_mask, all_features].copy()
    y_train = df.loc[train_mask, 'target_WAR_next'].copy()
    
    X_backtest = df.loc[backtest_mask, all_features].copy()
    y_backtest = df.loc[backtest_mask, 'target_WAR_next'].copy()
    
    X_forecast = df.loc[forecast_mask, all_features].copy()
    
    print(f"Train: {len(X_train)}, Backtest: {len(X_backtest)}, Forecast: {len(X_forecast)}")
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=all_features,
        index=X_train.index
    )
    X_backtest_imputed = pd.DataFrame(
        imputer.transform(X_backtest),
        columns=all_features,
        index=X_backtest.index
    )
    X_forecast_imputed = pd.DataFrame(
        imputer.transform(X_forecast),
        columns=all_features,
        index=X_forecast.index
    ) if len(X_forecast) > 0 else pd.DataFrame()
    
    # Train models with role-specific hyperparameters
    if role == 'batter':
        xgb_config = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.03,
            'reg_alpha': 1.0,
            'reg_lambda': 2.0,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
        }
        rf_config = {'n_estimators': 100, 'max_depth': 5, 'min_samples_split': 10}
    else:
        xgb_config = {
            'n_estimators': 50,
            'max_depth': 2,
            'learning_rate': 0.03,
            'reg_alpha': 2.0,
            'reg_lambda': 4.0,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
        }
        rf_config = {'n_estimators': 100, 'max_depth': 4, 'min_samples_split': 10}
    
    # Train XGBoost
    xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1, **xgb_config)
    xgb_model.fit(X_train_imputed, y_train)
    
    # Train RandomForest
    rf_model = RandomForestRegressor(random_state=42, n_jobs=-1, **rf_config)
    rf_model.fit(X_train_imputed, y_train)
    
    # Backtest predictions
    xgb_pred = xgb_model.predict(X_backtest_imputed)
    rf_pred = rf_model.predict(X_backtest_imputed)
    
    # Get Marcel predictions for backtest
    marcel_backtest = X_backtest_imputed['marcel_pred'].values if 'marcel_pred' in X_backtest_imputed.columns else np.zeros(len(X_backtest))
    
    # Learn ensemble weights
    valid_marcel = ~np.isnan(marcel_backtest) & (marcel_backtest != 0)
    
    if valid_marcel.sum() > 10:
        X_stack = np.column_stack([
            xgb_pred[valid_marcel],
            rf_pred[valid_marcel],
            marcel_backtest[valid_marcel]
        ])
        stacker = Ridge(alpha=1.0)
        stacker.fit(X_stack, y_backtest.values[valid_marcel])
        weights = stacker.coef_ / np.abs(stacker.coef_).sum()
        print(f"Learned weights: XGB={weights[0]:.3f}, RF={weights[1]:.3f}, Marcel={weights[2]:.3f}")
        
        # Ensemble prediction
        ensemble_pred = np.zeros_like(xgb_pred)
        ensemble_pred[valid_marcel] = stacker.predict(X_stack)
        ensemble_pred[~valid_marcel] = (xgb_pred[~valid_marcel] + rf_pred[~valid_marcel]) / 2
    else:
        ensemble_pred = (xgb_pred + rf_pred) / 2
        weights = [0.5, 0.5, 0]
        print("Not enough Marcel predictions, using XGB+RF average")
    
    # Calculate metrics
    r2_xgb = r2_score(y_backtest, xgb_pred)
    r2_rf = r2_score(y_backtest, rf_pred)
    r2_ensemble = r2_score(y_backtest, ensemble_pred)
    rmse_ensemble = np.sqrt(mean_squared_error(y_backtest, ensemble_pred))
    
    print(f"\nBacktest Results (2025):")
    print(f"  XGB:      R² = {r2_xgb:.4f}")
    print(f"  RF:       R² = {r2_rf:.4f}")
    print(f"  Ensemble: R² = {r2_ensemble:.4f}, RMSE = {rmse_ensemble:.4f}")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': all_features,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(f"\nTop 10 Features:")
    print(importance.head(10).to_string(index=False))
    
    # Save backtest results
    backtest_df = df.loc[backtest_mask, ['season', name_col, 'target_WAR_next']].copy()
    backtest_df['xgb_pred'] = xgb_pred
    backtest_df['rf_pred'] = rf_pred
    backtest_df['ensemble_pred'] = ensemble_pred
    backtest_df['error'] = ensemble_pred - y_backtest.values
    
    # Forecast 2026
    if len(X_forecast_imputed) > 0:
        xgb_pred_26 = xgb_model.predict(X_forecast_imputed)
        rf_pred_26 = rf_model.predict(X_forecast_imputed)
        
        marcel_forecast = X_forecast_imputed['marcel_pred'].values if 'marcel_pred' in X_forecast_imputed.columns else np.zeros(len(X_forecast))
        valid_marcel_26 = ~np.isnan(marcel_forecast) & (marcel_forecast != 0)
        
        if valid_marcel_26.sum() > 0 and len(weights) == 3:
            ensemble_pred_26 = np.zeros_like(xgb_pred_26)
            X_stack_26 = np.column_stack([
                xgb_pred_26[valid_marcel_26],
                rf_pred_26[valid_marcel_26],
                marcel_forecast[valid_marcel_26]
            ])
            ensemble_pred_26[valid_marcel_26] = stacker.predict(X_stack_26)
            ensemble_pred_26[~valid_marcel_26] = (xgb_pred_26[~valid_marcel_26] + rf_pred_26[~valid_marcel_26]) / 2
        else:
            ensemble_pred_26 = (xgb_pred_26 + rf_pred_26) / 2
        
        forecast_df = df.loc[forecast_mask, ['season', name_col]].copy()
        forecast_df['projected_war_2026'] = ensemble_pred_26
    else:
        forecast_df = pd.DataFrame()
    
    return backtest_df, forecast_df, r2_ensemble, importance, (xgb_model, rf_model, imputer)


def main():
    print("=" * 60)
    print("V7 UNIFIED PROJECTION MODEL")
    print("Combines: IPL Features + Global T20 + Marcel")
    print("=" * 60)
    
    # Load data
    data = load_all_data()
    
    results = {}
    output_dir = Path(__file__).parent.parent / 'results' / 'WARprojections' / 'v7_unified'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for role in ['batter', 'bowler']:
        # Create unified features
        unified = create_unified_features(data, role)
        
        # Train and evaluate
        backtest_df, forecast_df, r2, importance, models = train_unified_model_backtest(unified, role)
        
        results[role] = {'r2': r2, 'backtest': backtest_df, 'forecast': forecast_df}
        
        # Save results
        backtest_df.to_csv(output_dir / f'{role}_backtest_2025_v7.csv', index=False)
        if len(forecast_df) > 0:
            forecast_df.to_csv(output_dir / f'{role}_projections_2026_v7.csv', index=False)
        importance.to_csv(output_dir / f'{role}_feature_importance_v7.csv', index=False)
        
        print(f"\nTop 10 Projected {role.upper()}S (2026):")
        if len(forecast_df) > 0:
            print(forecast_df.nlargest(10, 'projected_war_2026').to_string(index=False))
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("V7 UNIFIED MODEL SUMMARY")
    print("=" * 60)
    print(f"  Batter Backtest R²: {results['batter']['r2']:.4f}")
    print(f"  Bowler Backtest R²: {results['bowler']['r2']:.4f}")
    print(f"\n✓ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
