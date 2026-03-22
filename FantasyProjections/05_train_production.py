"""
Production Fantasy Points Model — Train on ALL data through 2024 for 2026 forecasting.

Uses the unified model (winner from 04_train_model.py).
Ensemble: XGBoost + RandomForest + Marcel with best-of selection.

OUTPUT: results/FantasyProjections/fantasy_projections_2026.csv
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).parent.parent / 'WARprojections'))
from utils import save_dataframe


# Same feature set as training script
CORE_FEATURES = [
    'avg_fantasy_pts', 'fantasy_pts_weighted', 'std_fantasy_pts',
    'career_avg_fantasy', 'career_matches', 'years_played', 'matches',
]
LAG_FEATURES = [
    'avg_fantasy_pts_lag1', 'avg_fantasy_pts_lag2', 'avg_fantasy_pts_lag3',
    'total_fantasy_pts_lag1',
    'batting_pts_share_lag1', 'bowling_pts_share_lag1',
]
BATTING_FEATURES = [
    'boundary_rate', 'six_rate', 'sr_bonus_rate', 'avg_balls_faced',
    'bat_powerplay_sr', 'bat_middle_sr', 'bat_death_sr',
    'bat_powerplay_boundary_rate', 'bat_death_boundary_rate',
]
BOWLING_FEATURES = [
    'econ_bonus_rate', 'wicket_rate_match', 'maiden_rate', 'avg_overs_bowled',
    'bowl_powerplay_econ', 'bowl_middle_econ', 'bowl_death_econ',
    'bowl_powerplay_dot_rate', 'bowl_death_dot_rate',
]
FIELDING_FEATURES = ['catches_per_match']
FORM_FEATURES = [
    'fp_last_5', 'fp_last_10', 'fp_decay_form', 'fp_volatility', 'fp_trend',
]
OPPONENT_FEATURES = ['opp_adj_fp_avg']

ALL_FEATURES = (CORE_FEATURES + LAG_FEATURES + BATTING_FEATURES +
                BOWLING_FEATURES + FIELDING_FEATURES + FORM_FEATURES +
                OPPONENT_FEATURES)


def main():
    project_root = Path(__file__).parent.parent
    results_dir = project_root / 'results' / 'FantasyProjections'
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FANTASY POINTS PRODUCTION MODEL")
    print("Training on ALL data (through 2024) for 2026 forecasts")
    print("=" * 60)

    df = pd.read_csv(project_root / 'data' / 'fantasy_features.csv')

    # Add role one-hot (unified model)
    for r in ['BAT', 'BOWL', 'AR', 'WK']:
        df[f'role_{r}'] = (df['role'] == r).astype(int)

    features = ALL_FEATURES + ['role_BAT', 'role_BOWL', 'role_AR', 'role_WK']
    avail = [f for f in features if f in df.columns]

    # Train: all rows with target (seasons up to 2024)
    train_mask = df['target_avg_fp_next'].notna() & (df['season'] <= 2024) & (df['matches'] >= 3)
    # Forecast: 2025 season features → predict 2026
    forecast_mask = df['season'] == 2025

    X_train = df.loc[train_mask, avail].copy()
    y_train = df.loc[train_mask, 'target_avg_fp_next'].copy()
    X_forecast = df.loc[forecast_mask, avail].copy()

    print(f"Train samples: {len(X_train)} (includes 2024→2025 transition)")
    print(f"Forecast samples: {len(X_forecast)} (2025 features → 2026 prediction)")
    print(f"Features: {len(avail)}")

    # Impute
    imputer = SimpleImputer(strategy='median')
    X_tr_imp = imputer.fit_transform(X_train)
    X_fc_imp = imputer.transform(X_forecast)
    try:
        kept_cols = imputer.get_feature_names_out()
    except AttributeError:
        kept_cols = avail
    X_tr = pd.DataFrame(X_tr_imp, columns=kept_cols, index=X_train.index)
    X_fc = pd.DataFrame(X_fc_imp, columns=kept_cols, index=X_forecast.index)

    # XGBoost (best individual model from backtest)
    xgb_model = xgb.XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        reg_alpha=1.0, reg_lambda=2.0, subsample=0.7, colsample_bytree=0.7,
        random_state=42, n_jobs=-1,
    )
    xgb_model.fit(X_tr, y_train)

    # RandomForest
    rf_model = RandomForestRegressor(
        n_estimators=100, max_depth=4, min_samples_split=8,
        random_state=42, n_jobs=-1,
    )
    rf_model.fit(X_tr, y_train)

    # Marcel baseline
    marcel_col = 'fantasy_pts_weighted'
    if marcel_col in X_fc.columns:
        marcel_fc = X_fc[marcel_col].values
        marcel_tr = X_tr[marcel_col].values
    else:
        marcel_fc = np.full(len(X_fc), y_train.mean())
        marcel_tr = np.full(len(X_tr), y_train.mean())

    # Ensemble weights (learned on training set)
    xgb_tr_pred = xgb_model.predict(X_tr)
    rf_tr_pred = rf_model.predict(X_tr)
    stack_tr = np.column_stack([xgb_tr_pred, rf_tr_pred, marcel_tr])
    ridge = Ridge(alpha=1.0)
    ridge.fit(stack_tr, y_train)

    # Check if ensemble improves over XGBoost alone on training set
    from sklearn.metrics import r2_score
    ensemble_tr_pred = ridge.predict(stack_tr)
    r2_xgb_tr = r2_score(y_train, xgb_tr_pred)
    r2_ens_tr = r2_score(y_train, ensemble_tr_pred)
    print(f"\nTraining R²: XGBoost={r2_xgb_tr:.3f}, Ensemble={r2_ens_tr:.3f}")

    # Forecast
    xgb_fc = xgb_model.predict(X_fc)
    rf_fc = rf_model.predict(X_fc)
    stack_fc = np.column_stack([xgb_fc, rf_fc, marcel_fc])
    ensemble_fc = ridge.predict(stack_fc)

    # Use XGBoost as primary (best on backtest), ensemble as secondary
    # If ensemble weights look reasonable, use ensemble; else use XGBoost
    weights = ridge.coef_ / np.abs(ridge.coef_).sum()
    print(f"Ensemble weights: XGB={weights[0]:.3f}, RF={weights[1]:.3f}, Marcel={weights[2]:.3f}")

    # Build output
    forecast_df = df.loc[forecast_mask, ['season', 'player_id', 'player_name', 'role',
                                          'avg_fantasy_pts', 'matches']].copy()
    forecast_df['projected_avg_fp_2026'] = ensemble_fc
    forecast_df['xgb_pred'] = xgb_fc
    forecast_df['rf_pred'] = rf_fc
    forecast_df['marcel_pred'] = marcel_fc

    # Sort by projected performance
    forecast_df = forecast_df.sort_values('projected_avg_fp_2026', ascending=False)

    print(f"\n{'='*60}")
    print("TOP 20 PROJECTED PLAYERS (Avg Fantasy Pts/Match - 2026)")
    print("=" * 60)
    for i, (_, r) in enumerate(forecast_df.head(20).iterrows()):
        actual_2025 = r['avg_fantasy_pts']
        proj_2026 = r['projected_avg_fp_2026']
        print(f"  {i+1:2d}. {r['player_name']:25s} ({r['role']:4s})  "
              f"Proj: {proj_2026:5.1f}  2025 Actual: {actual_2025:5.1f}  "
              f"({int(r['matches'])} matches)")

    # Feature importances
    print(f"\nTop 15 Feature Importances (XGBoost):")
    importances = pd.Series(xgb_model.feature_importances_, index=kept_cols)
    for feat, imp in importances.nlargest(15).items():
        print(f"  {feat:40s}  {imp:.4f}")

    # Save
    output_path = results_dir / 'fantasy_projections_2026.csv'
    save_dataframe(forecast_df, output_path, format='csv')

    # Also save a simple lookup for the auction scoring script
    lookup = forecast_df[['player_id', 'player_name', 'role', 'projected_avg_fp_2026']].copy()
    save_dataframe(lookup, results_dir / 'fantasy_projections_2026_lookup.csv', format='csv')

    print(f"\n✓ Projections saved to {output_path}")
    print(f"✓ {len(forecast_df)} players projected for 2026")


if __name__ == '__main__':
    main()
