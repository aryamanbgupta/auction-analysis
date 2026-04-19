"""
Train an XGBoost regressor on the training table to predict full group-stage
custom-fantasy points, benchmark via LOSO (2020-2025) against Marcel, and save
the final model trained on all 2012-2025 rows for use in 2026 projection.

Input:
  - data/training_table_custom.parquet   (from 13c)

Outputs:
  - results/FantasyProjections/ipl2026_custom/xgb_loso_backtest.csv
  - results/FantasyProjections/ipl2026_custom/xgb_feature_importance.csv
  - results/FantasyProjections/ipl2026_custom/xgb_model.json
"""
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = Path(__file__).parent.parent
TABLE = PROJECT_ROOT / 'data' / 'training_table_custom.parquet'
OUT_DIR = PROJECT_ROOT / 'results' / 'FantasyProjections' / 'ipl2026_custom'
OUT_BACKTEST = OUT_DIR / 'xgb_loso_backtest.csv'
OUT_FI = OUT_DIR / 'xgb_feature_importance.csv'
OUT_MODEL = OUT_DIR / 'xgb_model.json'

LOSO_SEASONS = list(range(2020, 2026))  # backtest 2020..2025
GROUP_STAGE_GAMES = 14

FEATURE_COLS = [
    'career_matches', 'career_ppm',
    'career_bat_ppm', 'career_bowl_ppm', 'career_field_ppm',
    'lag1_matches', 'lag1_ppm',
    'lag2_matches', 'lag2_ppm',
    'lag3_matches', 'lag3_ppm',
    'marcel_ppm', 'role_prior_ppm',
    'early_matches', 'early_ppm',
    'early_bat_ppm', 'early_bowl_ppm', 'early_field_ppm',
    'early_runs', 'early_wickets', 'early_overs',
    'role_BAT', 'role_BOWL', 'role_AR', 'role_WK',
]
TARGET = 'target_total'


def one_hot_role(df: pd.DataFrame) -> pd.DataFrame:
    for r in ['BAT', 'BOWL', 'AR', 'WK']:
        df[f'role_{r}'] = (df['role'] == r).astype(int)
    return df


def metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    err = pred - true
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'n': len(true)}


def main():
    print(f"Loading {TABLE}")
    df = pd.read_parquet(TABLE)
    df = one_hot_role(df)
    print(f"  {len(df):,} rows, {df['season'].nunique()} seasons")

    # Train ONLY on players who actually played in the target season
    df_train = df[df['played_in_S']].copy()
    print(f"  {len(df_train):,} played-in-S rows used for training")

    params = dict(
        objective='reg:squarederror',
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_estimators=800,
        random_state=42,
        n_jobs=-1,
    )

    # LOSO backtest
    print("\n── LOSO backtest (2020-2025) ──")
    backtest_rows = []
    summary_rows = []
    for S in LOSO_SEASONS:
        train = df_train[df_train['season'] != S]
        test = df_train[df_train['season'] == S]
        if test.empty:
            continue
        X_tr = train[FEATURE_COLS].values
        y_tr = train[TARGET].values
        X_te = test[FEATURE_COLS].values
        y_te = test[TARGET].values

        model = xgb.XGBRegressor(**params)
        # Early-stopping on a held-out slice of training (one prior season)
        # Use S-1 as validation if available, else random split
        val_season = S - 1
        if val_season in train['season'].values:
            tr_mask = train['season'] != val_season
            X_tr_inner = train.loc[tr_mask, FEATURE_COLS].values
            y_tr_inner = train.loc[tr_mask, TARGET].values
            X_val = train.loc[~tr_mask, FEATURE_COLS].values
            y_val = train.loc[~tr_mask, TARGET].values
            model.set_params(early_stopping_rounds=30)
            model.fit(X_tr_inner, y_tr_inner, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_tr, y_tr, verbose=False)

        pred = model.predict(X_te)
        pred = np.clip(pred, 0, None)

        # Marcel baseline = marcel_ppm × 14
        marcel_pred = test['marcel_ppm'].values * GROUP_STAGE_GAMES

        xgb_m = metrics(pred, y_te)
        mar_m = metrics(marcel_pred, y_te)
        print(f"  {S}: XGB  MAE={xgb_m['mae']:.1f}  RMSE={xgb_m['rmse']:.1f}  R²={xgb_m['r2']:.3f}  (n={xgb_m['n']})")
        print(f"        Marc MAE={mar_m['mae']:.1f}  RMSE={mar_m['rmse']:.1f}  R²={mar_m['r2']:.3f}")

        summary_rows.append({
            'season': S, 'n': xgb_m['n'],
            'xgb_mae': xgb_m['mae'], 'xgb_rmse': xgb_m['rmse'], 'xgb_r2': xgb_m['r2'],
            'marcel_mae': mar_m['mae'], 'marcel_rmse': mar_m['rmse'], 'marcel_r2': mar_m['r2'],
        })

        out = test[['season', 'player_name', 'player_id', 'role',
                    'target_matches', 'target_total',
                    'marcel_ppm', 'early_matches', 'early_ppm']].copy()
        out['xgb_pred'] = pred
        out['marcel_pred'] = marcel_pred
        out['xgb_err'] = pred - y_te
        out['marcel_err'] = marcel_pred - y_te
        backtest_rows.append(out)

    summary = pd.DataFrame(summary_rows)
    print("\n── Summary ──")
    print(summary.to_string(index=False))
    print(f"\n  Mean  XGB    MAE={summary['xgb_mae'].mean():.1f}  R²={summary['xgb_r2'].mean():.3f}")
    print(f"  Mean  Marcel MAE={summary['marcel_mae'].mean():.1f}  R²={summary['marcel_r2'].mean():.3f}")

    bt = pd.concat(backtest_rows, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bt.to_csv(OUT_BACKTEST, index=False)
    print(f"\n✓ {OUT_BACKTEST} ({len(bt):,} rows)")

    # Final model: train on ALL 2012-2025 played-in-S rows
    print("\n── Fitting final model on 2012-2025 ──")
    X = df_train[FEATURE_COLS].values
    y = df_train[TARGET].values
    final = xgb.XGBRegressor(**params)
    # hold out 2025 for early stopping
    tr_mask = df_train['season'] != 2025
    val_mask = df_train['season'] == 2025
    final.set_params(early_stopping_rounds=30)
    final.fit(df_train.loc[tr_mask, FEATURE_COLS].values,
              df_train.loc[tr_mask, TARGET].values,
              eval_set=[(df_train.loc[val_mask, FEATURE_COLS].values,
                         df_train.loc[val_mask, TARGET].values)],
              verbose=False)
    final.save_model(str(OUT_MODEL))
    print(f"✓ {OUT_MODEL}  (best iter={final.best_iteration})")

    # Feature importance
    fi = pd.DataFrame({
        'feature': FEATURE_COLS,
        'gain': final.feature_importances_,
    }).sort_values('gain', ascending=False)
    fi.to_csv(OUT_FI, index=False)
    print(f"✓ {OUT_FI}")
    print("\nTop 10 features by gain:")
    print(fi.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
