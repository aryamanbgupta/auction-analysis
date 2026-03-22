"""
Train and compare models for fantasy point prediction.

Comparison A: Unified (no T20I) vs Unified + T20I features
Comparison B: Unified vs Role-Based (legacy)

XGBoost + RandomForest + Marcel ensemble.
Backtest: Train 2008-2023, predict 2025 season (using 2024 features).

OUTPUT: results/FantasyProjections/backtest_2025.csv, model comparison report
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent / 'WARprojections'))
from utils import save_dataframe


# ── Feature sets ───────────────────────────────────────────────────────────

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

FIELDING_FEATURES = [
    'catches_per_match',
]

FORM_FEATURES = [
    'fp_last_5', 'fp_last_10', 'fp_decay_form', 'fp_volatility', 'fp_trend',
]

OPPONENT_FEATURES = [
    'opp_adj_fp_avg',
]

ALL_FEATURES = (CORE_FEATURES + LAG_FEATURES + BATTING_FEATURES +
                BOWLING_FEATURES + FIELDING_FEATURES + FORM_FEATURES +
                OPPONENT_FEATURES)

T20I_FEATURES = [
    't20i_avg_fantasy_pts', 't20i_std_fantasy_pts', 't20i_max_fantasy_pts',
    't20i_decay_form', 't20i_matches_interseason',
    't20i_batting_avg_pts', 't20i_batting_sr', 't20i_boundary_rate',
    't20i_bowling_avg_pts', 't20i_bowling_econ', 't20i_wicket_rate',
    't20i_vs_ipl_ratio', 't20i_form_momentum', 't20i_recency_days',
    't20i_elite_share', 'has_t20i_data',
]


# ── Training helpers ───────────────────────────────────────────────────────

def prepare_data(df, features, min_matches=3):
    """Filter and prepare training data."""
    valid = df[df['target_avg_fp_next'].notna() & (df['matches'] >= min_matches)].copy()
    available = [f for f in features if f in valid.columns]
    return valid, available


def train_ensemble(X_train, y_train, X_test, role_config='default'):
    """Train XGB + RF ensemble, return predictions."""
    if role_config == 'bowler':
        xgb_cfg = dict(n_estimators=50, max_depth=2, learning_rate=0.05,
                        reg_alpha=2.0, reg_lambda=4.0, subsample=0.7, colsample_bytree=0.7)
        rf_cfg = dict(n_estimators=100, max_depth=3, min_samples_split=10)
    else:
        xgb_cfg = dict(n_estimators=100, max_depth=4, learning_rate=0.05,
                        reg_alpha=1.0, reg_lambda=2.0, subsample=0.7, colsample_bytree=0.7)
        rf_cfg = dict(n_estimators=100, max_depth=4, min_samples_split=8)

    imputer = SimpleImputer(strategy='median')
    X_tr_imp = imputer.fit_transform(X_train)
    X_te_imp = imputer.transform(X_test)
    # SimpleImputer may drop all-NaN columns; use get_feature_names_out
    try:
        kept_cols = imputer.get_feature_names_out()
    except AttributeError:
        kept_cols = X_train.columns[~np.all(np.isnan(X_train.values), axis=0)]
    X_tr = pd.DataFrame(X_tr_imp, columns=kept_cols, index=X_train.index)
    X_te = pd.DataFrame(X_te_imp, columns=kept_cols, index=X_test.index)

    xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1, **xgb_cfg)
    xgb_model.fit(X_tr, y_train)

    rf_model = RandomForestRegressor(random_state=42, n_jobs=-1, **rf_cfg)
    rf_model.fit(X_tr, y_train)

    xgb_pred = xgb_model.predict(X_te)
    rf_pred = rf_model.predict(X_te)

    # Marcel baseline: just use fantasy_pts_weighted (or lag1)
    marcel_col = 'fantasy_pts_weighted'
    if marcel_col in X_te.columns:
        marcel_pred = X_te[marcel_col].values
    elif 'avg_fantasy_pts_lag1' in X_te.columns:
        marcel_pred = X_te['avg_fantasy_pts_lag1'].values
    else:
        marcel_pred = np.full(len(X_te), y_train.mean())

    # Learn ensemble weights on training set (via Ridge)
    xgb_tr_pred = xgb_model.predict(X_tr)
    rf_tr_pred = rf_model.predict(X_tr)
    if marcel_col in X_tr.columns:
        marcel_tr = X_tr[marcel_col].values
    elif 'avg_fantasy_pts_lag1' in X_tr.columns:
        marcel_tr = X_tr['avg_fantasy_pts_lag1'].values
    else:
        marcel_tr = np.full(len(X_tr), y_train.mean())

    stack = np.column_stack([xgb_tr_pred, rf_tr_pred, marcel_tr])
    ridge = Ridge(alpha=1.0)
    ridge.fit(stack, y_train)
    weights = ridge.coef_
    weights = weights / np.abs(weights).sum()

    stack_test = np.column_stack([xgb_pred, rf_pred, marcel_pred])
    ensemble_pred = ridge.predict(stack_test)

    return {
        'ensemble': ensemble_pred,
        'xgb': xgb_pred,
        'rf': rf_pred,
        'marcel': marcel_pred,
        'weights': weights,
        'xgb_model': xgb_model,
        'rf_model': rf_model,
        'ridge': ridge,
        'imputer': imputer,
    }


def evaluate(y_true, y_pred, label=''):
    """Print evaluation metrics."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_t, y_p = y_true[mask], y_pred[mask]
    r2 = r2_score(y_t, y_p)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    mae = mean_absolute_error(y_t, y_p)
    spear = spearmanr(y_t, y_p).statistic
    print(f"  {label:30s}  R²={r2:.3f}  RMSE={rmse:.2f}  MAE={mae:.2f}  Spearman={spear:.3f}  (n={len(y_t)})")
    return {'r2': r2, 'rmse': rmse, 'mae': mae, 'spearman': spear, 'n': len(y_t)}


# ── Approach A: Unified model ──────────────────────────────────────────────

def train_unified(df, include_t20i=False, label=''):
    """Train a single model for all roles, optionally with T20I features."""
    suffix = ' + T20I' if include_t20i else ''
    header = label or f'UNIFIED MODEL{suffix}'
    print("\n" + "=" * 60)
    print(header)
    print("=" * 60)

    # One-hot encode role
    df = df.copy()
    for r in ['BAT', 'BOWL', 'AR', 'WK']:
        df[f'role_{r}'] = (df['role'] == r).astype(int)

    features = ALL_FEATURES + ['role_BAT', 'role_BOWL', 'role_AR', 'role_WK']
    if include_t20i:
        features = features + T20I_FEATURES
    valid, avail = prepare_data(df, features)

    train_mask = valid['season'] <= 2023
    test_mask = valid['season'] == 2024  # features from 2024, target = 2025 actual

    X_train = valid.loc[train_mask, avail]
    y_train = valid.loc[train_mask, 'target_avg_fp_next']
    X_test = valid.loc[test_mask, avail]
    y_test = valid.loc[test_mask, 'target_avg_fp_next']

    print(f"Train: {len(X_train)} samples (seasons <= 2023)")
    print(f"Test:  {len(X_test)} samples (2024 features → 2025 actuals)")

    results = train_ensemble(X_train, y_train, X_test)

    print(f"\nWeights: XGB={results['weights'][0]:.3f}, RF={results['weights'][1]:.3f}, Marcel={results['weights'][2]:.3f}")
    print("\nBacktest Results (2025):")
    metrics = {}
    for name, pred in [('XGBoost', results['xgb']), ('RandomForest', results['rf']),
                        ('Marcel', results['marcel']), ('Ensemble', results['ensemble'])]:
        metrics[name] = evaluate(y_test.values, pred, name)

    # Per-role breakdown
    print("\nPer-Role Breakdown (Ensemble):")
    test_roles = valid.loc[test_mask, 'role'].values
    for role in ['BAT', 'BOWL', 'AR', 'WK']:
        role_mask = test_roles == role
        if role_mask.sum() > 5:
            evaluate(y_test.values[role_mask], results['ensemble'][role_mask], f"  {role}")

    return metrics, results, valid, test_mask


# ── Approach B: Role-based models ──────────────────────────────────────────

def train_role_based(df):
    """Train separate models per role."""
    print("\n" + "=" * 60)
    print("APPROACH B: ROLE-BASED MODELS (separate BAT/BOWL/AR/WK)")
    print("=" * 60)

    valid, avail = prepare_data(df, ALL_FEATURES)
    all_preds = []
    all_actuals = []
    all_names = []
    all_roles = []
    role_metrics = {}

    for role in ['BAT', 'BOWL', 'AR', 'WK']:
        role_data = valid[valid['role'] == role]
        train_mask = role_data['season'] <= 2023
        test_mask = role_data['season'] == 2024

        X_train = role_data.loc[train_mask, avail]
        y_train = role_data.loc[train_mask, 'target_avg_fp_next']
        X_test = role_data.loc[test_mask, avail]
        y_test = role_data.loc[test_mask, 'target_avg_fp_next']

        if len(X_train) < 10 or len(X_test) < 3:
            print(f"\n  {role}: Skipping (train={len(X_train)}, test={len(X_test)})")
            continue

        config = 'bowler' if role == 'BOWL' else 'default'
        results = train_ensemble(X_train, y_train, X_test, role_config=config)

        print(f"\n  {role} (train={len(X_train)}, test={len(X_test)}):")
        print(f"    Weights: XGB={results['weights'][0]:.3f}, RF={results['weights'][1]:.3f}, Marcel={results['weights'][2]:.3f}")
        m = evaluate(y_test.values, results['ensemble'], f"  {role} Ensemble")
        role_metrics[role] = m

        all_preds.extend(results['ensemble'])
        all_actuals.extend(y_test.values)
        all_names.extend(role_data.loc[test_mask, 'player_name'].values)
        all_roles.extend([role] * len(y_test))

    # Overall metrics
    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)

    print(f"\n  OVERALL Role-Based:")
    overall = evaluate(all_actuals, all_preds, "  Combined")

    return overall, role_metrics, all_preds, all_actuals, all_names, all_roles


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    project_root = Path(__file__).parent.parent
    results_dir = project_root / 'results' / 'FantasyProjections'
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FANTASY POINTS MODEL TRAINING & COMPARISON")
    print("=" * 60)

    df = pd.read_csv(project_root / 'data' / 'fantasy_features.csv')
    print(f"Loaded {len(df)} player-seasons, {df['target_avg_fp_next'].notna().sum()} with target")

    # ── Check if T20I features are available ─────────────────────────
    has_t20i = 'has_t20i_data' in df.columns and df['has_t20i_data'].sum() > 0
    if has_t20i:
        n_t20i = df['has_t20i_data'].sum()
        print(f"\nT20I features detected: {n_t20i}/{len(df)} rows ({100*n_t20i/len(df):.1f}%)")
    else:
        print("\nNo T20I features found — running baseline only")

    # ── Train models ──────────────────────────────────────────────────
    # A) Baseline: Unified without T20I
    base_metrics, base_results, base_valid, base_test_mask = train_unified(
        df, include_t20i=False, label='MODEL A: UNIFIED BASELINE (no T20I)')

    # B) With T20I (if available)
    t20i_metrics, t20i_results, t20i_valid, t20i_test_mask = None, None, None, None
    if has_t20i:
        t20i_metrics, t20i_results, t20i_valid, t20i_test_mask = train_unified(
            df, include_t20i=True, label='MODEL B: UNIFIED + T20I FEATURES')

    # C) Legacy role-based comparison
    role_metrics_overall, role_metrics_by_role, role_preds, role_actuals, role_names, role_roles = train_role_based(df)

    # ── Comparison ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"\n{'Model':35s}  {'R²':>8s}  {'RMSE':>8s}  {'MAE':>8s}  {'Spearman':>10s}")
    print("-" * 75)

    u = base_metrics['Ensemble']
    print(f"{'A) Unified (no T20I)':35s}  {u['r2']:8.3f}  {u['rmse']:8.2f}  {u['mae']:8.2f}  {u['spearman']:10.3f}")

    if t20i_metrics:
        t = t20i_metrics['Ensemble']
        print(f"{'B) Unified + T20I':35s}  {t['r2']:8.3f}  {t['rmse']:8.2f}  {t['mae']:8.2f}  {t['spearman']:10.3f}")
        delta_r2 = t['r2'] - u['r2']
        delta_mae = u['mae'] - t['mae']  # positive = improvement
        print(f"\n  T20I impact: ΔR²={delta_r2:+.3f}  ΔMAE={delta_mae:+.2f}")
        if delta_r2 >= 0.02 or delta_mae >= 1.0:
            print("  → T20I features IMPROVE the model — recommend incorporating")
        elif delta_r2 > 0:
            print("  → T20I features show slight improvement — marginal benefit")
        else:
            print("  → T20I features do NOT improve the model")

        # Sub-analysis: capped players only (has_t20i_data == 1)
        if t20i_valid is not None:
            capped_mask = t20i_valid.loc[t20i_test_mask, 'has_t20i_data'] == 1
            if capped_mask.sum() > 10:
                y_capped = t20i_valid.loc[t20i_test_mask, 'target_avg_fp_next'].values[capped_mask]
                pred_capped_t20i = t20i_results['ensemble'][capped_mask]
                pred_capped_base = base_results['ensemble'][capped_mask]
                print(f"\n  Capped players only (n={capped_mask.sum()}):")
                evaluate(y_capped, pred_capped_base, "  A) Baseline (capped)")
                evaluate(y_capped, pred_capped_t20i, "  B) + T20I (capped)")

    r = role_metrics_overall
    print(f"{'C) Role-Based Combined':35s}  {r['r2']:8.3f}  {r['rmse']:8.2f}  {r['mae']:8.2f}  {r['spearman']:10.3f}")

    # Feature importances for T20I model
    if t20i_results:
        print(f"\n{'='*60}")
        print("FEATURE IMPORTANCES — Unified + T20I (XGBoost)")
        print("=" * 60)
        xgb_model = t20i_results['xgb_model']
        imputer = t20i_results['imputer']
        try:
            feat_names = imputer.get_feature_names_out()
        except AttributeError:
            feat_names = list(range(len(xgb_model.feature_importances_)))
        importances = pd.Series(xgb_model.feature_importances_, index=feat_names)
        top_feats = importances.nlargest(20)
        for feat, imp in top_feats.items():
            marker = " ← T20I" if str(feat).startswith('t20i_') or feat == 'has_t20i_data' else ""
            print(f"  {str(feat):40s}  {imp:.4f}{marker}")

        t20i_feat_importance = importances[[f for f in importances.index if str(f).startswith('t20i_') or f == 'has_t20i_data']]
        total_t20i_imp = t20i_feat_importance.sum()
        print(f"\n  Total T20I feature importance: {total_t20i_imp:.4f} ({100*total_t20i_imp/importances.sum():.1f}%)")

    # Pick winner
    best_r2 = u['r2']
    winner = 'unified'
    if t20i_metrics and t20i_metrics['Ensemble']['r2'] > best_r2:
        best_r2 = t20i_metrics['Ensemble']['r2']
        winner = 'unified_t20i'
    if r['r2'] > best_r2:
        winner = 'role_based'
    print(f"\n→ WINNER: {winner} (best R²={best_r2:.3f})")

    # ── Save backtest results ──────────────────────────────────────────
    test_data = base_valid.loc[base_test_mask].copy()
    test_data['predicted_fp'] = base_results['ensemble']
    test_data['xgb_pred'] = base_results['xgb']
    test_data['rf_pred'] = base_results['rf']
    test_data['marcel_pred'] = base_results['marcel']
    if t20i_results:
        test_data['predicted_fp_t20i'] = t20i_results['ensemble']
    backtest_cols = ['season', 'player_id', 'player_name', 'role',
                     'avg_fantasy_pts', 'target_avg_fp_next',
                     'predicted_fp', 'xgb_pred', 'rf_pred', 'marcel_pred']
    if t20i_results:
        backtest_cols.append('predicted_fp_t20i')
    test_data = test_data[[c for c in backtest_cols if c in test_data.columns]]
    save_dataframe(test_data, results_dir / 'backtest_2025_unified.csv', format='csv')

    # Save role-based backtest
    role_bt = pd.DataFrame({
        'player_name': role_names,
        'role': role_roles,
        'actual': role_actuals,
        'predicted': role_preds,
    })
    save_dataframe(role_bt, results_dir / 'backtest_2025_role_based.csv', format='csv')

    # Save comparison report
    models = ['Unified (no T20I)', 'Role-Based Combined']
    r2s = [u['r2'], r['r2']]
    rmses = [u['rmse'], r['rmse']]
    maes = [u['mae'], r['mae']]
    spearmans = [u['spearman'], r['spearman']]
    if t20i_metrics:
        t = t20i_metrics['Ensemble']
        models.append('Unified + T20I')
        r2s.append(t['r2'])
        rmses.append(t['rmse'])
        maes.append(t['mae'])
        spearmans.append(t['spearman'])
    report = pd.DataFrame({
        'model': models,
        'r2': r2s,
        'rmse': rmses,
        'mae': maes,
        'spearman': spearmans,
    })
    save_dataframe(report, results_dir / 'model_comparison.csv', format='csv')

    with open(results_dir / 'winner.txt', 'w') as f:
        f.write(winner)

    print(f"\n✓ Results saved to {results_dir}/")
    print(f"✓ Winner: {winner}")


if __name__ == '__main__':
    main()
