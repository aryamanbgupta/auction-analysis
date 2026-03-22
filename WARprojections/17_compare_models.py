"""
Compare All WAR Projection Models on 2025 Backtest.

MODELS COMPARED:
- Legacy V6 (backtest)
- V7 Unified (IPL + Global + Marcel)
- V8 Domestic (SMAT-focused for uncapped players)
- Marcel baseline

OUTPUT: results/WARprojections/model_comparison/
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from pathlib import Path


def load_all_backtests():
    """Load backtest results from all models."""
    project_root = Path(__file__).parent.parent
    results_dir = project_root / 'results' / 'WARprojections'
    
    backtests = {}
    
    # V6 (legacy best)
    try:
        v6_bat = pd.read_csv(results_dir / 'v6' / 'batter_backtest_2025_v6.csv')
        v6_bowl = pd.read_csv(results_dir / 'v6' / 'bowler_backtest_2025_v6.csv')
        backtests['V6'] = {'batter': v6_bat, 'bowler': v6_bowl}
        print(f"V6: {len(v6_bat)} batters, {len(v6_bowl)} bowlers")
    except Exception as e:
        print(f"V6 not loaded: {e}")
    
    # V7 Unified
    try:
        v7_bat = pd.read_csv(results_dir / 'v7_unified' / 'batter_backtest_2025_v7.csv')
        v7_bowl = pd.read_csv(results_dir / 'v7_unified' / 'bowler_backtest_2025_v7.csv')
        backtests['V7_Unified'] = {'batter': v7_bat, 'bowler': v7_bowl}
        print(f"V7 Unified: {len(v7_bat)} batters, {len(v7_bowl)} bowlers")
    except Exception as e:
        print(f"V7 not loaded: {e}")
    
    # Marcel
    try:
        marcel_bat = pd.read_csv(results_dir / 'marcel' / 'batter_projections_2025.csv')
        marcel_bowl = pd.read_csv(results_dir / 'marcel' / 'bowler_projections_2025.csv')
        
        # Need to match with actual 2025 WAR
        data_dir = project_root / 'data'
        actual_bat = pd.read_csv(data_dir / 'batter_war_full_history.csv')
        actual_bowl = pd.read_csv(data_dir / 'bowler_war_full_history.csv')
        
        # Get 2025 actuals
        actual_bat_25 = actual_bat[actual_bat['season'] == 2025][['batter_name', 'WAR']].copy()
        actual_bowl_25 = actual_bowl[actual_bowl['season'] == 2025][['bowler_name', 'WAR']].copy()
        
        # Merge Marcel projections with actuals
        marcel_bat = marcel_bat.merge(
            actual_bat_25, 
            left_on='player_name', 
            right_on='batter_name', 
            how='inner'
        )
        marcel_bat = marcel_bat.rename(columns={'projected_war_2025': 'marcel_pred', 'WAR': 'target_WAR_next'})
        
        marcel_bowl = marcel_bowl.merge(
            actual_bowl_25,
            left_on='player_name',
            right_on='bowler_name',
            how='inner'
        )
        marcel_bowl = marcel_bowl.rename(columns={'projected_war_2025': 'marcel_pred', 'WAR': 'target_WAR_next'})
        
        backtests['Marcel'] = {'batter': marcel_bat, 'bowler': marcel_bowl}
        print(f"Marcel: {len(marcel_bat)} batters, {len(marcel_bowl)} bowlers")
    except Exception as e:
        print(f"Marcel not loaded: {e}")
    
    return backtests


def calculate_metrics(actual, predicted):
    """Calculate R², MAE, RMSE."""
    r2 = r2_score(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return r2, mae, rmse


def compare_models(backtests):
    """Compare all models on backtest metrics."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON - 2025 BACKTEST")
    print("=" * 70)
    
    results = []
    
    for model_name, data in backtests.items():
        for role, df in data.items():
            # Determine column names
            if model_name == 'V6':
                pred_col = 'ensemble_pred'
                actual_col = 'target_WAR_next'
            elif model_name == 'V7_Unified':
                pred_col = 'ensemble_pred'
                actual_col = 'target_WAR_next'
            elif model_name == 'Marcel':
                pred_col = 'marcel_pred'
                actual_col = 'target_WAR_next'
            else:
                continue
            
            if pred_col not in df.columns or actual_col not in df.columns:
                print(f"Skipping {model_name} {role}: missing columns")
                continue
            
            # Remove NaN
            valid = df[[actual_col, pred_col]].dropna()
            
            if len(valid) == 0:
                continue
            
            r2, mae, rmse = calculate_metrics(valid[actual_col], valid[pred_col])
            
            results.append({
                'Model': model_name,
                'Role': role,
                'Samples': len(valid),
                'R²': r2,
                'MAE': mae,
                'RMSE': rmse
            })
    
    results_df = pd.DataFrame(results)
    
    # Print formatted table
    print("\n--- BATTERS ---")
    bat_results = results_df[results_df['Role'] == 'batter'].sort_values('R²', ascending=False)
    print(bat_results.to_string(index=False))
    
    print("\n--- BOWLERS ---")
    bowl_results = results_df[results_df['Role'] == 'bowler'].sort_values('R²', ascending=False)
    print(bowl_results.to_string(index=False))
    
    return results_df


def analyze_coverage(backtests):
    """Analyze model coverage for auction players."""
    print("\n" + "=" * 70)
    print("AUCTION COVERAGE ANALYSIS")
    print("=" * 70)
    
    project_root = Path(__file__).parent.parent
    results_dir = project_root / 'results' / 'WARprojections'
    
    # Load auction scoring results
    try:
        v1_auction = pd.read_csv(results_dir / 'auction_2026' / 'auction_pool_war_projections.csv')
        v2_auction = pd.read_csv(results_dir / 'auction_2026_v2' / 'auction_pool_war_projections_v2.csv')
        
        print("\n--- V1 (Name-only) vs V2 (ID-first) Auction Scoring ---")
        print(f"V1 Coverage: {(v1_auction['projected_war_2026'] != 0).sum()}/{len(v1_auction)}")
        print(f"V2 Coverage: {(v2_auction['projected_war_2026'] != 0).sum()}/{len(v2_auction)}")
        
        print("\nV2 Match Method Breakdown:")
        print(v2_auction['match_method'].value_counts())
        
        print("\nV2 Source Breakdown:")
        print(v2_auction['prediction_source'].value_counts())
        
    except Exception as e:
        print(f"Auction analysis error: {e}")
    
    # Check V8 domestic coverage
    try:
        v8_domestic = pd.read_csv(results_dir / 'v8_domestic' / 'all_domestic_predictions.csv')
        print(f"\nV8 Domestic: {len(v8_domestic)} players with no IPL history predicted")
    except Exception as e:
        print(f"V8 domestic error: {e}")


def generate_report(results_df):
    """Generate markdown comparison report."""
    output_dir = Path(__file__).parent.parent / 'results' / 'WARprojections' / 'model_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report = """# WAR Projection Model Comparison Report

## 2025 Backtest Results

### Performance Metrics

| Model | Role | Samples | R² | MAE | RMSE |
|-------|------|---------|-----|-----|------|
"""
    
    for _, row in results_df.sort_values(['Role', 'R²'], ascending=[True, False]).iterrows():
        report += f"| {row['Model']} | {row['Role']} | {row['Samples']} | {row['R²']:.4f} | {row['MAE']:.4f} | {row['RMSE']:.4f} |\n"
    
    report += """
### Key Findings

1. **Best Batter Model**: Check R² above
2. **Best Bowler Model**: Check R² above
3. **V7 Unified** integrates IPL + Global + Marcel features
4. **V8 Domestic** provides predictions for 173 players with no IPL history

### Recommendations

- Use V7 Unified for players with IPL history (best coverage + global context)
- Use V8 Domestic for uncapped players with SMAT/domestic data
- Fall back to Marcel for players with limited data
"""
    
    with open(output_dir / 'comparison_report.md', 'w') as f:
        f.write(report)
    
    results_df.to_csv(output_dir / 'model_metrics.csv', index=False)
    
    print(f"\n✓ Report saved to: {output_dir}")


def main():
    print("=" * 70)
    print("WAR PROJECTION MODEL COMPARISON")
    print("=" * 70)
    
    backtests = load_all_backtests()
    results_df = compare_models(backtests)
    analyze_coverage(backtests)
    generate_report(results_df)


if __name__ == "__main__":
    main()
