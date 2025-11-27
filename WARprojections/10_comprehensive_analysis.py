"""
Comprehensive Model Comparison and Analysis.
Compares:
1. Marcel (Baseline)
2. IPL Only ML (Full History)
3. Global ML (Full History + Global Data)

Outputs:
- Performance Metrics (2025 Backtest)
- Top 10 Comparisons
- 2026 Projections Report
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_data():
    """Load all datasets."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    results_dir = project_root / 'results' / 'WARprojections'
    
    # Actuals
    bat_actual = pd.read_csv(data_dir / 'batter_war_full_history.csv')
    bat_actual = bat_actual[bat_actual['season'] == 2025][['batter_name', 'WAR']].rename(columns={'batter_name': 'player_name', 'WAR': 'actual_war'})
    
    bowl_actual = pd.read_csv(data_dir / 'bowler_war_full_history.csv')
    bowl_actual = bowl_actual[bowl_actual['season'] == 2025][['bowler_name', 'WAR']].rename(columns={'bowler_name': 'player_name', 'WAR': 'actual_war'})
    
    # Backtests (2025)
    # Marcel
    marcel_bat = pd.read_csv(results_dir / 'marcel' / 'batter_projections_2025.csv')
    marcel_bat = marcel_bat[['player_name', 'projected_war_2025']].rename(columns={'projected_war_2025': 'marcel_war'})
    
    marcel_bowl = pd.read_csv(results_dir / 'marcel' / 'bowler_projections_2025.csv')
    marcel_bowl = marcel_bowl[['player_name', 'projected_war_2025']].rename(columns={'projected_war_2025': 'marcel_war'})
    
    # IPL Only ML
    ipl_bat = pd.read_csv(results_dir / 'batter_backtest_2025.csv')
    ipl_bat = ipl_bat[['batter_name', 'predicted_war']].rename(columns={'batter_name': 'player_name', 'predicted_war': 'ipl_ml_war'})
    
    ipl_bowl = pd.read_csv(results_dir / 'bowler_backtest_2025.csv')
    ipl_bowl = ipl_bowl[['bowler_name', 'predicted_war']].rename(columns={'bowler_name': 'player_name', 'predicted_war': 'ipl_ml_war'})
    
    # Global ML
    global_bat = pd.read_csv(results_dir / 'batter_backtest_2025_global.csv')
    global_bat = global_bat[['batter_name', 'predicted_war']].rename(columns={'batter_name': 'player_name', 'predicted_war': 'global_ml_war'})
    
    global_bowl = pd.read_csv(results_dir / 'bowler_backtest_2025_global.csv')
    global_bowl = global_bowl[['bowler_name', 'predicted_war']].rename(columns={'bowler_name': 'player_name', 'predicted_war': 'global_ml_war'})
    
    # Forecasts (2026)
    # Marcel
    marcel_bat_26 = pd.read_csv(results_dir / 'marcel' / 'batter_projections_2026.csv')
    marcel_bat_26 = marcel_bat_26[['player_name', 'projected_war_2026']].rename(columns={'projected_war_2026': 'marcel_war'})
    
    marcel_bowl_26 = pd.read_csv(results_dir / 'marcel' / 'bowler_projections_2026.csv')
    marcel_bowl_26 = marcel_bowl_26[['player_name', 'projected_war_2026']].rename(columns={'projected_war_2026': 'marcel_war'})
    
    # IPL Only ML
    ipl_bat_26 = pd.read_csv(results_dir / 'batter_projections_2026.csv')
    ipl_bat_26 = ipl_bat_26[['batter_name', 'projected_war_2026']].rename(columns={'batter_name': 'player_name', 'projected_war_2026': 'ipl_ml_war'})
    
    ipl_bowl_26 = pd.read_csv(results_dir / 'bowler_projections_2026.csv')
    ipl_bowl_26 = ipl_bowl_26[['bowler_name', 'projected_war_2026']].rename(columns={'bowler_name': 'player_name', 'projected_war_2026': 'ipl_ml_war'})
    
    # Global ML
    global_bat_26 = pd.read_csv(results_dir / 'batter_projections_2026_global.csv')
    global_bat_26 = global_bat_26[['batter_name', 'projected_war_2026']].rename(columns={'batter_name': 'player_name', 'projected_war_2026': 'global_ml_war'})
    
    global_bowl_26 = pd.read_csv(results_dir / 'bowler_projections_2026_global.csv')
    global_bowl_26 = global_bowl_26[['bowler_name', 'projected_war_2026']].rename(columns={'bowler_name': 'player_name', 'projected_war_2026': 'global_ml_war'})
    
    return {
        'batter': {
            'actual': bat_actual,
            'backtest': {'marcel': marcel_bat, 'ipl_ml': ipl_bat, 'global_ml': global_bat},
            'forecast': {'marcel': marcel_bat_26, 'ipl_ml': ipl_bat_26, 'global_ml': global_bat_26}
        },
        'bowler': {
            'actual': bowl_actual,
            'backtest': {'marcel': marcel_bowl, 'ipl_ml': ipl_bowl, 'global_ml': global_bowl},
            'forecast': {'marcel': marcel_bowl_26, 'ipl_ml': ipl_bowl_26, 'global_ml': global_bowl_26}
        }
    }

def evaluate_models(data, role):
    """Evaluate models on 2025 backtest."""
    actual = data[role]['actual']
    backtests = data[role]['backtest']
    
    # Merge all
    merged = actual
    for name, df in backtests.items():
        merged = merged.merge(df, on='player_name', how='inner')
        
    print(f"\n--- {role.upper()} MODEL EVALUATION (2025) ---")
    print(f"Sample Size: {len(merged)}")
    
    metrics = []
    for model in ['marcel', 'ipl_ml', 'global_ml']:
        col = f'{model}_war'
        mae = mean_absolute_error(merged['actual_war'], merged[col])
        rmse = np.sqrt(mean_squared_error(merged['actual_war'], merged[col]))
        r2 = r2_score(merged['actual_war'], merged[col])
        metrics.append({'Model': model, 'MAE': mae, 'RMSE': rmse, 'R2': r2})
        
    metrics_df = pd.DataFrame(metrics)
    print(metrics_df.to_string(index=False))
    
    return merged, metrics_df

def generate_top_10_comparison(merged, role):
    """Generate Top 10 comparison tables."""
    print(f"\n--- {role.upper()} TOP 10 COMPARISON (2025) ---")
    
    # Top 10 Actual
    top_actual = merged.sort_values('actual_war', ascending=False).head(10)[['player_name', 'actual_war']]
    print("\nTop 10 Actual:")
    print(top_actual.to_string(index=False))
    
    # Top 10 Predicted by each model
    for model in ['marcel', 'ipl_ml', 'global_ml']:
        col = f'{model}_war'
        top_pred = merged.sort_values(col, ascending=False).head(10)[['player_name', col, 'actual_war']]
        print(f"\nTop 10 Predicted by {model}:")
        print(top_pred.to_string(index=False))

def generate_2026_projections(data, role):
    """Generate 2026 projections comparison."""
    forecasts = data[role]['forecast']
    
    # Merge all (outer join to keep all predictions)
    merged = forecasts['marcel']
    merged = merged.merge(forecasts['ipl_ml'], on='player_name', how='outer')
    merged = merged.merge(forecasts['global_ml'], on='player_name', how='outer')
    
    print(f"\n--- {role.upper()} 2026 PROJECTIONS ---")
    
    for model in ['marcel', 'ipl_ml', 'global_ml']:
        col = f'{model}_war'
        top_proj = merged.sort_values(col, ascending=False).head(10)[['player_name', col]]
        print(f"\nTop 10 Projected by {model}:")
        print(top_proj.to_string(index=False))
        
    return merged

def main():
    data = load_data()
    
    # Batters
    bat_merged, bat_metrics = evaluate_models(data, 'batter')
    generate_top_10_comparison(bat_merged, 'batter')
    bat_2026 = generate_2026_projections(data, 'batter')
    
    # Bowlers
    bowl_merged, bowl_metrics = evaluate_models(data, 'bowler')
    generate_top_10_comparison(bowl_merged, 'bowler')
    bowl_2026 = generate_2026_projections(data, 'bowler')
    
    # Helper to write dataframe to markdown
    def write_md_table(df, f):
        f.write("| " + " | ".join(df.columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(df.columns)) + " |\n")
        for _, row in df.iterrows():
            f.write("| " + " | ".join(str(x) for x in row.values) + " |\n")

    # Save Report
    report_path = Path(__file__).parent.parent / 'results' / 'WARprojections' / 'model_comparison_report.md'
    with open(report_path, 'w') as f:
        f.write("# Model Comparison Report\n\n")
        
        f.write("## 2025 Backtest Performance\n\n")
        f.write("### Batting\n")
        write_md_table(bat_metrics.round(4), f)
        f.write("\n\n")
        f.write("### Bowling\n")
        write_md_table(bowl_metrics.round(4), f)
        f.write("\n\n")
        
        f.write("## 2026 Projections (Top 10)\n\n")
        
        for role, df in [('Batters', bat_2026), ('Bowlers', bowl_2026)]:
            f.write(f"### {role}\n\n")
            f.write("| Rank | Marcel | IPL ML | Global ML |\n")
            f.write("|---|---|---|---|\n")
            
            # Get top 10 for each
            m = df.sort_values('marcel_war', ascending=False).head(10)['player_name'].reset_index(drop=True)
            i = df.sort_values('ipl_ml_war', ascending=False).head(10)['player_name'].reset_index(drop=True)
            g = df.sort_values('global_ml_war', ascending=False).head(10)['player_name'].reset_index(drop=True)
            
            for rank in range(10):
                f.write(f"| {rank+1} | {m.get(rank, '-')} | {i.get(rank, '-')} | {g.get(rank, '-')} |\n")
            f.write("\n")
            
    print(f"\nReport saved to {report_path}")

if __name__ == "__main__":
    main()
