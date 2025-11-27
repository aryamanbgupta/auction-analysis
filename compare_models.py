import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths
PROJECT_ROOT = Path(__file__).parent
MARCEL_DIR = PROJECT_ROOT / 'results' / '13_projections' / 'backtest_2025'
ML_DIR = PROJECT_ROOT / 'results' / '15_ml_projections'
ACTUAL_DIR = PROJECT_ROOT / 'results' / '09_vorp_war'

def load_comparison_data(role='batter'):
    print(f"Loading data for {role}...")
    # Load Marcel
    marcel_path = MARCEL_DIR / f'{role}_projections_2025.csv'
    if not marcel_path.exists():
        print(f"Error: Marcel file not found at {marcel_path}")
        return None
        
    marcel = pd.read_csv(marcel_path)
    # Adjust column names if needed
    name_col = f'{role}_name'
    if name_col not in marcel.columns and 'player_name' in marcel.columns:
        name_col = 'player_name'
    
    marcel = marcel.rename(columns={name_col: 'player_name', 'projected_war_2025': 'marcel_war'})
    
    # Load ML
    ml_path = ML_DIR / f'{role}_backtest_2025.csv'
    if not ml_path.exists():
        print(f"Error: ML file not found at {ml_path}")
        return None
        
    ml = pd.read_csv(ml_path)
    # Adjust column names if needed
    name_col = f'{role}_name'
    if name_col not in ml.columns and 'player_name' in ml.columns:
        name_col = 'player_name'
        
    ml = ml.rename(columns={name_col: 'player_name', 'predicted_war': 'ml_war'})
    
    # Load Actuals
    actual_path = ACTUAL_DIR / f'{role}_war.csv'
    if not actual_path.exists():
        print(f"Error: Actual file not found at {actual_path}")
        return None
        
    actual = pd.read_csv(actual_path)
    # Filter for 2025
    actual = actual[actual['season'] == 2025]
    
    name_col = f'{role}_name'
    if name_col not in actual.columns and 'player_name' in actual.columns:
        name_col = 'player_name'
        
    actual = actual.rename(columns={name_col: 'player_name', 'WAR': 'actual_war'})
    
    # Merge
    # We want intersection of all three to compare fairly
    merged = marcel[['player_name', 'marcel_war']].merge(
        ml[['player_name', 'ml_war']], on='player_name', how='inner'
    ).merge(
        actual[['player_name', 'actual_war']], on='player_name', how='inner'
    )
    
    return merged

def print_metrics(df, role):
    if df is None or len(df) == 0:
        print(f"No data for {role}")
        return
        
    print(f"\n--- {role.upper()} METRICS (n={len(df)}) ---")
    
    results = {}
    
    for model in ['marcel', 'ml']:
        y_true = df['actual_war']
        y_pred = df[f'{model}_war']
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        results[model] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        print(f"{model.capitalize()}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
        
    # Comparison
    print("\nComparison:")
    for metric in ['MAE', 'RMSE']:
        marcel_val = results['marcel'][metric]
        ml_val = results['ml'][metric]
        diff = marcel_val - ml_val
        better = "ML" if diff > 0 else "Marcel" # Lower is better for error
        print(f"  {metric}: {better} is better by {abs(diff):.4f} ({abs(diff)/marcel_val*100:.1f}%)")
        
    r2_diff = results['ml']['R2'] - results['marcel']['R2']
    better_r2 = "ML" if r2_diff > 0 else "Marcel" # Higher is better for R2
    print(f"  R2:   {better_r2} is better by {abs(r2_diff):.4f}")

def main():
    print("Comparing Marcel vs XGBoost Projections (2025 Backtest)")
    print("="*60)
    
    # Batters
    bat_df = load_comparison_data('batter')
    print_metrics(bat_df, 'Batting')
    
    print("\n" + "="*60)
    
    # Bowlers
    bowl_df = load_comparison_data('bowler')
    print_metrics(bowl_df, 'Bowling')

if __name__ == "__main__":
    main()
