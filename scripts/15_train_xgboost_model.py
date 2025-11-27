import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_features():
    """Load feature datasets."""
    data_dir = Path(__file__).parent.parent / 'data' / 'ml_features'
    batter_features = pd.read_csv(data_dir / 'batter_features.csv')
    bowler_features = pd.read_csv(data_dir / 'bowler_features.csv')
    return batter_features, bowler_features

def train_and_evaluate(df, target_col, feature_cols, train_seasons, test_season, model_name):
    """
    Train XGBoost model and evaluate on test season.
    """
    print(f"\nTraining {model_name} Model...")
    print(f"Train Seasons: {train_seasons}")
    print(f"Test Season: {test_season}")
    
    # Split data
    train_df = df[df['season'].isin(train_seasons)]
    test_df = df[df['season'] == test_season]
    
    if len(train_df) == 0 or len(test_df) == 0:
        print("Error: Insufficient data for training or testing.")
        return None, None, None
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    # Train XGBoost
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Predict
    preds = model.predict(X_test)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"{model_name} Results (Test {test_season}):")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R2: {r2:.4f}")
    
    # Add predictions to test dataframe
    results_df = test_df.copy()
    results_df['predicted_war'] = preds
    results_df['residual'] = results_df[target_col] - results_df['predicted_war']
    
    return model, results_df, X_test

def plot_shap_summary(model, X, title, output_path):
    """Generate and save SHAP summary plot."""
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    # Load data
    batter_features, bowler_features = load_features()
    
    # Define features
    # Note: We use lagged features. 
    # 'recent_balls' is a proxy for experience/sample size.
    common_features = ['age', 'recent_balls']
    
    bat_features = common_features + [
        'WAR_lag1', 'RAA_lag1', 'balls_faced_lag1', 'WAR_per_ball_lag1',
        'consistency_score_lag1', 'home_advantage_lag1',
        'WAR_lag2', 'RAA_lag2', 'balls_faced_lag2', 'WAR_per_ball_lag2',
        'consistency_score_lag2', 'home_advantage_lag2'
    ]
    
    bowl_features = common_features + [
        'WAR_lag1', 'RAA_lag1', 'balls_bowled_lag1', 'WAR_per_ball_lag1',
        'consistency_score_lag1', 'home_advantage_lag1',
        'WAR_lag2', 'RAA_lag2', 'balls_bowled_lag2', 'WAR_per_ball_lag2',
        'consistency_score_lag2', 'home_advantage_lag2'
    ]
    
    # Output directory
    output_dir = Path(__file__).parent.parent / 'results' / '15_ml_projections'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Backtest: Train 2022-2024, Test 2025 ---
    # Note: We need at least one lag year. 
    # If we use lag1 and lag2, we need data from 2022 to predict 2024 (using 2023 and 2022 stats).
    # Our data starts in 2022.
    # So for 2022 season, lag1 is 2021 (missing).
    # For 2023 season, lag1 is 2022 (present), lag2 is 2021 (missing).
    # For 2024 season, lag1 is 2023, lag2 is 2022 (both present).
    # For 2025 season, lag1 is 2024, lag2 is 2023 (both present).
    
    # So we can effectively train on 2024 (using 2022-2023 history) and test on 2025.
    # Or train on 2023 (using 2022 history, lag2 missing) and 2024.
    
    # Let's use 2024 as training (target) and 2025 as test (target).
    # We can also include 2023 as training if we handle missing lag2.
    # XGBoost handles missing values, so we can include 2023.
    
    train_seasons = [2023, 2024]
    test_season = 2025
    
    # Batting Model
    bat_model, bat_results, bat_X_test = train_and_evaluate(
        batter_features, 'WAR', bat_features, train_seasons, test_season, "Batting WAR"
    )
    
    if bat_model:
        bat_results.to_csv(output_dir / 'batter_backtest_2025.csv', index=False)
        plot_shap_summary(bat_model, bat_X_test, "Batting WAR Feature Importance", output_dir / 'shap_batting.png')
        
    # Bowling Model
    bowl_model, bowl_results, bowl_X_test = train_and_evaluate(
        bowler_features, 'WAR', bowl_features, train_seasons, test_season, "Bowling WAR"
    )
    
    if bowl_model:
        bowl_results.to_csv(output_dir / 'bowler_backtest_2025.csv', index=False)
        plot_shap_summary(bowl_model, bowl_X_test, "Bowling WAR Feature Importance", output_dir / 'shap_bowling.png')

    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()
