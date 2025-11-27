"""
Calculate advanced metrics (XR, RAA, WAR) for the full IPL history (2008-2025).
Consolidated pipeline for efficiency.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import statsmodels.api as sm
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

def load_data():
    """Load full history ball-by-ball data."""
    path = Path(__file__).parent.parent / 'data' / 'ipl_matches_all.parquet'
    print(f"Loading data from {path}...")
    return pd.read_parquet(path)

def calculate_expected_runs(df):
    """
    Calculate Expected Runs (XR) using XGBoost.
    Features: balls_remaining, wickets_before, innings, season (to capture era effects).
    Target: runs_rest_of_innings.
    """
    print("\n--- Calculating Expected Runs (XR) ---")
    
    # Prepare target: runs scored from this ball to the end of the innings
    # We need to group by match and innings
    df = df.sort_values(['match_id', 'innings', 'over', 'ball_in_over'])
    
    # Calculate runs remaining
    # Reverse cumsum of total_runs within each innings
    df['runs_rest_of_innings'] = df.groupby(['match_id', 'innings'])['total_runs'].transform(lambda x: x[::-1].cumsum()[::-1])
    # The current ball's runs are included in 'runs_rest_of_innings', but XR is usually "expected runs from this state"
    # If we want "runs expected *after* this ball", we subtract current runs.
    # But usually XR is defined at the *start* of the delivery.
    # So 'runs_rest_of_innings' (inclusive) is correct for the state *before* the ball.
    
    # Features
    features = ['balls_remaining', 'wickets_before', 'innings', 'season']
    X = df[features].copy()
    y = df['runs_rest_of_innings']
    
    # Train XGBoost
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        n_jobs=-1,
        random_state=42
    )
    
    print("Training XR model...")
    model.fit(X, y)
    
    # Predict
    df['expected_runs'] = model.predict(X)
    print("XR calculated.")
    
    return df

def calculate_run_values(df):
    """
    Calculate Run Values.
    Run Value = Runs Scored + Change in Expected Runs
    """
    print("\n--- Calculating Run Values ---")
    
    # We need XR for the *next* state.
    # Shift expected_runs up by 1 within innings to get "next ball's XR"
    # For the last ball, next XR is 0.
    df['next_expected_runs'] = df.groupby(['match_id', 'innings'])['expected_runs'].shift(-1).fillna(0)
    
    # Run Value = Total Runs + (Next XR - Current XR)
    df['run_value'] = df['total_runs'] + (df['next_expected_runs'] - df['expected_runs'])
    
    # Handle wickets: The "next XR" logic handles it implicitly because 
    # the next ball will have higher 'wickets_before', thus lower XR.
    # So (Next XR - Current XR) will be negative, penalizing the wicket.
    
    print(f"Mean Run Value: {df['run_value'].mean():.4f}")
    return df

def calculate_leverage_index(df):
    """
    Calculate Leverage Index (LI).
    Simplified: Phase + Wickets + Score Diff (2nd innings).
    """
    print("\n--- Calculating Leverage Index ---")
    
    # Base LI by phase
    phase_li = {'powerplay': 0.9, 'middle': 1.0, 'death': 1.4}
    df['phase_li'] = df['phase'].map(phase_li).fillna(1.0)
    
    # Wickets factor (more wickets in hand = higher leverage)
    df['wickets_li'] = 1.0 + (10 - df['wickets_before']) * 0.05
    
    # Combine
    df['leverage_index'] = df['phase_li'] * df['wickets_li']
    
    # Normalize so mean LI is 1.0
    df['leverage_index'] = df['leverage_index'] / df['leverage_index'].mean()
    
    # Weighted Run Value
    df['weighted_run_value'] = df['run_value'] * df['leverage_index']
    
    print(f"Mean LI: {df['leverage_index'].mean():.4f}")
    return df

def calculate_raa(df):
    """
    Calculate RAA (Runs Above Average) using Context Adjustment Regression.
    RAA = Residuals of: weighted_run_value ~ venue + innings + season
    """
    print("\n--- Calculating RAA (Context Adjustments) ---")
    
    # Prepare regression data
    # We use 'season' as a categorical variable to capture era effects
    # We use 'venue' to capture park factors
    
    # Top venues only to avoid sparse dummy variables
    top_venues = df['venue'].value_counts().head(20).index
    df['venue_reg'] = df['venue'].apply(lambda x: x if x in top_venues else 'Other')
    
    # Create dummies
    X = pd.get_dummies(df[['venue_reg', 'season', 'innings']], drop_first=True, dtype=float)
    X = sm.add_constant(X)
    y = df['weighted_run_value']
    
    # Fit OLS
    print("Fitting Context Model...")
    model = sm.OLS(y, X).fit()
    
    # Get residuals (Batter RAA)
    df['batter_RAA'] = model.resid
    
    # Bowler RAA = -Batter RAA
    df['bowler_RAA'] = -df['batter_RAA']
    
    print("RAA calculated.")
    return df

def calculate_war(df):
    """
    Calculate WAR.
    1. Aggregate RAA by player-season.
    2. Calculate Replacement Level (e.g., 20th percentile of RAA/ball).
    3. VORP = RAA - (Replacement * Balls).
    4. WAR = VORP / RunsPerWin.
    """
    print("\n--- Calculating WAR ---")
    
    # Calculate Consistency (Std Dev of Match RAA) before aggregating
    # Batting Consistency
    bat_match_raa = df.groupby(['season', 'batter_id', 'match_id'])['batter_RAA'].sum().reset_index()
    bat_consistency = bat_match_raa.groupby(['season', 'batter_id'])['batter_RAA'].std().reset_index().rename(columns={'batter_RAA': 'consistency'})
    
    # Aggregate Batting
    bat_stats = df.groupby(['season', 'batter_id', 'batter_name']).agg({
        'batter_RAA': 'sum',
        'match_id': 'count' # balls faced
    }).rename(columns={'match_id': 'balls_faced', 'batter_RAA': 'RAA'}).reset_index()
    
    bat_stats = bat_stats.merge(bat_consistency, on=['season', 'batter_id'], how='left')
    bat_stats['RAA_per_ball'] = bat_stats['RAA'] / bat_stats['balls_faced']
    
    # Bowling Consistency
    bowl_match_raa = df.groupby(['season', 'bowler_id', 'match_id'])['bowler_RAA'].sum().reset_index()
    bowl_consistency = bowl_match_raa.groupby(['season', 'bowler_id'])['bowler_RAA'].std().reset_index().rename(columns={'bowler_RAA': 'consistency'})
    
    # Aggregate Bowling
    bowl_stats = df.groupby(['season', 'bowler_id', 'bowler_name']).agg({
        'bowler_RAA': 'sum',
        'match_id': 'count' # balls bowled
    }).rename(columns={'match_id': 'balls_bowled', 'bowler_RAA': 'RAA'}).reset_index()
    
    bowl_stats = bowl_stats.merge(bowl_consistency, on=['season', 'bowler_id'], how='left')
    bowl_stats['RAA_per_ball'] = bowl_stats['RAA'] / bowl_stats['balls_bowled']
    
    # Calculate Replacement Level per Season
    # We define replacement level as the RAA/ball of a player at the 20th percentile of volume-weighted performance?
    # Or just simple 20th percentile of players with > X balls.
    
    def get_replacement_level(stats_df, ball_col, quantile=0.2):
        # Filter for meaningful sample size (e.g., > 60 balls)
        qualified = stats_df[stats_df[ball_col] > 60]
        if len(qualified) == 0:
            return stats_df['RAA_per_ball'].mean()
        return qualified['RAA_per_ball'].quantile(quantile)
    
    # Batting Replacement
    bat_rep = bat_stats.groupby('season').apply(lambda x: get_replacement_level(x, 'balls_faced')).to_dict()
    bat_stats['rep_level'] = bat_stats['season'].map(bat_rep)
    bat_stats['VORP'] = bat_stats['RAA'] - (bat_stats['rep_level'] * bat_stats['balls_faced'])
    
    # Bowling Replacement
    bowl_rep = bowl_stats.groupby('season').apply(lambda x: get_replacement_level(x, 'balls_bowled')).to_dict()
    bowl_stats['rep_level'] = bowl_stats['season'].map(bowl_rep)
    bowl_stats['VORP'] = bowl_stats['RAA'] - (bowl_stats['rep_level'] * bowl_stats['balls_bowled'])
    
    # Runs Per Win (approx 13.5 for T20, can vary by season but let's keep constant or simple)
    RPW = 13.5
    
    bat_stats['WAR'] = bat_stats['VORP'] / RPW
    bowl_stats['WAR'] = bowl_stats['VORP'] / RPW
    
    return bat_stats, bowl_stats

def main():
    df = load_data()
    
    # 1. Expected Runs
    df = calculate_expected_runs(df)
    
    # 2. Run Values
    df = calculate_run_values(df)
    
    # 3. Leverage Index
    df = calculate_leverage_index(df)
    
    # 4. RAA
    df = calculate_raa(df)
    
    # 5. WAR
    bat_war, bowl_war = calculate_war(df)
    
    # Save
    output_dir = Path(__file__).parent.parent / 'data'
    bat_war.to_csv(output_dir / 'batter_war_full_history.csv', index=False)
    bowl_war.to_csv(output_dir / 'bowler_war_full_history.csv', index=False)
    
    print("\nTop 10 Batters (All Time WAR):")
    print(bat_war.groupby('batter_name')['WAR'].sum().sort_values(ascending=False).head(10))
    
    print("\nTop 10 Bowlers (All Time WAR):")
    print(bowl_war.groupby('bowler_name')['WAR'].sum().sort_values(ascending=False).head(10))
    
    print(f"\nSaved results to {output_dir}")

if __name__ == "__main__":
    main()
