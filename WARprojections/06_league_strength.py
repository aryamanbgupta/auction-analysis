"""
Calculate League Strength Factors (MLE) relative to IPL.
Method: Compare RAA/ball of players who played in both IPL and League X in the same year.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from tqdm import tqdm

def load_data():
    """Load global and IPL data."""
    data_dir = Path(__file__).parent.parent / 'data'
    
    # Load Global Data
    global_df = pd.read_parquet(data_dir / 'global_t20_matches.parquet')
    
    # Load IPL Data (we need raw ball data to calculate comparable RAA)
    # Actually, we can just treat IPL as another league in the global dataset if we merge them?
    # But global dataset doesn't have IPL matches (we filtered them out or didn't include them).
    # Let's load IPL matches and append them with league='IPL'
    
    ipl_df = pd.read_parquet(data_dir / 'ipl_matches_all.parquet')
    ipl_df['league'] = 'IPL'
    
    # Align columns
    common_cols = ['league', 'match_id', 'season', 'batter_id', 'bowler_id', 
                   'batter_runs', 'total_runs', 'is_wicket', 'venue', 'innings', 'over', 'ball_in_over']
    
    # Ensure columns exist
    for col in common_cols:
        if col not in global_df.columns:
            global_df[col] = np.nan
        if col not in ipl_df.columns:
            ipl_df[col] = np.nan
            
    combined = pd.concat([
        ipl_df[common_cols],
        global_df[common_cols]
    ], ignore_index=True)
    
    return combined

def calculate_raw_raa(df):
    """
    Calculate simplified RAA for all balls.
    We use a simplified Expected Runs model here for speed and universality.
    RAA = Runs Scored - Expected Runs
    Expected Runs ~ League Avg Runs per Ball
    """
    print("Calculating Raw RAA...")
    
    # Calculate global average runs per ball by league and season
    league_avgs = df.groupby(['league', 'season'])['total_runs'].mean().reset_index()
    league_avgs.rename(columns={'total_runs': 'league_avg_runs'}, inplace=True)
    
    df = df.merge(league_avgs, on=['league', 'season'], how='left')
    
    # RAA = Actual Runs - League Average
    # This is a very basic RAA. Ideally we'd use the full context model.
    # But for league strength, comparing "performance relative to league average" is exactly what we want.
    # If a player is +0.5 runs/ball in SMAT and +0.1 runs/ball in IPL, SMAT is easier.
    
    df['batter_RAA'] = df['batter_runs'] - df['league_avg_runs'] # Using batter runs or total runs? 
    # Usually RAA uses total runs (including extras) for the team impact, 
    # but for individual batter skill, batter_runs is better?
    # Let's stick to total_runs for consistency with the main pipeline, 
    # assuming the batter is responsible for the outcome of the ball (mostly).
    # Actually, main pipeline uses weighted_run_value.
    # Let's use (Total Runs - League Avg Total Runs) as a proxy.
    
    df['batter_RAA'] = df['total_runs'] - df['league_avg_runs']
    
    return df

def calculate_league_factors(df):
    """
    Calculate League Difficulty Factors.
    Factor = IPL_RAA / League_RAA (for same players)
    """
    print("Calculating League Factors...")
    
    # Aggregate by Player-League-Season
    player_stats = df.groupby(['season', 'league', 'batter_id']).agg({
        'batter_RAA': 'sum',
        'match_id': 'count'
    }).rename(columns={'match_id': 'balls', 'batter_RAA': 'RAA_sum'}).reset_index()
    
    player_stats['RAA_per_ball'] = player_stats['RAA_sum'] / player_stats['balls']
    
    # Filter for meaningful sample
    player_stats = player_stats[player_stats['balls'] >= 30]
    
    # Pivot to compare IPL vs Others
    # We want pairs of (IPL, League X) for the same player-season
    
    ipl_stats = player_stats[player_stats['league'] == 'IPL'][['season', 'batter_id', 'RAA_per_ball']].rename(
        columns={'RAA_per_ball': 'IPL_RAA'}
    )
    
    other_stats = player_stats[player_stats['league'] != 'IPL']
    
    merged = other_stats.merge(ipl_stats, on=['season', 'batter_id'], how='inner')
    
    # Calculate Factor for each league
    # We want to find F such that RAA_IPL ~ F * RAA_League
    # So F = RAA_IPL / RAA_League
    # We can use regression: IPL_RAA ~ League_RAA (no intercept) -> Coef is the factor
    
    factors = []
    
    print(f"Found overlaps for {merged['league'].nunique()} leagues")
    
    for league in merged['league'].unique():
        league_data = merged[merged['league'] == league]
        n_players = len(league_data)
        
        if n_players < 10:
            print(f"  Skipping {league}: only {n_players} overlapping player-seasons")
            factors.append({'league': league, 'factor': 1.0, 'n': n_players}) # Default to 1.0
            continue
            
        # Regression: IPL_RAA = beta * League_RAA
        X = league_data['RAA_per_ball']
        y = league_data['IPL_RAA']
        
        model = sm.OLS(y, X).fit()
        factor = model.params.iloc[0]
        
        print(f"  {league}: Factor={factor:.4f} (n={n_players})")
        factors.append({'league': league, 'factor': factor, 'n': n_players})
        
    # Add IPL itself
    factors.append({'league': 'IPL', 'factor': 1.0, 'n': 9999})
    
    return pd.DataFrame(factors)

def main():
    df = load_data()
    df = calculate_raw_raa(df)
    
    factors_df = calculate_league_factors(df)
    
    # Save
    output_dir = Path(__file__).parent.parent / 'data'
    factors_df.to_csv(output_dir / 'league_factors.csv', index=False)
    print(f"\nSaved league factors to {output_dir / 'league_factors.csv'}")

if __name__ == "__main__":
    main()
