"""
Calculate League Strength Factors (MLE) relative to IPL.

IMPROVEMENTS (v2):
1. T20I stratification by opponent strength (Elite/Mixed/Associate)
2. Intercept in regression (baseline difficulty offset)
3. Separate batter and bowler factors
4. Weighted regression by sample size
5. Shrinkage toward conservative prior (0.3) instead of 1.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from tqdm import tqdm

# Strong T20I teams (Full Members with consistent IPL representation)
STRONG_TEAMS = {
    'India', 'Australia', 'England', 'Pakistan', 'South Africa',
    'New Zealand', 'West Indies', 'Sri Lanka', 'Bangladesh', 'Afghanistan'
}

# Prior factor for leagues with insufficient data (assume weak)
CONSERVATIVE_PRIOR = 0.3
PRIOR_STRENGTH = 30  # Shrinkage weight


def load_data():
    """Load global and IPL data."""
    data_dir = Path(__file__).parent.parent / 'data'
    
    # Load Global Data
    global_df = pd.read_parquet(data_dir / 'global_t20_matches.parquet')
    
    # Load IPL Data
    ipl_df = pd.read_parquet(data_dir / 'ipl_matches_all.parquet')
    ipl_df['league'] = 'IPL'
    
    # Align columns
    common_cols = ['league', 'match_id', 'season', 'batter_id', 'bowler_id', 
                   'batter_runs', 'total_runs', 'is_wicket', 'venue', 'innings', 'over', 'ball_in_over']
    
    # Add team columns for T20I stratification (if available)
    if 'team1' in global_df.columns:
        common_cols.extend(['team1', 'team2'])
    
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


def get_t20i_tier(team1, team2):
    """
    Classify T20I match by opponent strength.
    
    Returns:
        'T20I_Elite': Both teams strong (harder than IPL)
        'T20I_Mixed': One strong, one associate
        'T20I_Associate': Both associates (easier)
    """
    if pd.isna(team1) or pd.isna(team2):
        return 'T20I'  # Fallback if teams unknown
    
    teams = {str(team1).strip(), str(team2).strip()}
    strong_count = len(teams & STRONG_TEAMS)
    
    if strong_count == 2:
        return 'T20I_Elite'
    elif strong_count == 1:
        return 'T20I_Mixed'
    else:
        return 'T20I_Associate'


def stratify_t20i(df):
    """Split T20I matches into tiers based on opponent strength."""
    print("Stratifying T20I matches by opponent strength...")
    
    # Identify T20I rows
    t20i_mask = df['league'].str.upper().str.contains('T20I|T20 INTERNATIONAL', na=False, regex=True)
    
    if t20i_mask.sum() == 0:
        print("  No T20I matches found to stratify")
        return df
    
    # Check if team columns exist
    if 'team1' not in df.columns or 'team2' not in df.columns:
        print("  Warning: team1/team2 columns not found. Skipping T20I stratification.")
        return df
    
    # Apply stratification
    df.loc[t20i_mask, 'league'] = df.loc[t20i_mask].apply(
        lambda row: get_t20i_tier(row.get('team1'), row.get('team2')), axis=1
    )
    
    # Count tiers
    tier_counts = df[df['league'].str.startswith('T20I')]['league'].value_counts()
    print(f"  T20I Tiers: {tier_counts.to_dict()}")
    
    return df


def calculate_raw_raa(df):
    """
    Calculate simplified RAA for all balls.
    RAA = Runs Scored - Expected Runs (League Avg per Ball)
    """
    print("Calculating Raw RAA...")
    
    # Calculate average runs per ball by league and season
    league_avgs = df.groupby(['league', 'season'])['total_runs'].mean().reset_index()
    league_avgs.rename(columns={'total_runs': 'league_avg_runs'}, inplace=True)
    
    df = df.merge(league_avgs, on=['league', 'season'], how='left')
    
    # RAA = Actual - Expected
    df['batter_RAA'] = df['total_runs'] - df['league_avg_runs']
    
    return df


def calculate_league_factors(df, role='batter'):
    """
    Calculate League Difficulty Factors using improved methodology.
    
    IMPROVEMENTS:
    1. Adds intercept to regression
    2. Weighted by sample size
    3. Shrinkage toward conservative prior for sparse data
    
    Args:
        df: DataFrame with ball-by-ball data and RAA
        role: 'batter' or 'bowler'
    """
    print(f"\nCalculating League Factors for {role}s...")
    
    id_col = 'batter_id' if role == 'batter' else 'bowler_id'
    raa_col = 'batter_RAA'  # For bowlers, we'll negate later
    
    # Aggregate by Player-League-Season
    player_stats = df.groupby(['season', 'league', id_col]).agg({
        raa_col: 'sum',
        'match_id': 'count'
    }).rename(columns={'match_id': 'balls', raa_col: 'RAA_sum'}).reset_index()
    
    player_stats['RAA_per_ball'] = player_stats['RAA_sum'] / player_stats['balls']
    
    # For bowlers, negate RAA (giving up fewer runs is good)
    if role == 'bowler':
        player_stats['RAA_per_ball'] = -player_stats['RAA_per_ball']
    
    # Filter for meaningful sample
    player_stats = player_stats[player_stats['balls'] >= 30]
    
    # Get IPL stats for comparison
    ipl_stats = player_stats[player_stats['league'] == 'IPL'][
        ['season', id_col, 'RAA_per_ball', 'balls']
    ].rename(columns={'RAA_per_ball': 'IPL_RAA', 'balls': 'IPL_balls'})
    
    other_stats = player_stats[player_stats['league'] != 'IPL']
    
    # Merge to get overlapping players
    merged = other_stats.merge(ipl_stats, on=['season', id_col], how='inner')
    
    factors = []
    print(f"Found overlaps for {merged['league'].nunique()} leagues")
    
    for league in merged['league'].unique():
        league_data = merged[merged['league'] == league]
        n_players = len(league_data)
        
        # Calculate weights (use min of balls in each league)
        weights = np.minimum(league_data['balls'], league_data['IPL_balls'])
        
        if n_players < 5:
            # Too few players - use conservative prior
            print(f"  {league}: n={n_players} (< 5), using prior={CONSERVATIVE_PRIOR:.2f}")
            raw_factor = CONSERVATIVE_PRIOR
            ci_lower, ci_upper = 0.0, 1.0
            
        else:
            # Weighted regression WITH intercept: IPL_RAA = α + β × League_RAA
            X = sm.add_constant(league_data['RAA_per_ball'])
            y = league_data['IPL_RAA']
            
            try:
                model = sm.WLS(y, X, weights=weights).fit()
                raw_factor = model.params.iloc[1]  # Beta (slope)
                intercept = model.params.iloc[0]   # Alpha
                
                # Confidence interval for the slope
                ci = model.conf_int().iloc[1]
                ci_lower, ci_upper = ci[0], ci[1]
                
                print(f"  {league}: β={raw_factor:.4f} (α={intercept:.4f}), n={n_players}, CI=[{ci_lower:.3f}, {ci_upper:.3f}]")
                
            except Exception as e:
                print(f"  {league}: Regression failed ({e}), using prior")
                raw_factor = CONSERVATIVE_PRIOR
                ci_lower, ci_upper = 0.0, 1.0
        
        # Apply shrinkage toward conservative prior
        shrinkage = n_players / (n_players + PRIOR_STRENGTH)
        adjusted_factor = (raw_factor * shrinkage) + (CONSERVATIVE_PRIOR * (1 - shrinkage))
        
        factors.append({
            'league': league,
            'role': role,
            'raw_factor': raw_factor,
            'factor': adjusted_factor,
            'n_players': n_players,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'shrinkage': shrinkage
        })
    
    # Add IPL itself (factor = 1.0, no shrinkage)
    factors.append({
        'league': 'IPL',
        'role': role,
        'raw_factor': 1.0,
        'factor': 1.0,
        'n_players': 9999,
        'ci_lower': 1.0,
        'ci_upper': 1.0,
        'shrinkage': 1.0
    })
    
    return pd.DataFrame(factors)


def main():
    df = load_data()
    
    # Stratify T20I by opponent strength
    df = stratify_t20i(df)
    
    # Calculate RAA
    df = calculate_raw_raa(df)
    
    # Calculate separate factors for batters and bowlers
    batter_factors = calculate_league_factors(df, role='batter')
    bowler_factors = calculate_league_factors(df, role='bowler')
    
    # Combine
    all_factors = pd.concat([batter_factors, bowler_factors], ignore_index=True)
    
    # Print summary
    print("\n" + "="*60)
    print("LEAGUE FACTORS SUMMARY")
    print("="*60)
    
    for role in ['batter', 'bowler']:
        print(f"\n{role.upper()}S:")
        role_df = all_factors[all_factors['role'] == role].sort_values('factor', ascending=False)
        for _, row in role_df.iterrows():
            print(f"  {row['league']:20s}: {row['factor']:.3f} (raw={row['raw_factor']:.3f}, n={row['n_players']}, shrink={row['shrinkage']:.2f})")
    
    # Save
    output_dir = Path(__file__).parent.parent / 'data'
    all_factors.to_csv(output_dir / 'league_factors.csv', index=False)
    print(f"\n✓ Saved league factors to {output_dir / 'league_factors.csv'}")


if __name__ == "__main__":
    main()
