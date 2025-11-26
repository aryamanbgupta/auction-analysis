"""
Calculate VORP (Value Over Replacement Player) and WAR (Wins Above Replacement).

This script:
1. Loads player RAA data from script 06
2. Loads replacement level thresholds from script 08
3. Calculates VORP: VORP = RAA - (avg.RAA_rep × balls)
4. Estimates Runs Per Win (RPW) from match results
5. Calculates WAR: WAR = VORP / RPW

VORP represents runs contributed above what a replacement-level player would provide.
WAR translates VORP into team wins.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import statsmodels.api as sm


def load_raa_data(input_dir: Path) -> tuple:
    """
    Load RAA data with replacement level flags.

    Args:
        input_dir: Directory with RAA CSV files

    Returns:
        Tuple of (batter_raa, bowler_raa) DataFrames
    """
    print(f"Loading RAA data from {input_dir}...")

    batter_file = input_dir / 'batter_raa_with_replacement.csv'
    bowler_file = input_dir / 'bowler_raa_with_replacement.csv'

    batter_raa = pd.read_csv(batter_file)
    bowler_raa = pd.read_csv(bowler_file)

    print(f"✓ Loaded {len(batter_raa)} batters")
    print(f"✓ Loaded {len(bowler_raa)} bowlers")

    return batter_raa, bowler_raa


def load_avg_raa_rep(input_dir: Path) -> dict:
    """
    Load average RAA for replacement-level players.

    Args:
        input_dir: Directory with avg_raa_replacement.json

    Returns:
        Dictionary with avg_raa_rep values
    """
    print(f"Loading avg.RAA_rep from {input_dir}...")

    avg_raa_file = input_dir / 'avg_raa_replacement.json'
    with open(avg_raa_file, 'r') as f:
        avg_raa_rep = json.load(f)

    print(f"✓ Loaded avg.RAA_rep for {len(avg_raa_rep['batting'])} seasons")
    return avg_raa_rep


def calculate_vorp(batter_raa: pd.DataFrame, bowler_raa: pd.DataFrame,
                  avg_raa_rep: dict) -> tuple:
    """
    Calculate VORP for batters and bowlers.

    VORP = RAA - (avg.RAA_rep × balls)

    Args:
        batter_raa: DataFrame with batter RAA
        bowler_raa: DataFrame with bowler RAA
        avg_raa_rep: Dictionary with avg_raa_rep values

    Returns:
        Tuple of (batter_vorp, bowler_vorp) DataFrames
    """
    print("\n" + "="*70)
    print("CALCULATING VORP (VALUE OVER REPLACEMENT)")
    print("="*70)

    # Calculate VORP for batters
    print("\nCalculating batter VORP...")
    
    # Map season to avg_raa_rep
    batter_raa['avg_raa_rep'] = batter_raa['season'].astype(str).map(avg_raa_rep['batting'])
    
    # Fill missing seasons with mean (fallback)
    if batter_raa['avg_raa_rep'].isnull().any():
        mean_rep = np.mean(list(avg_raa_rep['batting'].values()))
        batter_raa['avg_raa_rep'].fillna(mean_rep, inplace=True)
        print("  Warning: Some seasons missing replacement level, used mean.")

    batter_raa['VORP'] = (
        batter_raa['RAA'] -
        (batter_raa['avg_raa_rep'] * batter_raa['balls_faced'])
    )
    batter_raa['VORP_per_ball'] = batter_raa['VORP'] / batter_raa['balls_faced']

    print(f"✓ Calculated VORP for {len(batter_raa)} batters")
    print(f"  Mean VORP: {batter_raa['VORP'].mean():.2f}")

    # Calculate VORP for bowlers
    print("\nCalculating bowler VORP...")
    
    # Map season to avg_raa_rep
    bowler_raa['avg_raa_rep'] = bowler_raa['season'].astype(str).map(avg_raa_rep['bowling'])
    
    # Fill missing seasons with mean
    if bowler_raa['avg_raa_rep'].isnull().any():
        mean_rep = np.mean(list(avg_raa_rep['bowling'].values()))
        bowler_raa['avg_raa_rep'].fillna(mean_rep, inplace=True)

    bowler_raa['VORP'] = (
        bowler_raa['RAA'] -
        (bowler_raa['avg_raa_rep'] * bowler_raa['balls_bowled'])
    )
    bowler_raa['VORP_per_ball'] = bowler_raa['VORP'] / bowler_raa['balls_bowled']

    print(f"✓ Calculated VORP for {len(bowler_raa)} bowlers")
    print(f"  Mean VORP: {bowler_raa['VORP'].mean():.2f}")

    return batter_raa, bowler_raa


def estimate_runs_per_win(data_file: Path) -> float:
    """
    Estimate Runs Per Win (RPW) from match results.

    RPW = 1/β from regression: W_i = β₀ + β·RunDiff_i + εᵢ

    where:
    - W_i = 1 if team won, 0 if lost
    - RunDiff_i = team's run differential for the match

    Args:
        data_file: Path to ball-by-ball data with match results

    Returns:
        Runs per win estimate
    """
    print("\n" + "="*70)
    print("ESTIMATING RUNS PER WIN (RPW)")
    print("="*70)

    print(f"\nLoading match data from {data_file}...")
    df = pd.read_parquet(data_file)

    # Extract match-level results
    print("\nExtracting match-level results...")

    # Get final score for each innings (max cumulative runs)
    innings_final = df.groupby(['match_id', 'innings', 'batting_team']).agg({
        'score_before': 'max',  # Maximum score before delivery
        'total_runs': 'sum'     # Sum of all runs in delivery
    }).reset_index()

    # Calculate final score as max score_before + sum of total_runs
    # Actually, we need the final score which is already in the data
    # Let's use a different approach - get the maximum total score

    # Better approach: Calculate cumulative score per match-innings
    df_sorted = df.sort_values(['match_id', 'innings', 'over', 'ball_in_over'])
    df_sorted['cumulative_runs'] = df_sorted.groupby(['match_id', 'innings'])['total_runs'].cumsum()

    # Get final score for each innings
    innings_final = df_sorted.groupby(['match_id', 'innings', 'batting_team']).agg({
        'cumulative_runs': 'last'  # Final cumulative runs
    }).reset_index()
    innings_final.rename(columns={'cumulative_runs': 'final_score'}, inplace=True)

    # Filter for main innings only (1 and 2, exclude super overs 3 and 4)
    innings_main = innings_final[innings_final['innings'].isin([1, 2])]

    print(f"  Total innings: {len(innings_final)}")
    print(f"  Main innings (1 & 2): {len(innings_main)}")

    # Pivot to get both innings scores
    match_scores = innings_main.pivot_table(
        index='match_id',
        columns='innings',
        values='final_score',
        aggfunc='first'
    ).reset_index()

    # Check if we have both innings 1 and 2
    if 1 in match_scores.columns and 2 in match_scores.columns:
        match_scores.rename(columns={1: 'innings_1_score', 2: 'innings_2_score'}, inplace=True)

        # Get team names
        team_map = innings_main.pivot_table(
            index='match_id',
            columns='innings',
            values='batting_team',
            aggfunc='first'
        )
        if 1 in team_map.columns and 2 in team_map.columns:
            match_scores['team_1'] = team_map[1].values
            match_scores['team_2'] = team_map[2].values
        else:
            print("  Warning: Could not extract team names")
    else:
        raise ValueError("Could not find innings 1 and 2 in data")

    # Drop matches with missing scores
    match_scores = match_scores.dropna(subset=['innings_1_score', 'innings_2_score'])

    print(f"✓ Extracted {len(match_scores)} complete matches")

    # Calculate run differential and winner
    match_scores['run_diff_team1'] = match_scores['innings_1_score'] - match_scores['innings_2_score']
    match_scores['run_diff_team2'] = match_scores['innings_2_score'] - match_scores['innings_1_score']
    match_scores['team1_won'] = (match_scores['run_diff_team1'] > 0).astype(int)
    match_scores['team2_won'] = (match_scores['run_diff_team2'] > 0).astype(int)

    # Create team-level observations (2 per match)
    team_obs = []

    for _, row in match_scores.iterrows():
        # Team 1 observation
        team_obs.append({
            'match_id': row['match_id'],
            'team': row['team_1'],
            'run_diff': row['run_diff_team1'],
            'won': row['team1_won']
        })

        # Team 2 observation
        team_obs.append({
            'match_id': row['match_id'],
            'team': row['team_2'],
            'run_diff': row['run_diff_team2'],
            'won': row['team2_won']
        })

    team_df = pd.DataFrame(team_obs)

    print(f"✓ Created {len(team_df)} team-level observations")

    # Add season column to team_df
    # We need to map match_id to season
    match_season = df[['match_id', 'season']].drop_duplicates()
    team_df = team_df.merge(match_season, on='match_id', how='left')
    
    rpw_dict = {}
    
    print("\nCalculating RPW per season:")
    for season in sorted(team_df['season'].unique()):
        season_data = team_df[team_df['season'] == season]
        
        X = season_data['run_diff'].values.reshape(-1, 1)
        y = season_data['won'].values
        X = sm.add_constant(X)
        
        model = sm.OLS(y, X).fit()
        beta = model.params[1]
        rpw = 1 / beta
        
        rpw_dict[str(season)] = rpw
        print(f"  {season}: {rpw:.2f} runs/win (R2={model.rsquared:.3f}, n={len(season_data)})")
        
    return rpw_dict, None


def calculate_war(batter_vorp: pd.DataFrame, bowler_vorp: pd.DataFrame,
                 rpw: float) -> tuple:
    """
    Calculate WAR (Wins Above Replacement).

    WAR = VORP / RPW

    Args:
        batter_vorp: DataFrame with batter VORP
        bowler_vorp: DataFrame with bowler VORP
        rpw: Runs per win estimate

    Returns:
        Tuple of (batter_war, bowler_war) DataFrames
    """
    print("\n" + "="*70)
    print("CALCULATING WAR (WINS ABOVE REPLACEMENT)")
    print("="*70)

    # Calculate WAR for batters
    print(f"\nCalculating batter WAR...")
    
    # Map season to RPW
    batter_vorp['rpw'] = batter_vorp['season'].astype(str).map(rpw)
    # Fill missing
    if batter_vorp['rpw'].isnull().any():
        mean_rpw = np.mean(list(rpw.values()))
        batter_vorp['rpw'].fillna(mean_rpw, inplace=True)
        
    batter_vorp['WAR'] = batter_vorp['VORP'] / batter_vorp['rpw']
    batter_vorp['WAR_per_ball'] = batter_vorp['WAR'] / batter_vorp['balls_faced']

    print(f"✓ Calculated WAR for {len(batter_vorp)} batters")
    print(f"  Mean WAR: {batter_vorp['WAR'].mean():.4f}")

    # Calculate WAR for bowlers
    print(f"\nCalculating bowler WAR...")
    
    # Map season to RPW
    bowler_vorp['rpw'] = bowler_vorp['season'].astype(str).map(rpw)
    # Fill missing
    if bowler_vorp['rpw'].isnull().any():
        mean_rpw = np.mean(list(rpw.values()))
        bowler_vorp['rpw'].fillna(mean_rpw, inplace=True)

    bowler_vorp['WAR'] = bowler_vorp['VORP'] / bowler_vorp['rpw']
    bowler_vorp['WAR_per_ball'] = bowler_vorp['WAR'] / bowler_vorp['balls_bowled']

    print(f"✓ Calculated WAR for {len(bowler_vorp)} bowlers")
    print(f"  Mean WAR: {bowler_vorp['WAR'].mean():.4f}")

    return batter_vorp, bowler_vorp


def display_top_players(batter_war: pd.DataFrame, bowler_war: pd.DataFrame):
    """
    Display top players by WAR.

    Args:
        batter_war: DataFrame with batter WAR
        bowler_war: DataFrame with bowler WAR
    """
    print("\n" + "="*70)
    print("TOP PLAYERS BY WAR")
    print("="*70)

    # Top batters
    print("\nTOP 15 BATTERS BY WAR:")
    print("-"*70)
    top_batters = batter_war.nlargest(15, 'WAR')
    print(top_batters[['batter_name', 'WAR', 'VORP', 'RAA', 'balls_faced', 'WAR_per_ball']].to_string(index=False))

    # Top bowlers
    print("\n\nTOP 15 BOWLERS BY WAR:")
    print("-"*70)
    top_bowlers = bowler_war.nlargest(15, 'WAR')
    print(top_bowlers[['bowler_name', 'WAR', 'VORP', 'RAA', 'balls_bowled', 'WAR_per_ball']].to_string(index=False))

    # Combined leaderboard
    print("\n\nOVERALL TOP 20 PLAYERS BY WAR (COMBINED):")
    print("-"*70)

    # Combine batters and bowlers
    batters_combined = top_batters[['batter_name', 'WAR', 'VORP', 'balls_faced']].copy()
    batters_combined.columns = ['player_name', 'WAR', 'VORP', 'balls']
    batters_combined['role'] = 'Batter'

    bowlers_combined = top_bowlers[['bowler_name', 'WAR', 'VORP', 'balls_bowled']].copy()
    bowlers_combined.columns = ['player_name', 'WAR', 'VORP', 'balls']
    bowlers_combined['role'] = 'Bowler'

    combined = pd.concat([batters_combined, bowlers_combined])
    combined = combined.nlargest(20, 'WAR')

    print(combined.to_string(index=False))


def save_vorp_war_results(batter_war: pd.DataFrame, bowler_war: pd.DataFrame,
                          rpw: float, rpw_model, output_dir: Path):
    """
    Save VORP and WAR results.

    Args:
        batter_war: DataFrame with batter VORP and WAR
        bowler_war: DataFrame with bowler VORP and WAR
        rpw: Runs per win estimate
        rpw_model: Fitted RPW regression model
        output_dir: Output directory
    """
    print("\n" + "="*70)
    print("SAVING VORP AND WAR RESULTS")
    print("="*70)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save batter results
    batter_file = output_dir / 'batter_war.csv'
    batter_war_sorted = batter_war.sort_values('WAR', ascending=False)
    batter_war_sorted.to_csv(batter_file, index=False)
    print(f"✓ Saved batter WAR to {batter_file}")

    # Save bowler results
    bowler_file = output_dir / 'bowler_war.csv'
    bowler_war_sorted = bowler_war.sort_values('WAR', ascending=False)
    bowler_war_sorted.to_csv(bowler_file, index=False)
    print(f"✓ Saved bowler WAR to {bowler_file}")

    # Save RPW estimate
    rpw_file = output_dir / 'runs_per_win.json'
    with open(rpw_file, 'w') as f:
        json.dump(rpw, f, indent=2)
    print(f"✓ Saved RPW estimate to {rpw_file}")

    print(f"\n✓ All results saved to {output_dir}")


def main():
    """Calculate VORP and WAR for all players."""

    # Paths
    project_root = Path(__file__).parent.parent
    raa_dir = project_root / 'results' / '08_replacement_level'
    data_file = project_root / 'results' / '06_context_adjustments' / 'ipl_with_raa.parquet'
    output_dir = project_root / 'results' / '09_vorp_war'

    print("="*70)
    print("VORP AND WAR CALCULATION")
    print("="*70)
    print("\nVORP = RAA - (avg.RAA_rep × balls)")
    print("WAR = VORP / RPW")
    print("\nWAR represents the number of wins a player contributes")
    print("above what a replacement-level player would provide.\n")

    # Load RAA data
    batter_raa, bowler_raa = load_raa_data(raa_dir)

    # Load avg.RAA_rep
    avg_raa_rep = load_avg_raa_rep(raa_dir)

    # Calculate VORP
    batter_vorp, bowler_vorp = calculate_vorp(batter_raa, bowler_raa, avg_raa_rep)

    # Estimate Runs Per Win
    rpw, rpw_model = estimate_runs_per_win(data_file)

    # Calculate WAR
    batter_war, bowler_war = calculate_war(batter_vorp, bowler_vorp, rpw)

    # Display top players
    display_top_players(batter_war, bowler_war)

    # Save results
    save_vorp_war_results(batter_war, bowler_war, rpw, rpw_model, output_dir)

    print("\n" + "="*70)
    print("✓ VORP AND WAR CALCULATION COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Implement uncertainty estimation via resampling (script 10)")
    print("2. Validate results against paper (Tables 1, 2, 3)")
    print("3. Create analysis notebooks and visualizations")


if __name__ == '__main__':
    main()
