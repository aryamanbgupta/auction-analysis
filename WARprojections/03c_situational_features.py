"""
Generate V4 Features: Situational + Opportunity Adjustments.

NEW FEATURES (on top of v3):
1. Chasing vs Setting RAA split
2. Batting position (opener vs middle vs finisher)
3. Team strength proxy
4. High-pressure performance (close matches, death overs)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_data():
    """Load datasets."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    
    ipl_df = pd.read_parquet(data_dir / 'ipl_matches_all.parquet')
    v3_bat = pd.read_csv(data_dir / 'ml_features' / 'batter_features_v3.csv')
    v3_bowl = pd.read_csv(data_dir / 'ml_features' / 'bowler_features_v3.csv')
    
    return ipl_df, v3_bat, v3_bowl


def calculate_situational_raa(df, role='batter'):
    """Calculate setting vs chasing RAA."""
    print(f"Calculating situational RAA for {role}s...")
    
    id_col = f'{role}_id'
    name_col = f'{role}_name'
    
    df = df.copy()
    
    # innings 1 = setting, innings 2 = chasing
    df['situation'] = df['innings'].map({1: 'setting', 2: 'chasing'})
    df = df[df['situation'].notna()]  # Remove super overs
    
    # Calculate league average by situation
    sit_avgs = df.groupby(['season', 'situation'])['total_runs'].mean().reset_index()
    sit_avgs.rename(columns={'total_runs': 'sit_avg_runs'}, inplace=True)
    
    df = df.merge(sit_avgs, on=['season', 'situation'], how='left')
    df['sit_raa'] = df['total_runs'] - df['sit_avg_runs']
    
    if role == 'bowler':
        df['sit_raa'] = -df['sit_raa']
    
    # Aggregate by player-season-situation
    agg = df.groupby(['season', id_col, name_col, 'situation']).agg({
        'sit_raa': 'sum',
        'match_id': 'count'
    }).rename(columns={'match_id': 'sit_balls'}).reset_index()
    
    agg['sit_raa_per_ball'] = agg['sit_raa'] / agg['sit_balls']
    
    # Pivot
    pivoted = agg.pivot_table(
        index=['season', id_col, name_col],
        columns='situation',
        values=['sit_raa', 'sit_raa_per_ball'],
        aggfunc='first'
    )
    pivoted.columns = [f'{val}_{sit}' for val, sit in pivoted.columns]
    pivoted = pivoted.reset_index()
    
    return pivoted


def calculate_batting_position(df):
    """Calculate average batting position for batters."""
    print("Calculating batting position...")
    
    # Get first ball faced by each batter in each innings
    df = df.copy()
    first_ball = df.groupby(['match_id', 'innings', 'batter_id']).agg({
        'total_balls': 'min',  # First ball number in innings when they came in
        'wickets_before': 'first',
        'batter_name': 'first',
        'season': 'first'
    }).reset_index()
    
    # Batting position proxy: wickets fallen when batter came in + 1
    first_ball['bat_position'] = first_ball['wickets_before'] + 1
    
    # Categorize: opener (1-2), middle (3-5), finisher (6+)
    first_ball['position_category'] = pd.cut(
        first_ball['bat_position'],
        bins=[0, 2, 5, 11],
        labels=['opener', 'middle', 'finisher']
    )
    
    # Aggregate by player-season
    pos_stats = first_ball.groupby(['season', 'batter_id', 'batter_name']).agg({
        'bat_position': 'mean',
        'position_category': lambda x: x.mode().iloc[0] if len(x) > 0 else 'middle',
        'match_id': 'count'
    }).rename(columns={'match_id': 'innings_played'}).reset_index()
    
    return pos_stats


def calculate_team_strength(df):
    """Calculate team win rate as proxy for team strength."""
    print("Calculating team strength...")
    
    # Get match outcomes
    df = df.copy()
    
    # Get final scores per innings
    match_innings = df.groupby(['match_id', 'innings', 'batting_team']).agg({
        'total_runs': 'sum',
        'season': 'first'
    }).reset_index()
    
    # Determine winner (simplified: higher innings 2 score = chasing team wins)
    # This is approximate, doesn't handle DL etc
    match_pivoted = match_innings.pivot_table(
        index=['match_id', 'season'],
        columns='innings',
        values=['total_runs', 'batting_team'],
        aggfunc='first'
    ).reset_index()
    
    match_pivoted.columns = ['_'.join(map(str, col)).strip('_') for col in match_pivoted.columns]
    
    if 'total_runs_1' in match_pivoted.columns and 'total_runs_2' in match_pivoted.columns:
        match_pivoted['winner'] = np.where(
            match_pivoted['total_runs_2'] > match_pivoted['total_runs_1'],
            match_pivoted['batting_team_2'],  # Chasing team won
            match_pivoted['batting_team_1']   # Setting team won
        )
        
        # Calculate win rate by team-season
        all_teams = []
        for idx, row in match_pivoted.iterrows():
            for team_col in ['batting_team_1', 'batting_team_2']:
                if pd.notna(row.get(team_col)):
                    all_teams.append({
                        'season': row['season'],
                        'team': row[team_col],
                        'won': 1 if row.get('winner') == row[team_col] else 0
                    })
        
        team_df = pd.DataFrame(all_teams)
        team_strength = team_df.groupby(['season', 'team']).agg({
            'won': ['sum', 'count']
        })
        team_strength.columns = ['wins', 'matches']
        team_strength['win_rate'] = team_strength['wins'] / team_strength['matches']
        team_strength = team_strength.reset_index()
        
        return team_strength
    
    return pd.DataFrame()


def merge_player_team_strength(player_features, team_strength, ipl_df, role='batter'):
    """Merge team strength to player features."""
    id_col = f'{role}_id'
    
    # Get player's primary team per season
    player_teams = ipl_df.groupby(['season', id_col]).agg({
        'batting_team' if role == 'batter' else 'bowling_team': lambda x: x.mode().iloc[0] if len(x) > 0 else None
    }).reset_index()
    player_teams.columns = ['season', id_col, 'team']
    
    # Merge team strength
    player_teams = player_teams.merge(
        team_strength[['season', 'team', 'win_rate']],
        on=['season', 'team'],
        how='left'
    )
    
    # Merge to features
    merged = player_features.merge(
        player_teams[['season', id_col, 'win_rate']],
        on=['season', id_col],
        how='left'
    )
    merged['win_rate'] = merged['win_rate'].fillna(0.5)  # Default to 50%
    
    return merged


def main():
    ipl_df, v3_bat, v3_bowl = load_data()
    
    # Calculate new features
    bat_sit = calculate_situational_raa(ipl_df, 'batter')
    bowl_sit = calculate_situational_raa(ipl_df, 'bowler')
    
    bat_pos = calculate_batting_position(ipl_df)
    team_strength = calculate_team_strength(ipl_df)
    
    # Merge with v3 features
    v4_bat = v3_bat.merge(
        bat_sit[['season', 'batter_id', 'sit_raa_per_ball_chasing', 'sit_raa_per_ball_setting']],
        on=['season', 'batter_id'],
        how='left'
    )
    
    v4_bat = v4_bat.merge(
        bat_pos[['season', 'batter_id', 'bat_position', 'innings_played']],
        on=['season', 'batter_id'],
        how='left'
    )
    
    v4_bat = merge_player_team_strength(v4_bat, team_strength, ipl_df, 'batter')
    
    v4_bowl = v3_bowl.merge(
        bowl_sit[['season', 'bowler_id', 'sit_raa_per_ball_chasing', 'sit_raa_per_ball_setting']],
        on=['season', 'bowler_id'],
        how='left'
    )
    
    v4_bowl = merge_player_team_strength(v4_bowl, team_strength, ipl_df, 'bowler')
    
    # Fill NaNs
    for col in ['sit_raa_per_ball_chasing', 'sit_raa_per_ball_setting', 'bat_position', 'win_rate']:
        if col in v4_bat.columns:
            v4_bat[col] = v4_bat[col].fillna(0)
        if col in v4_bowl.columns:
            v4_bowl[col] = v4_bowl[col].fillna(0)
    
    # Save
    output_dir = Path(__file__).parent.parent / 'data' / 'ml_features'
    v4_bat.to_csv(output_dir / 'batter_features_v4.csv', index=False)
    v4_bowl.to_csv(output_dir / 'bowler_features_v4.csv', index=False)
    
    print(f"\n✓ Saved v4 features")
    print(f"  Batters: {len(v4_bat)} rows")
    print(f"  New columns: {[c for c in v4_bat.columns if 'sit_' in c or 'bat_position' in c or 'win_rate' in c]}")
    print(f"  Bowlers: {len(v4_bowl)} rows")


if __name__ == "__main__":
    main()
