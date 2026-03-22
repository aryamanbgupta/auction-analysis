"""
V9 Enhanced Feature Engineering.

NEW FEATURES:
1. Opponent Quality - RAA weighted by opponent team strength
2. Rolling Form Windows - last_10_matches, last_3_months, exponential decay
3. Better form recency weighting

OUTPUT: data/ml_features/batter_features_v9.csv, bowler_features_v9.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


def load_data():
    """Load IPL data and existing features."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    
    # Load raw IPL ball-by-ball
    ipl = pd.read_parquet(data_dir / 'ipl_matches_all.parquet')
    ipl['match_date'] = pd.to_datetime(ipl['match_date'])
    
    # Load existing v4 features as base
    bat_features = pd.read_csv(data_dir / 'ml_features' / 'batter_features_v4.csv')
    bowl_features = pd.read_csv(data_dir / 'ml_features' / 'bowler_features_v4.csv')
    
    return ipl, bat_features, bowl_features


def calculate_team_strength(ipl):
    """Calculate team strength (win rate) by season."""
    print("Calculating team strength by season...")
    
    # Get match results
    matches = ipl.groupby(['match_id', 'season', 'batting_team', 'bowling_team']).size().reset_index()
    matches = matches.drop(columns=[0])
    matches = matches.drop_duplicates()
    
    # We need match winners - approximate by looking at 2nd innings results
    # Team batting 2nd that reaches target wins
    second_innings = ipl[ipl['innings'] == 2].groupby(['match_id', 'batting_team']).agg({
        'total_runs': 'sum',
        'is_wicket': 'sum',
        'total_balls': 'max'
    }).reset_index()
    
    # For simplicity, use overall season win rates from existing data
    # Load from existing if available, otherwise calculate from match structure
    team_season_strength = {}
    
    for season in ipl['season'].unique():
        season_data = ipl[ipl['season'] == season]
        teams = season_data['batting_team'].unique()
        
        for team in teams:
            # Calculate team's average runs scored vs average runs conceded
            batting = season_data[season_data['batting_team'] == team]
            bowling = season_data[season_data['bowling_team'] == team]
            
            if len(batting) > 0 and len(bowling) > 0:
                runs_scored = batting['total_runs'].sum()
                balls_faced = len(batting)
                runs_conceded = bowling['total_runs'].sum()
                balls_bowled = len(bowling)
                
                # Net run rate proxy
                if balls_faced > 0 and balls_bowled > 0:
                    scoring_rate = runs_scored / balls_faced * 6
                    conceding_rate = runs_conceded / balls_bowled * 6
                    strength = (scoring_rate - conceding_rate + 2) / 4  # Normalize to 0-1 range
                    strength = max(0.2, min(0.8, strength))  # Clip to reasonable range
                else:
                    strength = 0.5
            else:
                strength = 0.5
            
            team_season_strength[(team, season)] = strength
    
    return team_season_strength


def calculate_opponent_adjusted_raa(ipl, team_strength, role='batter'):
    """Calculate RAA adjusted by opponent team strength."""
    print(f"Calculating opponent-adjusted RAA for {role}s...")
    
    id_col = f'{role}_id'
    name_col = f'{role}_name'
    
    if role == 'batter':
        opponent_col = 'bowling_team'
    else:
        opponent_col = 'batting_team'
    
    ipl = ipl.copy()
    
    # Get opponent strength for each ball
    ipl['opponent_strength'] = ipl.apply(
        lambda x: team_strength.get((x[opponent_col], x['season']), 0.5), axis=1
    )
    
    # Calculate expected runs (league average)
    ipl['league_avg'] = ipl.groupby('season')['total_runs'].transform('mean')
    
    # Raw RAA
    ipl['raw_raa'] = ipl['total_runs'] - ipl['league_avg']
    if role == 'bowler':
        ipl['raw_raa'] = -ipl['raw_raa']  # Fewer runs is better
    
    # Opponent-adjusted RAA: weight by opponent strength
    # Against strong opponents, good performance is worth more
    # opponent_strength is 0.2-0.8, normalize to multiplier 0.8-1.2
    ipl['opponent_multiplier'] = 0.8 + (ipl['opponent_strength'] * 0.5)
    ipl['opponent_adj_raa'] = ipl['raw_raa'] * ipl['opponent_multiplier']
    
    # Aggregate by player-season
    opp_adj = ipl.groupby([id_col, name_col, 'season']).agg({
        'opponent_adj_raa': 'sum',
        'raw_raa': 'sum',
        'match_id': 'count'
    }).reset_index()
    opp_adj.columns = [id_col, name_col, 'season', 'opponent_adj_raa', 'raw_raa', 'balls']
    opp_adj['opponent_adj_raa_per_ball'] = opp_adj['opponent_adj_raa'] / opp_adj['balls']
    
    return opp_adj


def calculate_rolling_form(ipl, role='batter'):
    """Calculate rolling form features: last 5, 10, 15 matches with decay weighting."""
    print(f"Calculating rolling form features for {role}s...")
    
    id_col = f'{role}_id'
    name_col = f'{role}_name'
    
    ipl = ipl.copy()
    
    # Calculate per-ball RAA
    ipl['league_avg'] = ipl.groupby('season')['total_runs'].transform('mean')
    ipl['raw_raa'] = ipl['total_runs'] - ipl['league_avg']
    if role == 'bowler':
        ipl['raw_raa'] = -ipl['raw_raa']
    
    # Aggregate to match-level
    match_level = ipl.groupby([id_col, name_col, 'match_id', 'match_date', 'season']).agg({
        'raw_raa': 'sum',
        'total_runs': 'count'  # balls
    }).reset_index()
    match_level.columns = [id_col, name_col, 'match_id', 'match_date', 'season', 'match_raa', 'match_balls']
    match_level['match_raa_per_ball'] = match_level['match_raa'] / match_level['match_balls']
    
    # Sort by player and date
    match_level = match_level.sort_values([id_col, 'match_date'])
    
    # Calculate rolling features for each player-season
    results = []
    
    for (pid, pname, season), group in match_level.groupby([id_col, name_col, 'season']):
        # Get all matches for this player BEFORE this season starts
        season_start = group['match_date'].min()
        player_history = match_level[
            (match_level[id_col] == pid) & 
            (match_level['match_date'] < season_start)
        ].sort_values('match_date', ascending=False)
        
        if len(player_history) == 0:
            # No prior history
            results.append({
                id_col: pid,
                name_col: pname,
                'season': season,
                'last_5_form': np.nan,
                'last_10_form': np.nan,
                'last_15_form': np.nan,
                'decay_weighted_form': np.nan,
                'form_volatility': np.nan,
                'form_trend': np.nan,
            })
            continue
        
        # Last N matches form
        last_5 = player_history.head(5)['match_raa_per_ball'].mean() if len(player_history) >= 5 else np.nan
        last_10 = player_history.head(10)['match_raa_per_ball'].mean() if len(player_history) >= 10 else np.nan
        last_15 = player_history.head(15)['match_raa_per_ball'].mean() if len(player_history) >= 10 else np.nan
        
        # Exponential decay weighted form (most recent = highest weight)
        decay_weights = np.exp(-np.arange(len(player_history)) * 0.1)
        decay_weights = decay_weights / decay_weights.sum()
        decay_form = (player_history['match_raa_per_ball'].values * decay_weights[:len(player_history)]).sum()
        
        # Form volatility (std of recent matches)
        volatility = player_history.head(10)['match_raa_per_ball'].std() if len(player_history) >= 5 else np.nan
        
        # Form trend (recent vs older)
        if len(player_history) >= 10:
            recent_5 = player_history.head(5)['match_raa_per_ball'].mean()
            older_5 = player_history.iloc[5:10]['match_raa_per_ball'].mean()
            trend = recent_5 - older_5  # Positive = improving
        else:
            trend = np.nan
        
        results.append({
            id_col: pid,
            name_col: pname,
            'season': season,
            'last_5_form': last_5,
            'last_10_form': last_10,
            'last_15_form': last_15,
            'decay_weighted_form': decay_form,
            'form_volatility': volatility,
            'form_trend': trend,
        })
    
    return pd.DataFrame(results)


def merge_new_features(base_features, opp_adj, form_features, role='batter'):
    """Merge new features into base feature set."""
    id_col = f'{role}_id'
    name_col = f'{role}_name'
    
    # Merge opponent-adjusted RAA
    enhanced = base_features.merge(
        opp_adj[[id_col, 'season', 'opponent_adj_raa', 'opponent_adj_raa_per_ball']],
        on=[id_col, 'season'],
        how='left'
    )
    
    # Merge form features
    enhanced = enhanced.merge(
        form_features[[id_col, 'season', 'last_5_form', 'last_10_form', 'last_15_form', 
                       'decay_weighted_form', 'form_volatility', 'form_trend']],
        on=[id_col, 'season'],
        how='left'
    )
    
    return enhanced


def main():
    print("=" * 60)
    print("V9 ENHANCED FEATURE ENGINEERING")
    print("=" * 60)
    
    ipl, bat_features, bowl_features = load_data()
    
    # Calculate team strength
    team_strength = calculate_team_strength(ipl)
    
    output_dir = Path(__file__).parent.parent / 'data' / 'ml_features'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for role, base_features in [('batter', bat_features), ('bowler', bowl_features)]:
        print(f"\n{'='*60}")
        print(f"Processing {role.upper()}S")
        print('='*60)
        
        # Calculate new features
        opp_adj = calculate_opponent_adjusted_raa(ipl, team_strength, role)
        form = calculate_rolling_form(ipl, role)
        
        # Merge into base features
        enhanced = merge_new_features(base_features, opp_adj, form, role)
        
        print(f"\nOriginal features: {len(base_features.columns)}")
        print(f"Enhanced features: {len(enhanced.columns)}")
        print(f"New columns: {set(enhanced.columns) - set(base_features.columns)}")
        
        # Show sample of new features
        new_cols = ['opponent_adj_raa_per_ball', 'last_5_form', 'last_10_form', 
                    'decay_weighted_form', 'form_volatility', 'form_trend']
        print(f"\nNew feature stats:")
        for col in new_cols:
            if col in enhanced.columns:
                print(f"  {col}: mean={enhanced[col].mean():.4f}, non-null={enhanced[col].notna().sum()}")
        
        # Save
        enhanced.to_csv(output_dir / f'{role}_features_v9.csv', index=False)
        print(f"\n✓ Saved to: {output_dir / f'{role}_features_v9.csv'}")
    
    print("\n" + "=" * 60)
    print("V9 Feature Engineering Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
