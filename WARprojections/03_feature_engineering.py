"""
Generate ML features for full history projections.
Features: Lags, Rolling Averages, Career Stats, Age.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def load_data():
    """Load WAR data and metadata."""
    data_dir = Path(__file__).parent.parent / 'data'
    
    bat_war = pd.read_csv(data_dir / 'batter_war_full_history.csv')
    bowl_war = pd.read_csv(data_dir / 'bowler_war_full_history.csv')
    
    # Load metadata for Age
    meta_path = data_dir / 'player_metadata_full.csv'
    if not meta_path.exists():
        meta_path = data_dir / 'player_metadata.csv'
    
    metadata = pd.read_csv(meta_path)
    
    return bat_war, bowl_war, metadata

def calculate_age(dob_str, season):
    """Calculate age at the start of the season."""
    if pd.isna(dob_str):
        return np.nan
    try:
        dob = datetime.strptime(str(dob_str), '%Y-%m-%d')
        return season - dob.year
    except:
        return np.nan

def generate_features(df, metadata, role='batter'):
    """Generate features for a given role."""
    print(f"\nGenerating features for {role}s...")
    
    # Merge metadata for Age
    # Ensure ID columns match
    id_col = f'{role}_id'
    
    # Check if 'player_id' in metadata matches id_col
    # Usually metadata has 'player_id' or 'id'
    meta_id_col = 'player_id' if 'player_id' in metadata.columns else 'id'
    
    # Merge
    df = df.merge(metadata[[meta_id_col, 'dob']], left_on=id_col, right_on=meta_id_col, how='left')
    
    # Calculate Age
    df['age'] = df.apply(lambda x: calculate_age(x['dob'], x['season']), axis=1)
    
    # Sort for time-series operations
    df = df.sort_values([id_col, 'season'])
    
    # --- Feature Engineering ---
    
    # 1. Lags (1, 2, 3 years)
    # We need to ensure we are shifting by *season*, not just row.
    # But since we have all seasons, we can just shift if we reindex?
    # Better to use a self-join or pivot to be safe about missing seasons.
    
    # Let's use a simpler approach: Loop through lags
    features = df.copy()
    
    cols_to_lag = ['WAR', 'RAA', 'RAA_per_ball', 'consistency']
    if role == 'batter':
        cols_to_lag.append('balls_faced')
    else:
        cols_to_lag.append('balls_bowled')
        
    for lag in [1, 2, 3]:
        shifted = df[[id_col, 'season'] + cols_to_lag].copy()
        shifted['season'] = shifted['season'] + lag # Shift season forward to match target
        
        suffix = f'_lag{lag}'
        shifted = shifted.rename(columns={c: c + suffix for c in cols_to_lag})
        
        features = features.merge(shifted, on=[id_col, 'season'], how='left')
        
    # 2. Weighted Average (Marcel-style)
    # 5*Lag1 + 4*Lag2 + 3*Lag3
    # We need to handle missing lags. Marcel usually re-weights if lags are missing.
    # For ML, we can just provide the lags and let the tree figure it out, 
    # OR provide the weighted avg as a helper feature.
    
    def weighted_avg(row, col_base):
        v1 = row.get(f'{col_base}_lag1', np.nan)
        v2 = row.get(f'{col_base}_lag2', np.nan)
        v3 = row.get(f'{col_base}_lag3', np.nan)
        
        w1, w2, w3 = 5, 4, 3
        num = 0
        den = 0
        
        if not pd.isna(v1): num += v1*w1; den += w1
        if not pd.isna(v2): num += v2*w2; den += w2
        if not pd.isna(v3): num += v3*w3; den += w3
        
        return num / den if den > 0 else np.nan

    features['WAR_weighted'] = features.apply(lambda x: weighted_avg(x, 'WAR'), axis=1)
    features['RAA_weighted'] = features.apply(lambda x: weighted_avg(x, 'RAA'), axis=1)
    
    # 3. Career Stats (Cumulative)
    # Shift by 1 so we don't include current season in "career so far"
    # Actually, we want career stats *entering* the season.
    
    # Cumulative sum
    df_grouped = df.groupby(id_col)
    
    features['career_war'] = df_grouped['WAR'].cumsum().shift(1)
    features['career_raa'] = df_grouped['RAA'].cumsum().shift(1)
    
    if role == 'batter':
        features['career_balls'] = df_grouped['balls_faced'].cumsum().shift(1)
    else:
        features['career_balls'] = df_grouped['balls_bowled'].cumsum().shift(1)
        
    # Fill NaN for first season (0 career stats)
    features['career_war'] = features['career_war'].fillna(0)
    features['career_raa'] = features['career_raa'].fillna(0)
    features['career_balls'] = features['career_balls'].fillna(0)
    
    # Experience (Years played)
    features['years_played'] = df_grouped.cumcount()
    
    # 4. Target Variable
    # We want to predict *next* season's WAR.
    # So we shift WAR back by 1 (or merge with season+1)
    target = df[[id_col, 'season', 'WAR']].copy()
    target['season'] = target['season'] - 1 # Shift season back to match "current" season
    target = target.rename(columns={'WAR': 'target_WAR_next'})
    
    features = features.merge(target, on=[id_col, 'season'], how='left')
    
    # Filter out rows where we don't have a target (e.g. 2025, since we don't have 2026 actuals)
    # BUT for 2025, we want to *predict* 2026. So we keep them for the "Forecast" set.
    # We will split later.
    
    print(f"Generated {len(features)} rows with features.")
    return features

def main():
    bat_war, bowl_war, metadata = load_data()
    
    bat_features = generate_features(bat_war, metadata, 'batter')
    bowl_features = generate_features(bowl_war, metadata, 'bowler')
    
    # Save
    output_dir = Path(__file__).parent.parent / 'data' / 'ml_features'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    bat_features.to_csv(output_dir / 'batter_features_full.csv', index=False)
    bowl_features.to_csv(output_dir / 'bowler_features_full.csv', index=False)
    
    print(f"\nSaved features to {output_dir}")

if __name__ == "__main__":
    main()
