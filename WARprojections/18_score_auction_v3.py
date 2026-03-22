"""
V3 Combined Auction Scoring - Uses ALL new models.

COMBINES:
1. V7 Unified (best for players with IPL history + global data)
2. V8 Domestic (for players with no IPL history but SMAT/domestic data)
3. Falls back to Marcel, Global-Only, or Replacement Level

OUTPUT: results/WARprojections/auction_2026_v3/
"""

import pandas as pd
import numpy as np
from pathlib import Path
from difflib import SequenceMatcher


def load_auction_list():
    """Load enriched auction list."""
    data_dir = Path(__file__).parent.parent / 'data'
    auction = pd.read_csv(data_dir / 'ipl_2026_auction_enriched.csv')
    print(f"Auction pool: {len(auction)} players")
    return auction


def load_all_predictions():
    """Load predictions from all model versions."""
    results_dir = Path(__file__).parent.parent / 'results' / 'WARprojections'
    data_dir = Path(__file__).parent.parent / 'data'
    
    preds = {}
    
    # V7 Unified (trained on 2025, forecasting 2026)
    try:
        v7_bat = pd.read_csv(results_dir / 'v7_unified' / 'batter_projections_2026_v7.csv')
        v7_bowl = pd.read_csv(results_dir / 'v7_unified' / 'bowler_projections_2026_v7.csv')
        
        # Add IDs from feature files
        bat_features = pd.read_csv(data_dir / 'ml_features' / 'batter_features_v4.csv')
        bowl_features = pd.read_csv(data_dir / 'ml_features' / 'bowler_features_v4.csv')
        
        bat_ids = bat_features.sort_values('season').groupby('batter_name')['batter_id'].last().reset_index()
        bowl_ids = bowl_features.sort_values('season').groupby('bowler_name')['bowler_id'].last().reset_index()
        
        v7_bat = v7_bat.merge(bat_ids, on='batter_name', how='left')
        v7_bowl = v7_bowl.merge(bowl_ids, on='bowler_name', how='left')
        
        preds['v7_bat'] = v7_bat
        preds['v7_bowl'] = v7_bowl
        print(f"V7 Unified: {len(v7_bat)} batters, {len(v7_bowl)} bowlers")
    except Exception as e:
        print(f"V7 error: {e}")
    
    # V8 Domestic (players with no IPL history)
    try:
        v8_bat = pd.read_csv(results_dir / 'v8_domestic' / 'batter_domestic_predictions.csv')
        v8_bowl = pd.read_csv(results_dir / 'v8_domestic' / 'bowler_domestic_predictions.csv')
        preds['v8_bat'] = v8_bat
        preds['v8_bowl'] = v8_bowl
        print(f"V8 Domestic: {len(v8_bat)} batters, {len(v8_bowl)} bowlers (no IPL history)")
    except Exception as e:
        print(f"V8 error: {e}")
    
    # V6 Production (legacy best)
    try:
        v6_bat = pd.read_csv(results_dir / 'v6_production' / 'batter_projections_2026_prod.csv')
        v6_bowl = pd.read_csv(results_dir / 'v6_production' / 'bowler_projections_2026_prod.csv')
        
        bat_ids = bat_features.sort_values('season').groupby('batter_name')['batter_id'].last().reset_index()
        bowl_ids = bowl_features.sort_values('season').groupby('bowler_name')['bowler_id'].last().reset_index()
        
        v6_bat = v6_bat.merge(bat_ids, on='batter_name', how='left')
        v6_bowl = v6_bowl.merge(bowl_ids, on='bowler_name', how='left')
        
        preds['v6_bat'] = v6_bat
        preds['v6_bowl'] = v6_bowl
        print(f"V6 Production: {len(v6_bat)} batters, {len(v6_bowl)} bowlers")
    except Exception as e:
        print(f"V6 error: {e}")
    
    # Marcel
    try:
        marcel_bat = pd.read_csv(results_dir / 'marcel' / 'batter_projections_2026.csv')
        marcel_bowl = pd.read_csv(results_dir / 'marcel' / 'bowler_projections_2026.csv')
        preds['marcel_bat'] = marcel_bat
        preds['marcel_bowl'] = marcel_bowl
        print(f"Marcel: {len(marcel_bat)} batters, {len(marcel_bowl)} bowlers")
    except Exception as e:
        print(f"Marcel error: {e}")
    
    # Global-only
    try:
        global_bat = pd.read_csv(results_dir / 'global_only' / 'batter_global_only_predictions.csv')
        global_bowl = pd.read_csv(results_dir / 'global_only' / 'bowler_global_only_predictions.csv')
        preds['global_bat'] = global_bat
        preds['global_bowl'] = global_bowl
        print(f"Global-only: {len(global_bat)} batters, {len(global_bowl)} bowlers")
    except Exception as e:
        print(f"Global-only error: {e}")
    
    return preds


def create_lookups(preds):
    """Create ID and name-based lookup dictionaries."""
    bat_id = {}
    bat_name = {}
    bowl_id = {}
    bowl_name = {}
    
    # Priority order: V7 > V6 > V8 > Marcel > Global
    
    # V7 Unified (highest priority for IPL players)
    if 'v7_bat' in preds:
        for _, row in preds['v7_bat'].iterrows():
            pid = row.get('batter_id')
            if pd.notna(pid) and pid not in bat_id:
                bat_id[pid] = (row['projected_war_2026'], 'V7_Unified')
            name = row['batter_name'].lower().strip()
            if name not in bat_name:
                bat_name[name] = (row['projected_war_2026'], 'V7_Unified')
    
    if 'v7_bowl' in preds:
        for _, row in preds['v7_bowl'].iterrows():
            pid = row.get('bowler_id')
            if pd.notna(pid) and pid not in bowl_id:
                bowl_id[pid] = (row['projected_war_2026'], 'V7_Unified')
            name = row['bowler_name'].lower().strip()
            if name not in bowl_name:
                bowl_name[name] = (row['projected_war_2026'], 'V7_Unified')
    
    # V6 Production (fallback for IPL players)
    if 'v6_bat' in preds:
        for _, row in preds['v6_bat'].iterrows():
            pid = row.get('batter_id')
            if pd.notna(pid) and pid not in bat_id:
                bat_id[pid] = (row['projected_war_2026'], 'V6_Production')
            name = row['batter_name'].lower().strip()
            if name not in bat_name:
                bat_name[name] = (row['projected_war_2026'], 'V6_Production')
    
    if 'v6_bowl' in preds:
        for _, row in preds['v6_bowl'].iterrows():
            pid = row.get('bowler_id')
            if pd.notna(pid) and pid not in bowl_id:
                bowl_id[pid] = (row['projected_war_2026'], 'V6_Production')
            name = row['bowler_name'].lower().strip()
            if name not in bowl_name:
                bowl_name[name] = (row['projected_war_2026'], 'V6_Production')
    
    # V8 Domestic (for players with no IPL history)
    if 'v8_bat' in preds:
        for _, row in preds['v8_bat'].iterrows():
            pid = row.get('batter_id')
            if pd.notna(pid) and pid not in bat_id:
                bat_id[pid] = (row['predicted_ipl_war'], 'V8_Domestic')
            name = row['batter_name'].lower().strip()
            if name not in bat_name:
                bat_name[name] = (row['predicted_ipl_war'], 'V8_Domestic')
    
    if 'v8_bowl' in preds:
        for _, row in preds['v8_bowl'].iterrows():
            pid = row.get('bowler_id')
            if pd.notna(pid) and pid not in bowl_id:
                bowl_id[pid] = (row['predicted_ipl_war'], 'V8_Domestic')
            name = row['bowler_name'].lower().strip()
            if name not in bowl_name:
                bowl_name[name] = (row['predicted_ipl_war'], 'V8_Domestic')
    
    # Marcel
    if 'marcel_bat' in preds:
        for _, row in preds['marcel_bat'].iterrows():
            pid = row.get('player_id')
            if pd.notna(pid) and pid not in bat_id:
                bat_id[pid] = (row['projected_war_2026'], 'Marcel')
            name = row['player_name'].lower().strip()
            if name not in bat_name:
                bat_name[name] = (row['projected_war_2026'], 'Marcel')
    
    if 'marcel_bowl' in preds:
        for _, row in preds['marcel_bowl'].iterrows():
            pid = row.get('player_id')
            if pd.notna(pid) and pid not in bowl_id:
                bowl_id[pid] = (row['projected_war_2026'], 'Marcel')
            name = row['player_name'].lower().strip()
            if name not in bowl_name:
                bowl_name[name] = (row['projected_war_2026'], 'Marcel')
    
    # Global-only (lowest priority)
    if 'global_bat' in preds:
        for _, row in preds['global_bat'].iterrows():
            pid = row.get('batter_id')
            if pd.notna(pid) and pid not in bat_id:
                bat_id[pid] = (row['predicted_ipl_war'], 'Global_Only')
            name = row['batter_name'].lower().strip()
            if name not in bat_name:
                bat_name[name] = (row['predicted_ipl_war'], 'Global_Only')
    
    if 'global_bowl' in preds:
        for _, row in preds['global_bowl'].iterrows():
            pid = row.get('bowler_id')
            if pd.notna(pid) and pid not in bowl_id:
                bowl_id[pid] = (row['predicted_ipl_war'], 'Global_Only')
            name = row['bowler_name'].lower().strip()
            if name not in bowl_name:
                bowl_name[name] = (row['predicted_ipl_war'], 'Global_Only')
    
    print(f"\nLookup sizes:")
    print(f"  Batter IDs: {len(bat_id)}, Names: {len(bat_name)}")
    print(f"  Bowler IDs: {len(bowl_id)}, Names: {len(bowl_name)}")
    
    return bat_id, bat_name, bowl_id, bowl_name


def fuzzy_match(name, candidates, threshold=0.8):
    """Find best fuzzy match."""
    best_match = None
    best_score = 0
    
    name_lower = name.lower().strip()
    
    for candidate in candidates:
        score = SequenceMatcher(None, name_lower, candidate).ratio()
        if score > best_score:
            best_score = score
            best_match = candidate
    
    return best_match if best_score >= threshold else None


def score_player(row, bat_id, bat_name, bowl_id, bowl_name):
    """Score an auction player using all available models."""
    cricsheet_id = row.get('cricsheet_id')
    
    war = None
    source = None
    method = None
    
    # Step 1: ID matching
    if pd.notna(cricsheet_id):
        if cricsheet_id in bat_id:
            war, source = bat_id[cricsheet_id]
            method = 'ID'
        elif cricsheet_id in bowl_id:
            war, source = bowl_id[cricsheet_id]
            method = 'ID'
    
    # Step 2: Name matching
    if war is None:
        names = []
        for col in ['unique_name', 'name', 'PLAYER', 'full_name']:
            if pd.notna(row.get(col)):
                names.append(row[col].lower().strip())
        
        for name in names:
            if name in bat_name:
                war, source = bat_name[name]
                method = 'Name'
                break
            if name in bowl_name:
                war, source = bowl_name[name]
                method = 'Name'
                break
    
    # Step 3: Fuzzy matching
    if war is None and names:
        all_names = list(bat_name.keys()) + list(bowl_name.keys())
        match = fuzzy_match(names[0], all_names, threshold=0.75)
        if match:
            if match in bat_name:
                war, source = bat_name[match]
            else:
                war, source = bowl_name[match]
            method = 'Fuzzy'
    
    return war, source, method


def main():
    print("=" * 70)
    print("V3 COMBINED AUCTION SCORING")
    print("Uses: V7 Unified + V8 Domestic + V6 + Marcel + Global")
    print("=" * 70)
    
    auction = load_auction_list()
    preds = load_all_predictions()
    bat_id, bat_name, bowl_id, bowl_name = create_lookups(preds)
    
    # Score all players
    results = []
    for _, row in auction.iterrows():
        war, source, method = score_player(row, bat_id, bat_name, bowl_id, bowl_name)
        
        results.append({
            'sr_no': row.get('SR. NO.'),
            'player': row.get('PLAYER') or row.get('name'),
            'country': row.get('COUNTRY') or row.get('country'),
            'role': row.get('playing_role'),
            'base_price': row.get('BASE PRICE (INR LAKH)'),
            'capped_status': row.get('C/U/A'),
            'projected_war_2026': war,
            'prediction_source': source,
            'match_method': method,
            'cricsheet_id': row.get('cricsheet_id'),
        })
    
    results_df = pd.DataFrame(results)
    
    # Statistics
    print("\n" + "=" * 70)
    print("COVERAGE STATISTICS (V3 - ALL MODELS)")
    print("=" * 70)
    
    total = len(results_df)
    matched = results_df['projected_war_2026'].notna().sum()
    unmatched = total - matched
    
    print(f"Total players: {total}")
    print(f"With predictions: {matched} ({100*matched/total:.1f}%)")
    print(f"No prediction: {unmatched} ({100*unmatched/total:.1f}%)")
    
    print("\nBy source:")
    print(results_df['prediction_source'].value_counts(dropna=False))
    
    print("\nBy match method:")
    print(results_df['match_method'].value_counts(dropna=False))
    
    # Fill NaN
    results_df['projected_war_2026'] = results_df['projected_war_2026'].fillna(0)
    results_df['prediction_source'] = results_df['prediction_source'].fillna('Replacement_Level')
    results_df['match_method'] = results_df['match_method'].fillna('None')
    
    # Sort
    results_df = results_df.sort_values('projected_war_2026', ascending=False)
    
    # Save
    output_dir = Path(__file__).parent.parent / 'results' / 'WARprojections' / 'auction_2026_v3'
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / 'auction_pool_war_projections_v3.csv', index=False)
    
    # Print top players
    print("\n" + "=" * 70)
    print("TOP 30 AUCTION PLAYERS (V3 - ALL MODELS)")
    print("=" * 70)
    print(results_df.head(30)[['player', 'country', 'role', 'projected_war_2026', 'prediction_source']].to_string(index=False))
    
    # Compare with V2
    print("\n" + "=" * 70)
    print("IMPROVEMENT SUMMARY")
    print("=" * 70)
    print(f"V3 Coverage: {matched}/{total} ({100*matched/total:.1f}%)")
    print(f"V8 Domestic adds predictions for players with no IPL history")
    
    print(f"\n✓ Saved to: {output_dir / 'auction_pool_war_projections_v3.csv'}")


if __name__ == "__main__":
    main()
