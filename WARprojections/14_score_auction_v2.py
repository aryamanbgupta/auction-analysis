"""
Score IPL 2026 Auction Pool with WAR Projections - V2.

IMPROVEMENTS:
1. Uses cricsheet_id for matching (instead of fuzzy name matching)
2. Falls back to name matching only when ID unavailable
3. Better coverage statistics

OUTPUT: results/WARprojections/auction_2026_v2/
"""

import pandas as pd
import numpy as np
from pathlib import Path
from difflib import SequenceMatcher


def load_auction_list():
    """Load enriched auction list with cricsheet IDs."""
    data_dir = Path(__file__).parent.parent / 'data'
    auction = pd.read_csv(data_dir / 'ipl_2026_auction_enriched.csv')
    print(f"Auction pool: {len(auction)} players")
    print(f"  With cricsheet_id: {auction['cricsheet_id'].notna().sum()}")
    print(f"  Without cricsheet_id: {auction['cricsheet_id'].isna().sum()}")
    return auction


def load_all_predictions_with_ids():
    """Load all prediction sources with player IDs where available."""
    project_root = Path(__file__).parent.parent
    results_dir = project_root / 'results' / 'WARprojections'
    data_dir = project_root / 'data'
    
    preds = {}
    
    # V6 Production - need to add IDs from feature files
    try:
        v6_bat = pd.read_csv(results_dir / 'v6_production' / 'batter_projections_2026_prod.csv')
        v6_bowl = pd.read_csv(results_dir / 'v6_production' / 'bowler_projections_2026_prod.csv')
        
        # Add IDs from feature files
        bat_features = pd.read_csv(data_dir / 'ml_features' / 'batter_features_v4.csv')
        bowl_features = pd.read_csv(data_dir / 'ml_features' / 'bowler_features_v4.csv')
        
        # Get latest season ID for each player
        bat_ids = bat_features.sort_values('season').groupby('batter_name')['batter_id'].last().reset_index()
        bowl_ids = bowl_features.sort_values('season').groupby('bowler_name')['bowler_id'].last().reset_index()
        
        v6_bat = v6_bat.merge(bat_ids, on='batter_name', how='left')
        v6_bowl = v6_bowl.merge(bowl_ids, on='bowler_name', how='left')
        
        preds['v6_bat'] = v6_bat
        preds['v6_bowl'] = v6_bowl
        print(f"V6 Production: {len(v6_bat)} batters, {len(v6_bowl)} bowlers")
        print(f"  With IDs: {v6_bat['batter_id'].notna().sum()} batters, {v6_bowl['bowler_id'].notna().sum()} bowlers")
    except Exception as e:
        print(f"V6 Production error: {e}")
    
    # Marcel - already has player_id
    try:
        marcel_bat = pd.read_csv(results_dir / 'marcel' / 'batter_projections_2026.csv')
        marcel_bowl = pd.read_csv(results_dir / 'marcel' / 'bowler_projections_2026.csv')
        preds['marcel_bat'] = marcel_bat
        preds['marcel_bowl'] = marcel_bowl
        print(f"Marcel: {len(marcel_bat)} batters, {len(marcel_bowl)} bowlers")
        print(f"  With IDs: {marcel_bat['player_id'].notna().sum()} batters, {marcel_bowl['player_id'].notna().sum()} bowlers")
    except Exception as e:
        print(f"Marcel error: {e}")
    
    # Global-only - need to add IDs
    try:
        global_bat = pd.read_csv(results_dir / 'global_only' / 'batter_global_only_predictions.csv')
        global_bowl = pd.read_csv(results_dir / 'global_only' / 'bowler_global_only_predictions.csv')
        preds['global_bat'] = global_bat
        preds['global_bowl'] = global_bowl
        
        # Global predictions should have IDs from their source
        id_col_bat = 'batter_id' if 'batter_id' in global_bat.columns else None
        id_col_bowl = 'bowler_id' if 'bowler_id' in global_bowl.columns else None
        print(f"Global-only: {len(global_bat)} batters, {len(global_bowl)} bowlers")
        if id_col_bat:
            print(f"  With IDs: {global_bat[id_col_bat].notna().sum()} batters")
    except Exception as e:
        print(f"Global-only error: {e}")
    
    return preds


def create_id_based_lookup(preds):
    """Create cricsheet_id -> (war, source) lookup dictionaries."""
    # Batter lookup by ID
    bat_id_lookup = {}
    bat_name_lookup = {}
    
    # V6 first (highest priority)
    if 'v6_bat' in preds:
        for _, row in preds['v6_bat'].iterrows():
            if pd.notna(row.get('batter_id')):
                bat_id_lookup[row['batter_id']] = (row['projected_war_2026'], 'V6_Production')
            name = row['batter_name'].lower().strip()
            bat_name_lookup[name] = (row['projected_war_2026'], 'V6_Production')
    
    # Marcel fills gaps
    if 'marcel_bat' in preds:
        for _, row in preds['marcel_bat'].iterrows():
            pid = row.get('player_id')
            if pd.notna(pid) and pid not in bat_id_lookup:
                bat_id_lookup[pid] = (row['projected_war_2026'], 'Marcel')
            name = row['player_name'].lower().strip()
            if name not in bat_name_lookup:
                bat_name_lookup[name] = (row['projected_war_2026'], 'Marcel')
    
    # Global-only for remaining
    if 'global_bat' in preds:
        for _, row in preds['global_bat'].iterrows():
            pid = row.get('batter_id')
            if pd.notna(pid) and pid not in bat_id_lookup:
                bat_id_lookup[pid] = (row['predicted_ipl_war'], 'Global_Only')
            name = row['batter_name'].lower().strip()
            if name not in bat_name_lookup:
                bat_name_lookup[name] = (row['predicted_ipl_war'], 'Global_Only')
    
    # Bowler lookup by ID
    bowl_id_lookup = {}
    bowl_name_lookup = {}
    
    if 'v6_bowl' in preds:
        for _, row in preds['v6_bowl'].iterrows():
            if pd.notna(row.get('bowler_id')):
                bowl_id_lookup[row['bowler_id']] = (row['projected_war_2026'], 'V6_Production')
            name = row['bowler_name'].lower().strip()
            bowl_name_lookup[name] = (row['projected_war_2026'], 'V6_Production')
    
    if 'marcel_bowl' in preds:
        for _, row in preds['marcel_bowl'].iterrows():
            pid = row.get('player_id')
            if pd.notna(pid) and pid not in bowl_id_lookup:
                bowl_id_lookup[pid] = (row['projected_war_2026'], 'Marcel')
            name = row['player_name'].lower().strip()
            if name not in bowl_name_lookup:
                bowl_name_lookup[name] = (row['projected_war_2026'], 'Marcel')
    
    if 'global_bowl' in preds:
        for _, row in preds['global_bowl'].iterrows():
            pid = row.get('bowler_id')
            if pd.notna(pid) and pid not in bowl_id_lookup:
                bowl_id_lookup[pid] = (row['predicted_ipl_war'], 'Global_Only')
            name = row['bowler_name'].lower().strip()
            if name not in bowl_name_lookup:
                bowl_name_lookup[name] = (row['predicted_ipl_war'], 'Global_Only')
    
    print(f"\nID Lookups: {len(bat_id_lookup)} batters, {len(bowl_id_lookup)} bowlers")
    print(f"Name Lookups: {len(bat_name_lookup)} batters, {len(bowl_name_lookup)} bowlers")
    
    return bat_id_lookup, bat_name_lookup, bowl_id_lookup, bowl_name_lookup


def fuzzy_match(name, candidates, threshold=0.8):
    """Find best fuzzy match for a name."""
    best_match = None
    best_score = 0
    
    name_lower = name.lower().strip()
    
    for candidate in candidates:
        candidate_lower = candidate.lower().strip()
        score = SequenceMatcher(None, name_lower, candidate_lower).ratio()
        if score > best_score:
            best_score = score
            best_match = candidate
    
    return best_match if best_score >= threshold else None


def match_player_v2(row, bat_id_lookup, bat_name_lookup, bowl_id_lookup, bowl_name_lookup):
    """Find prediction for an auction player using ID-first approach."""
    cricsheet_id = row.get('cricsheet_id')
    
    # Determine role
    role = row.get('playing_role', '')
    is_bowler = 'bowler' in str(role).lower() or 'bowl' in str(row.get('SET', '')).lower()
    is_batter = 'batter' in str(role).lower() or 'bat' in str(row.get('SET', '')).lower()
    is_allrounder = 'allrounder' in str(role).lower()
    
    war = None
    source = None
    match_method = None
    
    # STEP 1: Try ID-based matching first
    if pd.notna(cricsheet_id):
        # Try batter lookup
        if cricsheet_id in bat_id_lookup:
            war, source = bat_id_lookup[cricsheet_id]
            match_method = 'ID'
        # Try bowler lookup
        elif cricsheet_id in bowl_id_lookup:
            war, source = bowl_id_lookup[cricsheet_id]
            match_method = 'ID'
    
    # STEP 2: Fall back to name matching
    if war is None:
        name_options = []
        if pd.notna(row.get('unique_name')):
            name_options.append(row['unique_name'].lower().strip())
        if pd.notna(row.get('name')):
            name_options.append(row['name'].lower().strip())
        if pd.notna(row.get('PLAYER')):
            name_options.append(row['PLAYER'].lower().strip())
        if pd.notna(row.get('full_name')):
            name_options.append(row['full_name'].lower().strip())
        
        # Try exact name matches
        for name in name_options:
            if name in bat_name_lookup:
                war, source = bat_name_lookup[name]
                match_method = 'Name'
                break
            if name in bowl_name_lookup:
                war, source = bowl_name_lookup[name]
                match_method = 'Name'
                break
        
        # STEP 3: Fuzzy matching as last resort
        if war is None and name_options:
            primary_name = name_options[0]
            
            if is_bowler and not is_batter:
                match = fuzzy_match(primary_name, bowl_name_lookup.keys(), threshold=0.75)
                if match:
                    war, source = bowl_name_lookup[match]
                    match_method = 'Fuzzy'
            elif is_batter or is_allrounder:
                match = fuzzy_match(primary_name, bat_name_lookup.keys(), threshold=0.75)
                if match:
                    war, source = bat_name_lookup[match]
                    match_method = 'Fuzzy'
            else:
                # Try both
                all_names = list(bat_name_lookup.keys()) + list(bowl_name_lookup.keys())
                match = fuzzy_match(primary_name, all_names, threshold=0.75)
                if match:
                    if match in bat_name_lookup:
                        war, source = bat_name_lookup[match]
                    else:
                        war, source = bowl_name_lookup[match]
                    match_method = 'Fuzzy'
    
    return war, source, match_method


def main():
    print("=" * 60)
    print("SCORING IPL 2026 AUCTION POOL (V2 - ID-BASED)")
    print("=" * 60)
    
    # Load data
    auction = load_auction_list()
    preds = load_all_predictions_with_ids()
    bat_id_lookup, bat_name_lookup, bowl_id_lookup, bowl_name_lookup = create_id_based_lookup(preds)
    
    # Score each player
    results = []
    
    for idx, row in auction.iterrows():
        war, source, match_method = match_player_v2(
            row, bat_id_lookup, bat_name_lookup, bowl_id_lookup, bowl_name_lookup
        )
        
        results.append({
            'sr_no': row.get('SR. NO.'),
            'player': row.get('PLAYER') or row.get('name'),
            'country': row.get('COUNTRY') or row.get('country'),
            'role': row.get('playing_role'),
            'base_price': row.get('BASE PRICE (INR LAKH)'),
            'capped_status': row.get('C/U/A'),
            'projected_war_2026': war,
            'prediction_source': source,
            'match_method': match_method,
            'cricsheet_id': row.get('cricsheet_id'),
        })
    
    results_df = pd.DataFrame(results)
    
    # Statistics
    print("\n" + "=" * 60)
    print("COVERAGE STATISTICS (V2 - ID-BASED)")
    print("=" * 60)
    
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
    
    # Compare with legacy
    print("\n" + "=" * 60)
    print("COMPARISON WITH LEGACY (V1)")
    print("=" * 60)
    print("V1: Name-only matching")
    print("V2: ID-first, then Name, then Fuzzy")
    
    # Fill NaN with 0 (replacement level)
    results_df['projected_war_2026'] = results_df['projected_war_2026'].fillna(0)
    results_df['prediction_source'] = results_df['prediction_source'].fillna('Replacement_Level')
    results_df['match_method'] = results_df['match_method'].fillna('None')
    
    # Sort by projected WAR
    results_df = results_df.sort_values('projected_war_2026', ascending=False)
    
    # Save
    output_dir = Path(__file__).parent.parent / 'results' / 'WARprojections' / 'auction_2026_v2'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(output_dir / 'auction_pool_war_projections_v2.csv', index=False)
    
    # Show top players
    print("\n" + "=" * 60)
    print("TOP 30 AUCTION PLAYERS BY PROJECTED WAR (V2)")
    print("=" * 60)
    top30 = results_df.head(30)[['player', 'country', 'role', 'base_price', 'projected_war_2026', 'prediction_source', 'match_method']]
    print(top30.to_string(index=False))
    
    print(f"\n✓ Results saved to: {output_dir / 'auction_pool_war_projections_v2.csv'}")


if __name__ == "__main__":
    main()
