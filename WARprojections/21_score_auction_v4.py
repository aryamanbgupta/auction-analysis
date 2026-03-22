"""
V4 Combined Auction Scoring - Uses V9 model + aggressive player matching.

IMPROVEMENTS:
1. Uses V9 model with enhanced features
2. Lower fuzzy threshold (0.65 instead of 0.75)
3. Multiple name normalization strategies
4. Fills more of the 96 missing players

OUTPUT: results/WARprojections/auction_2026_v4/
"""

import pandas as pd
import numpy as np
from pathlib import Path
from difflib import SequenceMatcher
import re


def normalize_name(name):
    """Normalize a player name for better matching."""
    if pd.isna(name):
        return None
    
    name = str(name).lower().strip()
    
    # Remove common prefixes/suffixes
    name = re.sub(r'^(mr\.?|dr\.?|ms\.?)\s+', '', name)
    name = re.sub(r'\s+(jr\.?|sr\.?|ii|iii)$', '', name)
    
    # Remove special characters
    name = re.sub(r'[^\w\s]', '', name)
    
    # Normalize whitespace
    name = ' '.join(name.split())
    
    return name


def generate_name_variants(row):
    """Generate multiple name variants for matching."""
    variants = set()
    
    for col in ['unique_name', 'name', 'PLAYER', 'full_name']:
        if pd.notna(row.get(col)):
            name = str(row[col])
            variants.add(normalize_name(name))
            
            # Also try with/without middle names
            parts = name.split()
            if len(parts) >= 2:
                # First + Last
                variants.add(normalize_name(f"{parts[0]} {parts[-1]}"))
                # Initial + Last (e.g., "K Pandya")
                variants.add(normalize_name(f"{parts[0][0]} {parts[-1]}"))
    
    return [v for v in variants if v]


def fuzzy_match_aggressive(name, candidates, threshold=0.65):
    """More aggressive fuzzy matching with lower threshold."""
    best_match = None
    best_score = 0
    
    name_norm = normalize_name(name)
    if not name_norm:
        return None
    
    for candidate in candidates:
        cand_norm = normalize_name(candidate)
        if not cand_norm:
            continue
        
        # Try exact normalized match first
        if name_norm == cand_norm:
            return candidate
        
        # Fuzzy match
        score = SequenceMatcher(None, name_norm, cand_norm).ratio()
        if score > best_score:
            best_score = score
            best_match = candidate
        
        # Also try matching with parts
        name_parts = set(name_norm.split())
        cand_parts = set(cand_norm.split())
        overlap = len(name_parts & cand_parts) / max(len(name_parts), len(cand_parts))
        if overlap > best_score:
            best_score = overlap
            best_match = candidate
    
    return best_match if best_score >= threshold else None


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
    
    # V9 Enhanced (latest)
    try:
        v9_bat = pd.read_csv(results_dir / 'v9_enhanced' / 'batter_projections_2026_v9.csv')
        v9_bowl = pd.read_csv(results_dir / 'v9_enhanced' / 'bowler_projections_2026_v9.csv')
        
        # Add IDs from feature files
        bat_features = pd.read_csv(data_dir / 'ml_features' / 'batter_features_v9.csv')
        bowl_features = pd.read_csv(data_dir / 'ml_features' / 'bowler_features_v9.csv')
        
        bat_ids = bat_features.sort_values('season').groupby('batter_name')['batter_id'].last().reset_index()
        bowl_ids = bowl_features.sort_values('season').groupby('bowler_name')['bowler_id'].last().reset_index()
        
        v9_bat = v9_bat.merge(bat_ids, on='batter_name', how='left')
        v9_bowl = v9_bowl.merge(bowl_ids, on='bowler_name', how='left')
        
        preds['v9_bat'] = v9_bat
        preds['v9_bowl'] = v9_bowl
        print(f"V9 Enhanced: {len(v9_bat)} batters, {len(v9_bowl)} bowlers")
    except Exception as e:
        print(f"V9 error: {e}")
    
    # V8 Domestic
    try:
        v8_bat = pd.read_csv(results_dir / 'v8_domestic' / 'batter_domestic_predictions.csv')
        v8_bowl = pd.read_csv(results_dir / 'v8_domestic' / 'bowler_domestic_predictions.csv')
        preds['v8_bat'] = v8_bat
        preds['v8_bowl'] = v8_bowl
        print(f"V8 Domestic: {len(v8_bat)} batters, {len(v8_bowl)} bowlers")
    except Exception as e:
        print(f"V8 error: {e}")
    
    # V6 Production (fallback)
    try:
        v6_bat = pd.read_csv(results_dir / 'v6_production' / 'batter_projections_2026_prod.csv')
        v6_bowl = pd.read_csv(results_dir / 'v6_production' / 'bowler_projections_2026_prod.csv')
        
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
    """Create comprehensive lookup dictionaries."""
    bat_id = {}
    bat_name = {}
    bowl_id = {}
    bowl_name = {}
    
    # Priority: V9 > V6 > V8 > Marcel > Global
    sources_bat = [
        ('v9_bat', 'batter_id', 'batter_name', 'projected_war_2026', 'V9_Enhanced'),
        ('v6_bat', 'batter_id', 'batter_name', 'projected_war_2026', 'V6_Production'),
        ('v8_bat', 'batter_id', 'batter_name', 'predicted_ipl_war', 'V8_Domestic'),
        ('marcel_bat', 'player_id', 'player_name', 'projected_war_2026', 'Marcel'),
        ('global_bat', 'batter_id', 'batter_name', 'predicted_ipl_war', 'Global_Only'),
    ]
    
    sources_bowl = [
        ('v9_bowl', 'bowler_id', 'bowler_name', 'projected_war_2026', 'V9_Enhanced'),
        ('v6_bowl', 'bowler_id', 'bowler_name', 'projected_war_2026', 'V6_Production'),
        ('v8_bowl', 'bowler_id', 'bowler_name', 'predicted_ipl_war', 'V8_Domestic'),
        ('marcel_bowl', 'player_id', 'player_name', 'projected_war_2026', 'Marcel'),
        ('global_bowl', 'bowler_id', 'bowler_name', 'predicted_ipl_war', 'Global_Only'),
    ]
    
    for key, id_col, name_col, war_col, source in sources_bat:
        if key in preds:
            for _, row in preds[key].iterrows():
                pid = row.get(id_col)
                if pd.notna(pid) and pid not in bat_id:
                    bat_id[pid] = (row[war_col], source)
                name = normalize_name(row.get(name_col))
                if name and name not in bat_name:
                    bat_name[name] = (row[war_col], source)
    
    for key, id_col, name_col, war_col, source in sources_bowl:
        if key in preds:
            for _, row in preds[key].iterrows():
                pid = row.get(id_col)
                if pd.notna(pid) and pid not in bowl_id:
                    bowl_id[pid] = (row[war_col], source)
                name = normalize_name(row.get(name_col))
                if name and name not in bowl_name:
                    bowl_name[name] = (row[war_col], source)
    
    print(f"\nLookup sizes:")
    print(f"  Batter IDs: {len(bat_id)}, Names: {len(bat_name)}")
    print(f"  Bowler IDs: {len(bowl_id)}, Names: {len(bowl_name)}")
    
    return bat_id, bat_name, bowl_id, bowl_name


def score_player_v4(row, bat_id, bat_name, bowl_id, bowl_name):
    """Score player with aggressive matching."""
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
    
    # Step 2: Normalized name matching
    if war is None:
        variants = generate_name_variants(row)
        for name in variants:
            if name in bat_name:
                war, source = bat_name[name]
                method = 'Name'
                break
            if name in bowl_name:
                war, source = bowl_name[name]
                method = 'Name'
                break
    
    # Step 3: Aggressive fuzzy matching (lower threshold)
    if war is None and variants:
        all_names = list(bat_name.keys()) + list(bowl_name.keys())
        
        for variant in variants:
            match = fuzzy_match_aggressive(variant, all_names, threshold=0.65)
            if match:
                if match in bat_name:
                    war, source = bat_name[match]
                else:
                    war, source = bowl_name[match]
                method = 'Fuzzy65'
                break
    
    return war, source, method


def main():
    print("=" * 70)
    print("V4 COMBINED AUCTION SCORING")
    print("Uses: V9 Enhanced + V8 Domestic + Aggressive Matching")
    print("=" * 70)
    
    auction = load_auction_list()
    preds = load_all_predictions()
    bat_id, bat_name, bowl_id, bowl_name = create_lookups(preds)
    
    # Score all players
    results = []
    for _, row in auction.iterrows():
        war, source, method = score_player_v4(row, bat_id, bat_name, bowl_id, bowl_name)
        
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
    print("COVERAGE STATISTICS (V4 - AGGRESSIVE MATCHING)")
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
    
    # Compare with previous versions
    print("\n" + "=" * 70)
    print("COVERAGE COMPARISON")
    print("=" * 70)
    print("V1 (Legacy):     234/350 (66.9%)")
    print("V3 (Combined):   254/350 (72.6%)")
    print(f"V4 (Aggressive): {matched}/350 ({100*matched/total:.1f}%)")
    
    # Fill NaN
    results_df['projected_war_2026'] = results_df['projected_war_2026'].fillna(0)
    results_df['prediction_source'] = results_df['prediction_source'].fillna('Replacement_Level')
    results_df['match_method'] = results_df['match_method'].fillna('None')
    
    # Sort
    results_df = results_df.sort_values('projected_war_2026', ascending=False)
    
    # Save
    output_dir = Path(__file__).parent.parent / 'results' / 'WARprojections' / 'auction_2026_v4'
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / 'auction_pool_war_projections_v4.csv', index=False)
    
    # Print top players
    print("\n" + "=" * 70)
    print("TOP 30 AUCTION PLAYERS (V4)")
    print("=" * 70)
    print(results_df.head(30)[['player', 'country', 'role', 'projected_war_2026', 'prediction_source']].to_string(index=False))
    
    print(f"\n✓ Saved to: {output_dir / 'auction_pool_war_projections_v4.csv'}")


if __name__ == "__main__":
    main()
