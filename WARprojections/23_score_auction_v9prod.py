"""
V9 Production Auction Scoring - FIXED VERSION.

FIXES:
1. Ensures players with IPL history use V9_Production predictions
2. Explicitly checks IPL WAR history before using V8_Domestic
3. Clear priority: V9_Production > Marcel > V8_Domestic > Global_Only

OUTPUT: results/WARprojections/auction_2026_v9prod/
"""

import pandas as pd
import numpy as np
from pathlib import Path
from difflib import SequenceMatcher
import re


def normalize_name(name):
    if pd.isna(name):
        return None
    name = str(name).lower().strip()
    name = re.sub(r'^(mr\.?|dr\.?|ms\.?)\s+', '', name)
    name = re.sub(r'\s+(jr\.?|sr\.?|ii|iii)$', '', name)
    name = re.sub(r'[^\w\s]', '', name)
    name = ' '.join(name.split())
    return name


def generate_name_variants(row):
    variants = set()
    for col in ['unique_name', 'name', 'PLAYER', 'full_name']:
        if pd.notna(row.get(col)):
            name = str(row[col])
            variants.add(normalize_name(name))
            parts = name.split()
            if len(parts) >= 2:
                variants.add(normalize_name(f"{parts[0]} {parts[-1]}"))
                variants.add(normalize_name(f"{parts[0][0]} {parts[-1]}"))
    return [v for v in variants if v]


def fuzzy_match(name, candidates, threshold=0.65):
    best_match, best_score = None, 0
    name_norm = normalize_name(name)
    if not name_norm:
        return None
    
    for candidate in candidates:
        cand_norm = normalize_name(candidate) if candidate else None
        if not cand_norm:
            continue
        if name_norm == cand_norm:
            return candidate
        score = SequenceMatcher(None, name_norm, cand_norm).ratio()
        if score > best_score:
            best_score = score
            best_match = candidate
    
    return best_match if best_score >= threshold else None


def load_data():
    """Load all required data."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    results_dir = project_root / 'results' / 'WARprojections'
    
    data = {
        'auction': pd.read_csv(data_dir / 'ipl_2026_auction_enriched.csv'),
        'v9prod_bat': pd.read_csv(results_dir / 'v9_production' / 'batter_projections_2026_v9prod.csv'),
        'v9prod_bowl': pd.read_csv(results_dir / 'v9_production' / 'bowler_projections_2026_v9prod.csv'),
        'v8_bat': pd.read_csv(results_dir / 'v8_domestic' / 'batter_domestic_predictions.csv'),
        'v8_bowl': pd.read_csv(results_dir / 'v8_domestic' / 'bowler_domestic_predictions.csv'),
        'marcel_bat': pd.read_csv(results_dir / 'marcel' / 'batter_projections_2026.csv'),
        'marcel_bowl': pd.read_csv(results_dir / 'marcel' / 'bowler_projections_2026.csv'),
        'global_bat': pd.read_csv(results_dir / 'global_only' / 'batter_global_only_predictions.csv'),
        'global_bowl': pd.read_csv(results_dir / 'global_only' / 'bowler_global_only_predictions.csv'),
        # IPL history to check who has played IPL
        'ipl_bat_war': pd.read_csv(data_dir / 'batter_war_full_history.csv'),
        'ipl_bowl_war': pd.read_csv(data_dir / 'bowler_war_full_history.csv'),
    }
    
    # Create sets of IDs with IPL history
    data['ipl_batter_ids'] = set(data['ipl_bat_war']['batter_id'].unique())
    data['ipl_bowler_ids'] = set(data['ipl_bowl_war']['bowler_id'].unique())
    
    print(f"Auction: {len(data['auction'])} players")
    print(f"V9 Prod: {len(data['v9prod_bat'])} batters, {len(data['v9prod_bowl'])} bowlers")
    print(f"IPL history: {len(data['ipl_batter_ids'])} batters, {len(data['ipl_bowler_ids'])} bowlers")
    
    return data


def create_lookups(data):
    """Create lookups: first by ID, then by name."""
    # By ID - priority: V9 > Marcel > V8 > Global
    bat_id_lookup = {}
    bowl_id_lookup = {}
    bat_name_lookup = {}
    bowl_name_lookup = {}
    
    # V9 Production (HIGHEST PRIORITY)
    for _, row in data['v9prod_bat'].iterrows():
        pid = row.get('batter_id')
        if pd.notna(pid):
            bat_id_lookup[pid] = (row['projected_war_2026'], 'V9_Production')
        name = normalize_name(row.get('batter_name'))
        if name:
            bat_name_lookup[name] = (row['projected_war_2026'], 'V9_Production')
    
    for _, row in data['v9prod_bowl'].iterrows():
        pid = row.get('bowler_id')
        if pd.notna(pid):
            bowl_id_lookup[pid] = (row['projected_war_2026'], 'V9_Production')
        name = normalize_name(row.get('bowler_name'))
        if name:
            bowl_name_lookup[name] = (row['projected_war_2026'], 'V9_Production')
    
    # Marcel (2nd priority) - only if not already in lookup
    for _, row in data['marcel_bat'].iterrows():
        pid = row.get('player_id')
        if pd.notna(pid) and pid not in bat_id_lookup:
            bat_id_lookup[pid] = (row['projected_war_2026'], 'Marcel')
        name = normalize_name(row.get('player_name'))
        if name and name not in bat_name_lookup:
            bat_name_lookup[name] = (row['projected_war_2026'], 'Marcel')
    
    for _, row in data['marcel_bowl'].iterrows():
        pid = row.get('player_id')
        if pd.notna(pid) and pid not in bowl_id_lookup:
            bowl_id_lookup[pid] = (row['projected_war_2026'], 'Marcel')
        name = normalize_name(row.get('player_name'))
        if name and name not in bowl_name_lookup:
            bowl_name_lookup[name] = (row['projected_war_2026'], 'Marcel')
    
    # V8 Domestic (3rd priority) - ONLY for players WITHOUT IPL history
    for _, row in data['v8_bat'].iterrows():
        pid = row.get('batter_id')
        # Skip if player has IPL history (should use V9/Marcel/Global instead)
        if pd.notna(pid) and pid in data['ipl_batter_ids']:
            continue
        # Add to ID lookup if not already present
        if pd.notna(pid) and pid not in bat_id_lookup:
            bat_id_lookup[pid] = (row['predicted_ipl_war'], 'V8_Domestic')
        # Add to name lookup if not already present
        name = normalize_name(row.get('batter_name'))
        if name and name not in bat_name_lookup:
            bat_name_lookup[name] = (row['predicted_ipl_war'], 'V8_Domestic')
    
    for _, row in data['v8_bowl'].iterrows():
        pid = row.get('bowler_id')
        # Skip if player has IPL history (should use V9/Marcel/Global instead)
        if pd.notna(pid) and pid in data['ipl_bowler_ids']:
            continue
        if pd.notna(pid) and pid not in bowl_id_lookup:
            bowl_id_lookup[pid] = (row['predicted_ipl_war'], 'V8_Domestic')
        name = normalize_name(row.get('bowler_name'))
        if name and name not in bowl_name_lookup:
            bowl_name_lookup[name] = (row['predicted_ipl_war'], 'V8_Domestic')
    
    # Global-only (lowest priority)
    for _, row in data['global_bat'].iterrows():
        pid = row.get('batter_id')
        if pd.notna(pid) and pid not in bat_id_lookup:
            bat_id_lookup[pid] = (row['predicted_ipl_war'], 'Global_Only')
        name = normalize_name(row.get('batter_name'))
        if name and name not in bat_name_lookup:
            bat_name_lookup[name] = (row['predicted_ipl_war'], 'Global_Only')
    
    for _, row in data['global_bowl'].iterrows():
        pid = row.get('bowler_id')
        if pd.notna(pid) and pid not in bowl_id_lookup:
            bowl_id_lookup[pid] = (row['predicted_ipl_war'], 'Global_Only')
        name = normalize_name(row.get('bowler_name'))
        if name and name not in bowl_name_lookup:
            bowl_name_lookup[name] = (row['predicted_ipl_war'], 'Global_Only')
    
    print(f"\nLookup sizes:")
    print(f"  Batter IDs: {len(bat_id_lookup)}, Names: {len(bat_name_lookup)}")
    print(f"  Bowler IDs: {len(bowl_id_lookup)}, Names: {len(bowl_name_lookup)}")
    
    # Verification: check Pathirana
    pathirana_id = '64839cb3'
    if pathirana_id in bowl_id_lookup:
        print(f"\n✓ Pathirana in bowl_id_lookup: {bowl_id_lookup[pathirana_id]}")
    
    return bat_id_lookup, bat_name_lookup, bowl_id_lookup, bowl_name_lookup


def score_player(row, bat_id, bat_name, bowl_id, bowl_name):
    """Score auction player with proper priority and role awareness."""
    cricsheet_id = row.get('cricsheet_id')
    war, source, method = None, None, None
    
    # Determine player's primary role
    role = str(row.get('playing_role', '')).lower()
    is_bowler = 'bowler' in role or 'bowl' in role
    is_batter = 'batter' in role or 'bat' in role or 'wicketkeeper' in role
    
    # Step 1: ID matching (most reliable) - check both, prioritize by role
    if pd.notna(cricsheet_id):
        # Check role-appropriate lookup first
        if is_bowler and cricsheet_id in bowl_id:
            war, source = bowl_id[cricsheet_id]
            method = 'ID'
        elif is_batter and cricsheet_id in bat_id:
            war, source = bat_id[cricsheet_id]
            method = 'ID'
        elif cricsheet_id in bowl_id:  # Fallback for allrounders/unknown
            war, source = bowl_id[cricsheet_id]
            method = 'ID'
        elif cricsheet_id in bat_id:
            war, source = bat_id[cricsheet_id]
            method = 'ID'
    
    # Step 2: Name matching - role-aware order
    if war is None:
        variants = generate_name_variants(row)
        for name in variants:
            # Check role-appropriate lookup first
            if is_bowler:
                if name in bowl_name:
                    war, source = bowl_name[name]
                    method = 'Name'
                    break
                elif name in bat_name:
                    war, source = bat_name[name]
                    method = 'Name'
                    break
            else:  # Batter or unknown - check batter first
                if name in bat_name:
                    war, source = bat_name[name]
                    method = 'Name'
                    break
                elif name in bowl_name:
                    war, source = bowl_name[name]
                    method = 'Name'
                    break
    
    # Step 3: Fuzzy matching (role-aware)
    if war is None and variants:
        if is_bowler:
            pool = list(bowl_name.keys()) + list(bat_name.keys())
        else:
            pool = list(bat_name.keys()) + list(bowl_name.keys())
        
        for variant in variants:
            match = fuzzy_match(variant, pool, threshold=0.65)
            if match:
                if match in bat_name:
                    war, source = bat_name[match]
                else:
                    war, source = bowl_name[match]
                method = 'Fuzzy'
                break
    
    return war, source, method


def main():
    print("=" * 70)
    print("V9 PRODUCTION AUCTION SCORING - FIXED")
    print("Ensures IPL players use V9_Production")
    print("=" * 70)
    
    data = load_data()
    bat_id, bat_name, bowl_id, bowl_name = create_lookups(data)
    
    results = []
    for _, row in data['auction'].iterrows():
        war, source, method = score_player(row, bat_id, bat_name, bowl_id, bowl_name)
        
        results.append({
            'sr_no': row.get('SR. NO.'),
            'player': row.get('PLAYER') or row.get('name'),
            'country': row.get('COUNTRY') or row.get('country'),
            'role': row.get('playing_role'),
            'base_price': row.get('BASE PRICE (INR LAKH)'),
            'capped': row.get('C/U/A'),
            'projected_war_2026': war if war is not None else 0,
            'prediction_source': source if source else 'Replacement_Level',
            'match_method': method if method else 'None',
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('projected_war_2026', ascending=False)
    
    # Stats
    print("\n" + "=" * 70)
    print("COVERAGE")
    print("=" * 70)
    matched = (results_df['prediction_source'] != 'Replacement_Level').sum()
    print(f"Matched: {matched}/350 ({100*matched/350:.1f}%)")
    print(f"\nBy source:\n{results_df['prediction_source'].value_counts()}")
    
    # Verification: check Pathirana
    pathirana = results_df[results_df['player'].str.contains('Pathirana', case=False, na=False)]
    print(f"\n✓ Pathirana result: {pathirana[['player', 'projected_war_2026', 'prediction_source']].to_string(index=False)}")
    
    # Save
    output_dir = Path(__file__).parent.parent / 'results' / 'WARprojections' / 'auction_2026_v9prod'
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / 'auction_war_projections_v9prod.csv', index=False)
    
    # Print top players
    print("\n" + "=" * 70)
    print("TOP 50 AUCTION PLAYERS BY PROJECTED WAR")
    print("=" * 70)
    top50 = results_df.head(50)[['player', 'country', 'role', 'base_price', 'projected_war_2026', 'prediction_source']]
    print(top50.to_string(index=False))
    
    print(f"\n✓ Saved to: {output_dir / 'auction_war_projections_v9prod.csv'}")


if __name__ == "__main__":
    main()
