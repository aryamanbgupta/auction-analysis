"""
Score IPL 2026 Auction Pool with WAR Projections.

STRATEGY:
1. V6 Production (best) - for players who played IPL 2025
2. Marcel - for players with IPL history but not in 2025
3. Global-to-IPL model - for players with no IPL but global T20 data
4. Replacement level (0) - for players with no data

OUTPUT: results/WARprojections/auction_2026/
"""

import pandas as pd
import numpy as np
from pathlib import Path
from difflib import SequenceMatcher


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


def load_all_predictions():
    """Load all prediction sources."""
    results_dir = Path(__file__).parent.parent / 'results' / 'WARprojections'
    
    preds = {}
    
    # V6 Production (priority 1)
    try:
        preds['v6_bat'] = pd.read_csv(results_dir / 'v6_production' / 'batter_projections_2026_prod.csv')
        preds['v6_bowl'] = pd.read_csv(results_dir / 'v6_production' / 'bowler_projections_2026_prod.csv')
        print(f"V6 Production: {len(preds['v6_bat'])} batters, {len(preds['v6_bowl'])} bowlers")
    except Exception as e:
        print(f"V6 Production not found: {e}")
    
    # Combined (includes Marcel fallback)
    try:
        preds['combined_bat'] = pd.read_csv(results_dir / 'combined_2026' / 'batter_projections_2026_combined.csv')
        preds['combined_bowl'] = pd.read_csv(results_dir / 'combined_2026' / 'bowler_projections_2026_combined.csv')
        print(f"Combined: {len(preds['combined_bat'])} batters, {len(preds['combined_bowl'])} bowlers")
    except Exception as e:
        print(f"Combined not found: {e}")
    
    # Global-only (priority 3)
    try:
        preds['global_bat'] = pd.read_csv(results_dir / 'global_only' / 'batter_global_only_predictions.csv')
        preds['global_bowl'] = pd.read_csv(results_dir / 'global_only' / 'bowler_global_only_predictions.csv')
        print(f"Global-only: {len(preds['global_bat'])} batters, {len(preds['global_bowl'])} bowlers")
    except Exception as e:
        print(f"Global-only not found: {e}")
    
    return preds


def create_prediction_lookup(preds):
    """Create name -> (war, source) lookup dictionaries."""
    # Batter lookup
    bat_lookup = {}
    
    # V6 first (highest priority)
    if 'v6_bat' in preds:
        for _, row in preds['v6_bat'].iterrows():
            name = row['batter_name'].lower().strip()
            bat_lookup[name] = (row['projected_war_2026'], 'V6_Production')
    
    # Combined fills gaps 
    if 'combined_bat' in preds:
        for _, row in preds['combined_bat'].iterrows():
            name = row['player_name'].lower().strip()
            if name not in bat_lookup:
                bat_lookup[name] = (row['projected_war_2026'], row['source'])
    
    # Global-only for remaining
    if 'global_bat' in preds:
        for _, row in preds['global_bat'].iterrows():
            name = row['batter_name'].lower().strip()
            if name not in bat_lookup:
                bat_lookup[name] = (row['predicted_ipl_war'], 'Global_Only')
    
    # Bowler lookup
    bowl_lookup = {}
    
    if 'v6_bowl' in preds:
        for _, row in preds['v6_bowl'].iterrows():
            name = row['bowler_name'].lower().strip()
            bowl_lookup[name] = (row['projected_war_2026'], 'V6_Production')
    
    if 'combined_bowl' in preds:
        for _, row in preds['combined_bowl'].iterrows():
            name = row['player_name'].lower().strip()
            if name not in bowl_lookup:
                bowl_lookup[name] = (row['projected_war_2026'], row['source'])
    
    if 'global_bowl' in preds:
        for _, row in preds['global_bowl'].iterrows():
            name = row['bowler_name'].lower().strip()
            if name not in bowl_lookup:
                bowl_lookup[name] = (row['predicted_ipl_war'], 'Global_Only')
    
    print(f"\nLookup tables: {len(bat_lookup)} batters, {len(bowl_lookup)} bowlers")
    
    return bat_lookup, bowl_lookup


def match_player(row, bat_lookup, bowl_lookup):
    """Find prediction for an auction player."""
    # Try multiple name columns
    name_options = []
    
    if pd.notna(row.get('unique_name')):
        name_options.append(row['unique_name'].lower().strip())
    if pd.notna(row.get('name')):
        name_options.append(row['name'].lower().strip())
    if pd.notna(row.get('PLAYER')):
        name_options.append(row['PLAYER'].lower().strip())
    if pd.notna(row.get('full_name')):
        name_options.append(row['full_name'].lower().strip())
    
    # Determine role
    role = row.get('playing_role', '')
    is_bowler = 'bowler' in str(role).lower() or 'bowl' in str(row.get('SET', '')).lower()
    is_batter = 'batter' in str(role).lower() or 'bat' in str(row.get('SET', '')).lower()
    is_allrounder = 'allrounder' in str(role).lower()
    
    war = None
    source = None
    
    # Try exact matches first
    for name in name_options:
        # Try batting lookup
        if name in bat_lookup:
            war, source = bat_lookup[name]
            break
        # Try bowling lookup
        if name in bowl_lookup:
            war, source = bowl_lookup[name]
            break
    
    # If still no match, try fuzzy matching
    if war is None and name_options:
        primary_name = name_options[0]
        
        # Search in appropriate lookup
        if is_bowler and not is_batter:
            match = fuzzy_match(primary_name, bowl_lookup.keys(), threshold=0.75)
            if match:
                war, source = bowl_lookup[match]
                source = f"{source} (fuzzy)"
        elif is_batter or is_allrounder:
            match = fuzzy_match(primary_name, bat_lookup.keys(), threshold=0.75)
            if match:
                war, source = bat_lookup[match]
                source = f"{source} (fuzzy)"
        else:
            # Try both
            match = fuzzy_match(primary_name, list(bat_lookup.keys()) + list(bowl_lookup.keys()), threshold=0.75)
            if match:
                if match in bat_lookup:
                    war, source = bat_lookup[match]
                else:
                    war, source = bowl_lookup[match]
                source = f"{source} (fuzzy)"
    
    return war, source


def main():
    print("="*60)
    print("SCORING IPL 2026 AUCTION POOL")
    print("="*60)
    
    # Load auction list
    data_dir = Path(__file__).parent.parent / 'data'
    auction = pd.read_csv(data_dir / 'ipl_2026_auction_enriched.csv')
    print(f"\nAuction pool: {len(auction)} players")
    
    # Load predictions
    preds = load_all_predictions()
    bat_lookup, bowl_lookup = create_prediction_lookup(preds)
    
    # Score each player
    results = []
    
    for idx, row in auction.iterrows():
        war, source = match_player(row, bat_lookup, bowl_lookup)
        
        results.append({
            'sr_no': row.get('SR. NO.'),
            'player': row.get('PLAYER') or row.get('name'),
            'country': row.get('COUNTRY') or row.get('country'),
            'role': row.get('playing_role'),
            'base_price': row.get('BASE PRICE (INR LAKH)'),
            'capped_status': row.get('C/U/A'),
            'projected_war_2026': war,
            'prediction_source': source,
            'cricsheet_id': row.get('cricsheet_id'),
        })
    
    results_df = pd.DataFrame(results)
    
    # Statistics
    print("\n" + "="*60)
    print("COVERAGE STATISTICS")
    print("="*60)
    
    total = len(results_df)
    matched = results_df['projected_war_2026'].notna().sum()
    unmatched = total - matched
    
    print(f"Total players: {total}")
    print(f"With predictions: {matched} ({100*matched/total:.1f}%)")
    print(f"No prediction: {unmatched} ({100*unmatched/total:.1f}%)")
    
    print("\nBy source:")
    print(results_df['prediction_source'].value_counts(dropna=False))
    
    # Fill NaN with 0 (replacement level)
    results_df['projected_war_2026'] = results_df['projected_war_2026'].fillna(0)
    results_df['prediction_source'] = results_df['prediction_source'].fillna('Replacement_Level')
    
    # Sort by projected WAR
    results_df = results_df.sort_values('projected_war_2026', ascending=False)
    
    # Save
    output_dir = Path(__file__).parent.parent / 'results' / 'WARprojections' / 'auction_2026'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(output_dir / 'auction_pool_war_projections.csv', index=False)
    
    # Show top players
    print("\n" + "="*60)
    print("TOP 30 AUCTION PLAYERS BY PROJECTED WAR")
    print("="*60)
    top30 = results_df.head(30)[['player', 'country', 'role', 'base_price', 'projected_war_2026', 'prediction_source']]
    print(top30.to_string(index=False))
    
    # Show by role
    print("\n" + "="*60)
    print("TOP 10 BY ROLE")
    print("="*60)
    
    for role_keyword in ['batter', 'bowler', 'allrounder', 'Wicketkeeper']:
        role_players = results_df[results_df['role'].str.contains(role_keyword, case=False, na=False)]
        if len(role_players) > 0:
            print(f"\n{role_keyword.upper()}S:")
            print(role_players.head(10)[['player', 'projected_war_2026', 'prediction_source']].to_string(index=False))
    
    # Show unmatched players
    unmatched_df = results_df[results_df['prediction_source'] == 'Replacement_Level']
    if len(unmatched_df) > 0:
        print(f"\n--- PLAYERS WITH NO DATA (Replacement Level: 0 WAR) ---")
        print(f"Count: {len(unmatched_df)}")
        print(unmatched_df[['player', 'country', 'role']].head(20).to_string(index=False))
    
    print(f"\n✓ Results saved to: {output_dir / 'auction_pool_war_projections.csv'}")


if __name__ == "__main__":
    main()
