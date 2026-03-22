"""
Create Combined WAR Projections for 2026.

PRIORITY ORDER:
1. V6 Production (best model, IPL 2025 players only)
2. Marcel (players with any IPL history)
3. Global ML (players with global T20 data but no recent IPL)
4. Replacement Level (0 WAR for unknown players)

OUTPUT: results/WARprojections/combined_2026/
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_all_projections():
    """Load all available projections."""
    results_dir = Path(__file__).parent.parent / 'results' / 'WARprojections'
    
    projections = {}
    
    # V6 Production (best)
    try:
        projections['v6_bat'] = pd.read_csv(results_dir / 'v6_production' / 'batter_projections_2026_prod.csv')
        projections['v6_bowl'] = pd.read_csv(results_dir / 'v6_production' / 'bowler_projections_2026_prod.csv')
        print(f"V6 Production: {len(projections['v6_bat'])} batters, {len(projections['v6_bowl'])} bowlers")
    except:
        print("V6 Production not found")
    
    # Marcel
    try:
        projections['marcel_bat'] = pd.read_csv(results_dir / 'marcel' / 'batter_projections_2026.csv')
        projections['marcel_bowl'] = pd.read_csv(results_dir / 'marcel' / 'bowler_projections_2026.csv')
        print(f"Marcel: {len(projections['marcel_bat'])} batters, {len(projections['marcel_bowl'])} bowlers")
    except:
        print("Marcel not found")
    
    # Global ML
    try:
        projections['global_bat'] = pd.read_csv(results_dir / 'batter_projections_2026_global.csv')
        projections['global_bowl'] = pd.read_csv(results_dir / 'bowler_projections_2026_global.csv')
        print(f"Global ML: {len(projections['global_bat'])} batters, {len(projections['global_bowl'])} bowlers")
    except:
        print("Global ML not found")
    
    return projections


def load_all_known_players():
    """Load all players from various sources."""
    data_dir = Path(__file__).parent.parent / 'data'
    
    all_players = set()
    
    # From IPL history
    try:
        ipl = pd.read_parquet(data_dir / 'ipl_matches_all.parquet')
        all_players.update(ipl['batter_name'].unique())
        all_players.update(ipl['bowler_name'].unique())
        print(f"IPL history: {len(all_players)} unique players")
    except:
        pass
    
    # From global data
    try:
        global_data = pd.read_parquet(data_dir / 'global_t20_matches.parquet')
        all_players.update(global_data['batter_name'].unique())
        all_players.update(global_data['bowler_name'].unique())
        print(f"After global data: {len(all_players)} unique players")
    except:
        pass
    
    return all_players


def combine_projections(projections, role):
    """Combine projections with priority: V6 > Marcel > Global > 0."""
    name_col = 'batter_name' if role == 'batter' else 'bowler_name'
    
    combined = {}
    sources = {}
    
    # Start with V6 (highest priority)
    v6_key = f'v6_{role[:3]}' if role == 'batter' else 'v6_bowl'
    if v6_key in projections:
        for _, row in projections[v6_key].iterrows():
            name = row[name_col]
            combined[name] = row['projected_war_2026']
            sources[name] = 'V6_Production'
    
    # Add Marcel for missing players
    marcel_key = f'marcel_{role[:3]}' if role == 'batter' else 'marcel_bowl'
    if marcel_key in projections:
        for _, row in projections[marcel_key].iterrows():
            name = row['player_name']
            if name not in combined:
                combined[name] = row['projected_war_2026']
                sources[name] = 'Marcel'
    
    # Add Global ML for remaining players
    global_key = f'global_{role[:3]}' if role == 'batter' else 'global_bowl'
    if global_key in projections:
        for _, row in projections[global_key].iterrows():
            name = row[name_col]
            if name not in combined:
                combined[name] = row['projected_war_2026']
                sources[name] = 'Global_ML'
    
    # Create DataFrame
    result = pd.DataFrame({
        'player_name': list(combined.keys()),
        'projected_war_2026': list(combined.values()),
        'source': [sources[k] for k in combined.keys()]
    })
    
    return result.sort_values('projected_war_2026', ascending=False)


def main():
    print("="*60)
    print("CREATING COMBINED 2026 PROJECTIONS")
    print("="*60)
    
    projections = load_all_projections()
    
    print("\n--- Combining Batter Projections ---")
    bat_combined = combine_projections(projections, 'batter')
    
    print("\n--- Combining Bowler Projections ---")  
    bowl_combined = combine_projections(projections, 'bowler')
    
    # Statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print(f"\nBatters ({len(bat_combined)} total):")
    print(bat_combined['source'].value_counts().to_string())
    
    print(f"\nBowlers ({len(bowl_combined)} total):")
    print(bowl_combined['source'].value_counts().to_string())
    
    # Save
    output_dir = Path(__file__).parent.parent / 'results' / 'WARprojections' / 'combined_2026'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    bat_combined.to_csv(output_dir / 'batter_projections_2026_combined.csv', index=False)
    bowl_combined.to_csv(output_dir / 'bowler_projections_2026_combined.csv', index=False)
    
    # Show top players
    print("\n" + "="*60)
    print("TOP 15 BATTERS (Combined)")
    print("="*60)
    print(bat_combined.head(15).to_string(index=False))
    
    print("\n" + "="*60)
    print("TOP 15 BOWLERS (Combined)")
    print("="*60)
    print(bowl_combined.head(15).to_string(index=False))
    
    # Show source breakdown for top 20
    print("\n" + "="*60)
    print("TOP 20 BATTERS BY SOURCE")
    print("="*60)
    top20_bat = bat_combined.head(20)
    for source in ['V6_Production', 'Marcel', 'Global_ML']:
        subset = top20_bat[top20_bat['source'] == source]
        if len(subset) > 0:
            print(f"\n{source} ({len(subset)} players):")
            print(subset[['player_name', 'projected_war_2026']].to_string(index=False))
    
    print("\n" + "="*60)
    print("✓ Combined projections saved to:")
    print(f"  {output_dir}/batter_projections_2026_combined.csv")
    print(f"  {output_dir}/bowler_projections_2026_combined.csv")
    print("="*60)


if __name__ == "__main__":
    main()
