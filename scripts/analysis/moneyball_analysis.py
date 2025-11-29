import pandas as pd
import numpy as np
from pathlib import Path
from moneyball_price_model import MoneyballPriceModel

def load_data(project_root):
    print("Loading data...")
    # Main ball-by-ball data (for Death Bowling stats)
    ipl_df = pd.read_parquet(project_root / 'results' / '06_context_adjustments' / 'ipl_with_raa.parquet')
    ipl_df['season'] = ipl_df['season'].astype(str)
    
    # Auction Pool
    pool_batters = pd.read_csv(project_root / 'results' / 'analysis' / 'auction_pool' / 'auction_pool_batters.csv', index_col=0)
    pool_bowlers = pd.read_csv(project_root / 'results' / 'analysis' / 'auction_pool' / 'auction_pool_bowlers.csv', index_col=0)
    
    # All Players (2022-2025 Stats)
    all_batters = pd.read_csv(project_root / 'results' / 'analysis' / 'auction_stats' / 'batters_2022_2025.csv', index_col=0)
    all_bowlers = pd.read_csv(project_root / 'results' / 'analysis' / 'auction_stats' / 'bowlers_2022_2025.csv', index_col=0)
    
    # Metadata
    metadata = pd.read_csv(project_root / 'data' / 'player_metadata.csv')
    
    # Projections
    proj_bat = pd.read_csv(project_root / 'results' / 'WARprojections' / 'batter_projections_2026.csv')
    proj_bowl = pd.read_csv(project_root / 'results' / 'WARprojections' / 'bowler_projections_2026.csv')
    
    return ipl_df, pool_batters, pool_bowlers, all_batters, all_bowlers, metadata, proj_bat, proj_bowl

def get_targets(ipl_df, pool_batters, pool_bowlers, all_batters, all_bowlers, metadata):
    print("Identifying Strategic Targets...")
    
    def norm(n): return str(n).strip().lower()
    
    # Identify Auction Players
    auction_batters = set(pool_batters.index.map(norm))
    auction_bowlers = set(pool_bowlers.index.map(norm))
    
    targets = []
    
    # --- 1. Openers (All) ---
    # Filter: Balls_powerplay >= 60
    openers = all_batters[all_batters['Balls_powerplay'] >= 60].copy()
    for name in openers.index:
        if norm(name) in auction_batters:
            targets.append({'name': name, 'role': 'Opener', 'source': 'Strategic Analysis'})

    # --- 2. Middle Order ---
    # Filter: Balls_middle >= 60
    middle = all_batters[all_batters['Balls_middle'] >= 60].copy()
    for name in middle.index:
        if norm(name) in auction_batters:
            targets.append({'name': name, 'role': 'Middle Order', 'source': 'Strategic Analysis'})

    # --- 3. Powerplay Pacers ---
    # Filter: Balls_powerplay >= 60 & Type == Pace
    bowler_types = ipl_df.groupby('bowler_name')['bowling_type'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'pace')
    all_bowlers['Type'] = all_bowlers.index.map(bowler_types)
    pp_pacers = all_bowlers[(all_bowlers['Balls_powerplay'] >= 60) & (all_bowlers['Type'] == 'pace')].copy()
    
    for name in pp_pacers.index:
        if norm(name) in auction_bowlers:
            targets.append({'name': name, 'role': 'Powerplay Pacer', 'source': 'Strategic Analysis'})

    # --- 4. Death Bowlers (Qualified) ---
    # Filter: 2024-2025, Balls >= 60, Econ < Threshold, Dot% > 30
    df_recent = ipl_df[ipl_df['season'].isin(['2024', '2025'])]
    death_recent = df_recent[df_recent['over'] >= 16]
    
    death_stats = death_recent.groupby('bowler_name').agg({
        'total_runs': 'sum', 'ball_in_over': 'count'
    }).rename(columns={'ball_in_over': 'Balls', 'total_runs': 'Runs'})
    
    dots = death_recent[death_recent['total_runs'] == 0].groupby('bowler_name')['ball_in_over'].count()
    death_stats['Dots'] = dots.fillna(0)
    death_stats['Econ'] = (death_stats['Runs'] / death_stats['Balls']) * 6
    death_stats['Dot_Pct'] = (death_stats['Dots'] / death_stats['Balls']) * 100
    
    league_avg_econ = (death_recent['total_runs'].sum() / death_recent['ball_in_over'].count()) * 6
    threshold = league_avg_econ - 0.5
    
    qualified_death = death_stats[
        (death_stats['Balls'] >= 60) & 
        (death_stats['Econ'] < threshold) & 
        (death_stats['Dot_Pct'] > 30)
    ]
    
    for name in qualified_death.index:
        if norm(name) in auction_bowlers:
            targets.append({'name': name, 'role': 'Death Bowler', 'source': 'Strategic Analysis'})
            
    return pd.DataFrame(targets).drop_duplicates(subset=['name'])

def main():
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / 'results' / 'analysis' / 'moneyball'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    ipl_df, pool_batters, pool_bowlers, all_batters, all_bowlers, metadata, proj_bat, proj_bowl = load_data(project_root)
    
    # 2. Get Targets
    targets_df = get_targets(ipl_df, pool_batters, pool_bowlers, all_batters, all_bowlers, metadata)
    print(f"Total Unique Targets: {len(targets_df)}")
    
    # 3. Merge Projections (2026)
    # Projections have 'player_name' and 'WAR_2026' (or similar, let's check column names dynamically if needed)
    # Assuming standard names.
    
    # Normalize names for merging
    def norm(n): return str(n).strip().lower()
    targets_df['name_norm'] = targets_df['name'].apply(norm)
    
    # Rename columns to standard 'player_name' and 'WAR_2026'
    proj_bat = proj_bat.rename(columns={'batter_name': 'player_name', 'projected_war_2026': 'WAR_2026'})
    proj_bowl = proj_bowl.rename(columns={'bowler_name': 'player_name', 'projected_war_2026': 'WAR_2026'})
    
    proj_bat['name_norm'] = proj_bat['player_name'].apply(norm)
    proj_bowl['name_norm'] = proj_bowl['player_name'].apply(norm)
    
    # Merge
    merged_bat = targets_df.merge(proj_bat[['name_norm', 'WAR_2026']], on='name_norm', how='left')
    merged_all = merged_bat.merge(proj_bowl[['name_norm', 'WAR_2026']], on='name_norm', how='left', suffixes=('_bat', '_bowl'))
    
    # Combine WAR
    # If in both, sum them? Or take max?
    # A player can have batting WAR and bowling WAR. Total WAR is sum.
    merged_all['WAR_2026'] = merged_all['WAR_2026_bat'].fillna(0) + merged_all['WAR_2026_bowl'].fillna(0)
    
    # 4. Add Metadata (Overseas, Role)
    metadata['name_norm'] = metadata['player_name'].apply(norm)
    final_df = merged_all.merge(metadata[['name_norm', 'country', 'role_category']], on='name_norm', how='left')
    
    final_df['is_overseas'] = final_df['country'].apply(lambda x: 0 if str(x).lower() in ['india', 'ind'] else 1)
    final_df['role_category'] = final_df['role_category'].fillna('Unknown')
    
    # 5. Predict Price
    price_model = MoneyballPriceModel(project_root)
    price_model.load_and_train()
    
    # Prepare for prediction
    # Rename for model compatibility
    # NOTE: The training data (war_vs_price_full.csv) appears to use WAR Per Match (values ~0.1 to 1.0).
    # The projections (WAR_2026) are Total Season WAR (values ~1.0 to 8.0).
    # We assume a standard 14-match season to convert Projections to Model Scale.
    final_df['WAR'] = final_df['WAR_2026'] / 14.0
    
    print("Predicting Prices...")
    final_df['Predicted_Price_Cr'] = price_model.predict_batch(final_df)
    
    # Clip negative prices to base price (e.g., 0.2 Cr)
    final_df['Predicted_Price_Cr'] = final_df['Predicted_Price_Cr'].clip(lower=0.20)
    
    # 6. Calculate Value
    # Value = WAR / Price
    final_df['Value_Ratio'] = final_df['WAR_2026'] / final_df['Predicted_Price_Cr']
    
    # 7. Save
    output_file = output_dir / 'moneyball_targets.csv'
    final_df = final_df.sort_values('Value_Ratio', ascending=False)
    
    columns_to_save = ['name', 'role', 'country', 'role_category', 'WAR_2026', 'Predicted_Price_Cr', 'Value_Ratio']
    final_df[columns_to_save].to_csv(output_file, index=False)
    
    print(f"Moneyball Analysis Saved to {output_file}")
    print("Top 5 Value Picks:")
    print(final_df[columns_to_save].head(5))

if __name__ == "__main__":
    main()
