import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
from pathlib import Path
from valuation_engine import ValuationEngine

# Set default template for plotly
pio.templates.default = "plotly_white"

def load_data(project_root):
    print("Loading data...")
    # Main ball-by-ball data
    ipl_df = pd.read_parquet(project_root / 'results' / '06_context_adjustments' / 'ipl_with_raa.parquet')
    ipl_df['season'] = ipl_df['season'].astype(str)
    
    # Auction Pool (Available Players)
    pool_batters = pd.read_csv(project_root / 'results' / 'analysis' / 'auction_pool' / 'auction_pool_batters.csv', index_col=0)
    pool_bowlers = pd.read_csv(project_root / 'results' / 'analysis' / 'auction_pool' / 'auction_pool_bowlers.csv', index_col=0)
    
    # All Players (2022-2025 Stats) - For Comparison
    all_batters = pd.read_csv(project_root / 'results' / 'analysis' / 'auction_stats' / 'batters_2022_2025.csv', index_col=0)
    all_bowlers = pd.read_csv(project_root / 'results' / 'analysis' / 'auction_stats' / 'bowlers_2022_2025.csv', index_col=0)
    
    # Metadata
    metadata = pd.read_csv(project_root / 'data' / 'player_metadata.csv')
    
    return ipl_df, pool_batters, pool_bowlers, all_batters, all_bowlers, metadata

def save_plot(fig, output_dir, filename):
    fig.write_html(output_dir / f"{filename}.html")
    try:
        fig.write_image(output_dir / f"{filename}.png", scale=2)
    except Exception as e:
        print(f"Warning: Could not save static image for {filename}: {e}")

def add_valuation_data(df, engine):
    """
    Enrich dataframe with WAR_2026 and Est_Price columns using the ValuationEngine.
    Expects df index to be player names.
    """
    war_list = []
    price_list = []
    vomam_list = []
    vope_list = []
    
    # Cache projections for speed
    # engine.projections is a DF. Convert to dict for faster lookup?
    # engine.projections has 'name_norm' and 'war_2026'
    proj_map = engine.projections.set_index('name_norm')['war_2026'].to_dict()
    
    # Metadata lookup
    # Deduplicate metadata by name_norm to avoid index errors
    meta_unique = engine.metadata.drop_duplicates(subset='name_norm')
    meta_map = meta_unique.set_index('name_norm')[['role_category', 'country']].to_dict('index')
    
    def normalize(n): return str(n).strip().lower()
    
    for name in df.index:
        n_norm = normalize(name)
        
        # 1. WAR
        war = proj_map.get(n_norm, 0.0)
        if pd.isna(war): war = 0.0
        war_list.append(round(war, 2))
        
        # 2. Price
        meta = meta_map.get(n_norm, {'role_category': 'Unknown', 'country': 'Unknown'})
        role = meta['role_category']
        country = meta['country']
        is_overseas = 0 if str(country).lower() in ['india', 'ind'] else 1
        
        vomam, vope = engine.predict_price(war, is_overseas, role)
        avg_price = (vomam + vope) / 2
        
        price_list.append(round(avg_price, 2))
        vomam_list.append(round(vomam, 2))
        vope_list.append(round(vope, 2))
        
    df['WAR_2026'] = war_list
    df['Est_Price'] = price_list
    df['Est_Price_VOMAM'] = vomam_list
    df['Est_Price_VOPE'] = vope_list
    return df

def phase1_diagnosis(ipl_df, output_dir):
    print("Running Phase 1: Diagnosis...")
    df_2025 = ipl_df[ipl_df['season'] == '2025']
    
    # --- Module 1: Middle Over Slump ---
    print("  Module 1: Middle Over Slump")
    middle_overs = df_2025[(df_2025['over'] >= 6) & (df_2025['over'] <= 14)]
    
    team_stats = middle_overs.groupby(['batting_team', 'bowling_type']).agg({
        'total_runs': 'sum', 'ball_in_over': 'count'
    }).reset_index()
    
    overall_stats = middle_overs.groupby('batting_team').agg({
        'total_runs': 'sum', 'ball_in_over': 'count'
    }).reset_index()
    overall_stats['bowling_type'] = 'Overall'
    
    combined_stats = pd.concat([team_stats, overall_stats])
    combined_stats['RunRate'] = (combined_stats['total_runs'] / combined_stats['ball_in_over']) * 6
    
    dc_stats = combined_stats[combined_stats['batting_team'] == 'Delhi Capitals'].copy()
    league_stats = combined_stats[combined_stats['batting_team'] != 'Delhi Capitals'].groupby('bowling_type')['RunRate'].mean().reset_index()
    
    comparison = pd.merge(dc_stats, league_stats, on='bowling_type', suffixes=('_DC', '_League'))
    
    ranks = []
    for btype in ['Overall', 'pace', 'spin']:
        subset = combined_stats[combined_stats['bowling_type'] == btype].copy()
        subset['Rank'] = subset['RunRate'].rank(ascending=False)
        dc_rank = subset[subset['batting_team'] == 'Delhi Capitals']['Rank'].values[0]
        ranks.append({'bowling_type': btype, 'Rank': int(dc_rank)})
    
    rank_df = pd.DataFrame(ranks)
    comparison = pd.merge(comparison, rank_df, on='bowling_type')
    
    comp_melted = comparison.melt(id_vars=['bowling_type', 'Rank'], value_vars=['RunRate_DC', 'RunRate_League'], 
                                  var_name='Team', value_name='RunRate')
    comp_melted['Team'] = comp_melted['Team'].map({'RunRate_DC': 'Delhi Capitals', 'RunRate_League': 'League Average'})
    
    fig = px.bar(comp_melted, x='bowling_type', y='RunRate', color='Team', barmode='group',
                 title='DC Middle Overs Performance (2025)', text_auto='.2f',
                 color_discrete_map={'Delhi Capitals': 'blue', 'League Average': 'grey'})
    
    for i, row in comparison.iterrows():
        fig.add_annotation(x=row['bowling_type'], y=row['RunRate_DC'] + 0.5,
                           text=f"Rank #{row['Rank']}", showarrow=False, font=dict(color='blue', size=14))
    save_plot(fig, output_dir, 'diagnosis_middle_overs')
    
    # --- Module 2: Opening Stability ---
    print("  Module 2: Opening Stability")
    openers = df_2025[(df_2025['over'] == 0) & (df_2025['ball_in_over'] == 1)].copy()
    openers['pair'] = openers.apply(lambda x: tuple(sorted([x['batter_name'], x['non_striker_name']])), axis=1)
    stability = openers.groupby('batting_team')['pair'].nunique().reset_index(name='Unique_Pairs')
    
    pp_overs = df_2025[df_2025['over'] < 6]
    pp_scores = pp_overs.groupby(['match_id', 'batting_team'])['total_runs'].sum().reset_index()
    avg_pp = pp_scores.groupby('batting_team')['total_runs'].mean().reset_index(name='Avg_PP_Score')
    
    stability_perf = pd.merge(stability, avg_pp, on='batting_team')
    
    fig = px.scatter(stability_perf, x='Unique_Pairs', y='Avg_PP_Score', text='batting_team',
                     title='Opening Partnership Stability vs Performance', size='Avg_PP_Score', 
                     color='batting_team', color_discrete_map={'Delhi Capitals': 'red'})
    fig.update_traces(textposition='top center')
    save_plot(fig, output_dir, 'diagnosis_opening_stability')

    # --- Module 3: Death Bowling Crisis ---
    print("  Module 3: Death Bowling Crisis")
    death_bowling = df_2025[df_2025['over'] >= 16]
    
    # Calculate Economy
    death_stats = death_bowling.groupby('bowling_team').agg({
        'total_runs': 'sum', 
        'ball_in_over': 'count',
        'match_id': 'nunique'
    }).reset_index()
    death_stats['Death_Econ'] = (death_stats['total_runs'] / death_stats['ball_in_over']) * 6
    
    # Calculate Sixes per Match ("Damage")
    sixes = death_bowling[death_bowling['batter_runs'] == 6].groupby('bowling_team')['ball_in_over'].count().reset_index(name='Sixes')
    death_stats = pd.merge(death_stats, sixes, on='bowling_team', how='left').fillna(0)
    death_stats['Sixes_Per_Match'] = death_stats['Sixes'] / death_stats['match_id']
    
    death_stats = death_stats.sort_values('Death_Econ', ascending=True)
    
    # Plot 1: Economy Rate (League Ladder)
    fig = px.bar(death_stats, y='bowling_team', x='Death_Econ', orientation='h',
                 title='Death Overs Economy Rate (2025)', text_auto='.2f',
                 color='bowling_team', color_discrete_map={'Delhi Capitals': 'red'})
    fig.add_vline(x=death_stats['Death_Econ'].mean(), line_dash="dash", annotation_text="League Avg")
    save_plot(fig, output_dir, 'diagnosis_death_bowling')
    
    # Plot 2: Damage (Sixes per Match) - Optional but requested
    fig2 = px.bar(death_stats.sort_values('Sixes_Per_Match'), y='bowling_team', x='Sixes_Per_Match', orientation='h',
                  title='Death Overs Damage (Sixes per Match)', text_auto='.1f',
                  color='bowling_team', color_discrete_map={'Delhi Capitals': 'red'})
    save_plot(fig2, output_dir, 'diagnosis_death_damage')

    # --- Module 6: Powerplay Bowling ---
    print("  Module 6: Powerplay Bowling Diagnosis")
    pp_bowling = df_2025[df_2025['over'] < 6]
    pp_wickets = pp_bowling.groupby('bowling_team')['is_wicket'].sum().reset_index(name='PP_Wickets')
    pp_wickets = pp_wickets.sort_values('PP_Wickets', ascending=True)
    
    fig = px.bar(pp_wickets, y='bowling_team', x='PP_Wickets', orientation='h',
                 title='Powerplay Wickets (2025)', text_auto=True,
                 color='bowling_team', color_discrete_map={'Delhi Capitals': 'red'})
    fig.add_vline(x=pp_wickets['PP_Wickets'].mean(), line_dash="dash", annotation_text="League Avg")
    save_plot(fig, output_dir, 'diagnosis_pp_bowling')

def phase2_solution(ipl_df, pool_batters, pool_bowlers, all_batters, all_bowlers, metadata, output_dir, engine):
    print("Running Phase 2: Solution (Comparison Mode)...")
    
    def norm(n): return str(n).strip().lower()
    metadata['name_norm'] = metadata['player_name'].apply(norm)
    country_map = metadata.set_index('name_norm')['country'].to_dict()
    
    # --- Prepare Batters (All + Status) ---
    all_batters['name_norm'] = all_batters.index.map(norm)
    all_batters['Country'] = all_batters['name_norm'].map(country_map)
    
    # Determine Status
    pool_batters_norm = set(pool_batters.index.map(norm))
    all_batters['Status'] = all_batters['name_norm'].apply(lambda x: 'Auction' if x in pool_batters_norm else 'Retained')
    
    # --- Add Valuation Data to All Batters ---
    print("  Adding Valuation Data to Batters...")
    all_batters = add_valuation_data(all_batters, engine)
    
    # --- Module 4: Explosive Opener Search ---
    print("  Module 4: Explosive Opener Search")
    # Filter: Overseas & Min Balls
    overseas_batters = all_batters[all_batters['Country'] != 'India'].copy()
    
    # Strict Filters:
    # 1. Speed: PP SR > 145
    # 2. Quality: Avg > 25
    # 3. Spin Proficiency: SR vs Spin > 130
    
    qualified_openers = overseas_batters[
        (overseas_batters['Balls_powerplay'] >= 60) &
        (overseas_batters['SR_powerplay'] > 145) &
        (overseas_batters['Avg'] > 25) &
        (overseas_batters['SR_spin'] > 130)
    ].copy()
    
    print(f"  Found {len(qualified_openers)} qualified explosive openers.")
    
    # Radar Chart for Top 5 Candidates
    if not qualified_openers.empty:
        top_5 = qualified_openers.sort_values('SR_powerplay', ascending=False).head(5)
        
        # Normalize for Radar Chart
        radar_data = []
        for name, row in top_5.iterrows():
            radar_data.append(dict(Player=name, Metric='PP SR', Value=row['SR_powerplay']))
            radar_data.append(dict(Player=name, Metric='Pace SR', Value=row['SR_pace']))
            radar_data.append(dict(Player=name, Metric='Spin SR', Value=row['SR_spin']))
            radar_data.append(dict(Player=name, Metric='Avg', Value=row['Avg']))
            
        radar_df = pd.DataFrame(radar_data)
        
        fig_radar = px.line_polar(radar_df, r='Value', theta='Metric', line_close=True, 
                                  color='Player', title='Top Explosive Opener Candidates')
        save_plot(fig_radar, output_dir, 'solution_openers_radar')
    
    # Scatter Plot (Context)
    fig = px.scatter(overseas_batters[overseas_batters['Balls_powerplay'] >= 60], 
                     x='RAA_powerplay', y='SR_powerplay', 
                     size='Balls_powerplay', hover_name=overseas_batters[overseas_batters['Balls_powerplay'] >= 60].index,
                     color='Status',
                     title='Overseas Opener Targets (Auction vs Retained)',
                     labels={'RAA_powerplay': 'Powerplay RAA', 'SR_powerplay': 'Powerplay Strike Rate'},
                     color_discrete_map={'Auction': 'blue', 'Retained': 'red'},
                     hover_data=['Status', 'WAR', 'SR_spin', 'Avg', 'WAR_2026', 'Est_Price'])
    
    # Highlight qualified ones
    if not qualified_openers.empty:
        fig.add_trace(px.scatter(qualified_openers, x='RAA_powerplay', y='SR_powerplay').data[0])
        fig.data[-1].marker.symbol = 'star'
        fig.data[-1].marker.size = 12
        fig.data[-1].marker.color = 'gold'
        fig.data[-1].name = 'Qualified Targets'
        
    save_plot(fig, output_dir, 'solution_openers')
    
    # Save Top Auction Targets
    top_auction = qualified_openers[qualified_openers['Status'] == 'Auction'].sort_values('SR_powerplay', ascending=False).head(10)
    top_auction[['SR_powerplay', 'RAA_powerplay', 'Balls_powerplay', 'WAR', 'SR_spin', 'Avg']].to_csv(output_dir / 'target_openers.csv')

    # --- Module 9: All Opener Targets (NEW) ---
    print("  Module 9: All Opener Targets")
    # Filter: Min 60 balls in PP (All Nationalities)
    qualified_all_openers = all_batters[all_batters['Balls_powerplay'] >= 60].copy()
    
    fig = px.scatter(qualified_all_openers, x='RAA_powerplay', y='SR_powerplay', 
                     size='Balls_powerplay', hover_name=qualified_all_openers.index,
                     color='Status',
                     title='All Opener Targets (Indian & Overseas)',
                     labels={'RAA_powerplay': 'Powerplay RAA', 'SR_powerplay': 'Powerplay Strike Rate'},
                     color_discrete_map={'Auction': 'blue', 'Retained': 'red'},
                     hover_data=['Status', 'Country', 'WAR', 'WAR_2026', 'Est_Price'])
    save_plot(fig, output_dir, 'solution_openers_all')
    
    top_all_auction = qualified_all_openers[qualified_all_openers['Status'] == 'Auction'].sort_values('SR_powerplay', ascending=False).head(10)
    top_all_auction[['SR_powerplay', 'RAA_powerplay', 'Balls_powerplay', 'WAR']].to_csv(output_dir / 'target_openers_all.csv')

    # --- Module 8: Middle Order Targets (NEW) ---
    print("  Module 8: Middle Order Targets")
    # Filter: Min 60 balls in Middle Overs (7-15)
    qualified_middle = all_batters[all_batters['Balls_middle'] >= 60].copy()
    
    fig = px.scatter(qualified_middle, x='RAA_middle', y='SR_middle', 
                     size='Balls_middle', hover_name=qualified_middle.index,
                     color='Status',
                     title='Middle Order Targets (Auction vs Retained)',
                     labels={'RAA_middle': 'Middle Overs RAA', 'SR_middle': 'Middle Overs Strike Rate'},
                     color_discrete_map={'Auction': 'blue', 'Retained': 'red'},
                     hover_data=['Status', 'WAR', 'WAR_2026', 'Est_Price'])
    save_plot(fig, output_dir, 'solution_middle_order')
    
    top_middle = qualified_middle[qualified_middle['Status'] == 'Auction'].sort_values('RAA_middle', ascending=False).head(10)
    top_middle[['SR_middle', 'RAA_middle', 'Balls_middle', 'WAR']].to_csv(output_dir / 'target_middle_order.csv')

    # --- Prepare Bowlers (All + Status) ---
    all_bowlers['name_norm'] = all_bowlers.index.map(norm)
    pool_bowlers_norm = set(pool_bowlers.index.map(norm))
    all_bowlers['Status'] = all_bowlers['name_norm'].apply(lambda x: 'Auction' if x in pool_bowlers_norm else 'Retained')
    
    # --- Add Valuation Data to Bowlers ---
    print("  Adding Valuation Data to Bowlers...")
    all_bowlers = add_valuation_data(all_bowlers, engine)
    
    # --- Module 5: Death Bowler Hunt ---
    print("  Module 5: Death Bowler Hunt")
    
    # Calculate Dot % and Death Econ from raw data for accuracy
    # Filter for 2024-2025 (Last 2 seasons) as requested
    df_recent = ipl_df[ipl_df['season'].isin(['2024', '2025'])]
    death_recent = df_recent[df_recent['over'] >= 16]
    
    death_stats_recent = death_recent.groupby('bowler_name').agg({
        'total_runs': 'sum',
        'ball_in_over': 'count',
        'is_wicket': 'sum',
        'bowler_RAA': 'sum'
    }).rename(columns={'ball_in_over': 'Balls', 'total_runs': 'Runs', 'is_wicket': 'Wickets', 'bowler_RAA': 'RAA'})
    
    # Calculate Dot Balls (runs_off_bat == 0 AND extras == 0 usually, or just total_runs == 0)
    # Strict definition: total_runs == 0
    dots = death_recent[death_recent['total_runs'] == 0].groupby('bowler_name')['ball_in_over'].count()
    death_stats_recent['Dots'] = dots
    death_stats_recent['Dots'] = death_stats_recent['Dots'].fillna(0)
    
    death_stats_recent['Econ'] = (death_stats_recent['Runs'] / death_stats_recent['Balls']) * 6
    death_stats_recent['Dot_Pct'] = (death_stats_recent['Dots'] / death_stats_recent['Balls']) * 100
    
    # Filter: Min 60 balls
    qualified_death = death_stats_recent[death_stats_recent['Balls'] >= 60].copy()
    
    # Add Valuation Data to Death Stats (Need to map from all_bowlers or re-calculate)
    # Since death_stats_recent index is bowler_name, we can use add_valuation_data directly
    qualified_death = add_valuation_data(qualified_death, engine)
    
    # League Average Death Economy (for filter)
    league_avg_death_econ = (death_recent['total_runs'].sum() / death_recent['ball_in_over'].count()) * 6
    # Relaxed threshold: Just beat the league average (was -0.5)
    econ_threshold = league_avg_death_econ 
    print(f"  League Avg Death Econ: {league_avg_death_econ:.2f}, Threshold: {econ_threshold:.2f}")
    
    # Apply Filters:
    # 1. Econ < Threshold
    # 2. Dot % > 25% (Relaxed from 30%)
    
    targets = qualified_death[
        (qualified_death['Econ'] < econ_threshold) &
        (qualified_death['Dot_Pct'] > 25)
    ].copy()
    
    # Add Status
    targets['name_norm'] = targets.index.map(norm)
    targets['Status'] = targets['name_norm'].apply(lambda x: 'Auction' if x in pool_bowlers_norm else 'Retained')
    
    print(f"  Found {len(targets)} qualified death bowlers.")
    
    # Bubble Chart
    # X: Economy, Y: Dot %, Size: Wickets
    fig = px.scatter(targets, x='Econ', y='Dot_Pct', size='Wickets',
                     hover_name=targets.index, color='Status',
                     title='Death Bowler Targets (High Dots, Low Economy)',
                     labels={'Econ': 'Death Economy', 'Dot_Pct': 'Dot Ball %'},
                     color_discrete_map={'Auction': 'blue', 'Retained': 'red'},
                     hover_data=['Status', 'RAA', 'WAR_2026', 'Est_Price', 'Est_Price_VOMAM', 'Est_Price_VOPE'])
    fig.update_xaxes(autorange="reversed") # Low Econ is good
    
    # Add quadrant lines
    fig.add_vline(x=econ_threshold, line_dash="dash", annotation_text="Econ Threshold")
    fig.add_hline(y=25, line_dash="dash", annotation_text="Dot % Threshold")
    
    save_plot(fig, output_dir, 'solution_death_bowlers')
    
    top_death = targets[targets['Status'] == 'Auction'].sort_values('RAA', ascending=False).head(10)
    top_death.to_csv(output_dir / 'target_death_bowlers.csv')

    # --- Module 7: Powerplay Pacer Hunt ---
    print("  Module 7: Powerplay Pacer Hunt")
    # Filter for Pacers (using metadata bowling_style)
    # The 'bowling_type' in IPL data is unreliable (classifies spinners as pace).
    # We use metadata 'bowling_style' to exclude spinners.
    
    pp_bowlers = all_bowlers[all_bowlers['Balls_powerplay'] >= 60].copy()
    
    # Get bowling style from metadata
    meta_unique = engine.metadata.drop_duplicates(subset='name_norm')
    style_map = meta_unique.set_index('name_norm')['bowling_style'].to_dict()
    
    def is_pacer(name):
        style = str(style_map.get(norm(name), '')).lower()
        if 'spin' in style or 'break' in style or 'slow' in style:
            return False
        return True
        
    pp_bowlers['Is_Pacer'] = pp_bowlers.index.map(is_pacer)
    pp_pacers = pp_bowlers[pp_bowlers['Is_Pacer'] == True].copy()
    
    fig = px.scatter(pp_pacers, x='Wickets_powerplay', y='RAA_powerplay', size='Balls_powerplay',
                     hover_name=pp_pacers.index, color='Status',
                     title='Powerplay Pacer Targets (Auction vs Retained)',
                     labels={'Wickets_powerplay': 'PP Wickets', 'RAA_powerplay': 'PP RAA'},
                     color_discrete_map={'Auction': 'blue', 'Retained': 'red'},
                     hover_data=['Status', 'WAR', 'WAR_2026', 'Est_Price', 'Est_Price_VOMAM', 'Est_Price_VOPE'])
    save_plot(fig, output_dir, 'solution_pp_pacers')
    
    # Sort by Wickets then RAA (descending) to favor better RAA for same wickets
    top_pp = pp_pacers[pp_pacers['Status'] == 'Auction'].sort_values(['Wickets_powerplay', 'RAA_powerplay'], ascending=[False, False]).head(15)
    top_pp[['Wickets_powerplay', 'RAA_powerplay', 'Balls_powerplay', 'WAR_2026', 'Est_Price', 'Est_Price_VOMAM', 'Est_Price_VOPE']].to_csv(output_dir / 'target_pp_pacers.csv')

def main():
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / 'results' / 'analysis' / 'strategic'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Valuation Engine
    engine = ValuationEngine(project_root)
    engine.load_data()
    engine.train_models()
    
    ipl_df, pool_batters, pool_bowlers, all_batters, all_bowlers, metadata = load_data(project_root)
    
    phase1_diagnosis(ipl_df, output_dir)
    phase2_solution(ipl_df, pool_batters, pool_bowlers, all_batters, all_bowlers, metadata, output_dir, engine)
    
    print(f"Analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
