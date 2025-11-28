import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
from pathlib import Path

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

    # --- Module 3: Death Bowling ---
    print("  Module 3: Death Bowling Crisis")
    death_bowling = df_2025[df_2025['over'] >= 16]
    death_stats = death_bowling.groupby('bowling_team').agg({'total_runs': 'sum', 'ball_in_over': 'count'}).reset_index()
    death_stats['Death_Econ'] = (death_stats['total_runs'] / death_stats['ball_in_over']) * 6
    death_stats = death_stats.sort_values('Death_Econ', ascending=True)
    
    fig = px.bar(death_stats, y='bowling_team', x='Death_Econ', orientation='h',
                 title='Death Overs Economy Rate (2025)', text_auto='.2f',
                 color='bowling_team', color_discrete_map={'Delhi Capitals': 'red'})
    fig.add_vline(x=death_stats['Death_Econ'].mean(), line_dash="dash", annotation_text="League Avg")
    save_plot(fig, output_dir, 'diagnosis_death_bowling')

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

def phase2_solution(ipl_df, pool_batters, pool_bowlers, all_batters, all_bowlers, metadata, output_dir):
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
    
    # --- Module 4: Explosive Opener Search ---
    print("  Module 4: Explosive Opener Search")
    # Filter: Overseas & Min Balls
    overseas_batters = all_batters[all_batters['Country'] != 'India'].copy()
    qualified_openers = overseas_batters[overseas_batters['Balls_powerplay'] >= 60].copy()
    
    fig = px.scatter(qualified_openers, x='RAA_powerplay', y='SR_powerplay', 
                     size='Balls_powerplay', hover_name=qualified_openers.index,
                     color='Status', # Color by Status
                     title='Overseas Opener Targets (Auction vs Retained)',
                     labels={'RAA_powerplay': 'Powerplay RAA', 'SR_powerplay': 'Powerplay Strike Rate'},
                     color_discrete_map={'Auction': 'blue', 'Retained': 'red'},
                     hover_data=['Status', 'WAR'])
    save_plot(fig, output_dir, 'solution_openers')
    
    # Save Top Auction Targets
    top_auction = qualified_openers[qualified_openers['Status'] == 'Auction'].sort_values('SR_powerplay', ascending=False).head(10)
    top_auction[['SR_powerplay', 'RAA_powerplay', 'Balls_powerplay', 'WAR']].to_csv(output_dir / 'target_openers.csv')

    # --- Prepare Bowlers (All + Status) ---
    all_bowlers['name_norm'] = all_bowlers.index.map(norm)
    pool_bowlers_norm = set(pool_bowlers.index.map(norm))
    all_bowlers['Status'] = all_bowlers['name_norm'].apply(lambda x: 'Auction' if x in pool_bowlers_norm else 'Retained')
    
    # --- Module 5: Death Bowler Hunt ---
    print("  Module 5: Death Bowler Hunt")
    # Using 2022-2025 stats from all_bowlers (which is aggregated)
    # Note: all_bowlers has 'Runs_death', 'Balls_death' etc.
    
    death_bowlers = all_bowlers[all_bowlers['Balls_death'] >= 60].copy()
    death_bowlers['Econ_Death'] = (death_bowlers['Runs_death'] / death_bowlers['Balls_death']) * 6
    # Dot % might not be in all_bowlers, let's calculate if possible or skip.
    # all_bowlers from generate_auction_stats doesn't have Dot %. 
    # We can calculate it from ipl_df if needed, but for now let's use Econ and RAA.
    
    fig = px.scatter(death_bowlers, x='Econ_Death', y='RAA_death', size='Wickets_death',
                     hover_name=death_bowlers.index, color='Status',
                     title='Death Bowler Targets (Auction vs Retained)',
                     labels={'Econ_Death': 'Death Economy', 'RAA_death': 'Death RAA'},
                     color_discrete_map={'Auction': 'blue', 'Retained': 'red'},
                     hover_data=['Status', 'WAR'])
    fig.update_xaxes(autorange="reversed") # Low Econ is good
    save_plot(fig, output_dir, 'solution_death_bowlers')
    
    top_death = death_bowlers[death_bowlers['Status'] == 'Auction'].sort_values('RAA_death', ascending=False).head(10)
    top_death[['RAA_death', 'Econ_Death', 'Wickets_death']].to_csv(output_dir / 'target_death_bowlers.csv')

    # --- Module 7: Powerplay Pacer Hunt ---
    print("  Module 7: Powerplay Pacer Hunt")
    # Filter for Pacers (using metadata or inference)
    # We can infer from ipl_df but that's slow. Let's use metadata if available or just plot all.
    # Let's use ipl_df to get bowling type map again quickly.
    bowler_types = ipl_df.groupby('bowler_name')['bowling_type'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'pace')
    
    pp_bowlers = all_bowlers[all_bowlers['Balls_powerplay'] >= 60].copy()
    pp_bowlers['Type'] = pp_bowlers.index.map(bowler_types)
    pp_pacers = pp_bowlers[pp_bowlers['Type'] == 'pace'].copy()
    
    fig = px.scatter(pp_pacers, x='Wickets_powerplay', y='RAA_powerplay', size='Balls_powerplay',
                     hover_name=pp_pacers.index, color='Status',
                     title='Powerplay Pacer Targets (Auction vs Retained)',
                     labels={'Wickets_powerplay': 'PP Wickets', 'RAA_powerplay': 'PP RAA'},
                     color_discrete_map={'Auction': 'blue', 'Retained': 'red'},
                     hover_data=['Status', 'WAR'])
    save_plot(fig, output_dir, 'solution_pp_pacers')
    
    top_pp = pp_pacers[pp_pacers['Status'] == 'Auction'].sort_values('Wickets_powerplay', ascending=False).head(10)
    top_pp[['Wickets_powerplay', 'RAA_powerplay', 'Balls_powerplay']].to_csv(output_dir / 'target_pp_pacers.csv')

def main():
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / 'results' / 'analysis' / 'strategic'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ipl_df, pool_batters, pool_bowlers, all_batters, all_bowlers, metadata = load_data(project_root)
    
    phase1_diagnosis(ipl_df, output_dir)
    phase2_solution(ipl_df, pool_batters, pool_bowlers, all_batters, all_bowlers, metadata, output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
