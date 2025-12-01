import pandas as pd
import plotly.express as px
import plotly.io as pio
from pathlib import Path
from valuation_engine import ValuationEngine, normalize_name

# Set default template
pio.templates.default = "plotly_white"

def main():
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / 'results' / 'analysis' / 'strategic' / 'valuation_context'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading Data...")
    # 1. Load Projections via Valuation Engine
    engine = ValuationEngine(project_root)
    engine.load_data()
    projections = engine.projections.copy()
    
    # 2. Load Retentions
    retentions = pd.read_csv(project_root / 'data' / 'ipl_2026_retentions.csv')
    retention_names = set(retentions['Player'].dropna().apply(normalize_name))
    
    # 3. Add Status (Retained vs Auction)
    projections['Status'] = projections['name_norm'].apply(lambda x: 'Retained' if x in retention_names else 'Auction')
    
    # 4. Filter for relevant players (WAR > -1 to remove noise, or keep all?)
    # Let's keep top 300 to focus on relevant pool
    df = projections.sort_values('war_2026', ascending=False).head(300).copy()
    
    # 5. Add Rank
    df['Rank'] = range(1, len(df) + 1)
    df['Percentile'] = (1 - (df['Rank'] / len(df))) * 100
    
    print(f"Top 5 Players by Projected WAR 2026:")
    print(df[['player_name', 'war_2026', 'Status']].head(5))
    
    # --- Visualization 1: Scatter Plot (Rank vs WAR) ---
    fig = px.scatter(df, x='Rank', y='war_2026', color='Status',
                     hover_name='player_name', title='Projected WAR 2026: League Hierarchy',
                     labels={'war_2026': 'Projected WAR (2026)', 'Rank': 'League Rank'},
                     color_discrete_map={'Retained': 'red', 'Auction': 'blue'})
    
    # Highlight Specific Targets
    targets = ['Jake Fraser-McGurk', 'Glenn Maxwell', 'Josh Inglis', 'Liam Livingstone', 'Andre Russell', 'Matheesha Pathirana']
    benchmarks = ['Virat Kohli', 'Jasprit Bumrah', 'Heinrich Klaasen', 'Rishabh Pant']
    
    # Add annotations for Targets
    for name in targets:
        row = df[df['name_norm'] == normalize_name(name)]
        if not row.empty:
            fig.add_annotation(x=row['Rank'].values[0], y=row['war_2026'].values[0],
                               text=name, showarrow=True, arrowhead=1, ax=40, ay=-40,
                               font=dict(color='blue', size=10))
            print(f"Target: {name} | Rank: {row['Rank'].values[0]} | WAR: {row['war_2026'].values[0]:.2f}")

    # Add annotations for Benchmarks
    for name in benchmarks:
        row = df[df['name_norm'] == normalize_name(name)]
        if not row.empty:
            fig.add_annotation(x=row['Rank'].values[0], y=row['war_2026'].values[0],
                               text=name, showarrow=True, arrowhead=1, ax=-40, ay=-40,
                               font=dict(color='red', size=10))

    fig.write_html(output_dir / 'war_rankings.html')
    try:
        fig.write_image(output_dir / 'war_rankings.png', scale=2)
    except:
        pass
    print(f"Saved plot to {output_dir / 'war_rankings.html'}")

    # --- Visualization 2: Histogram ---
    fig2 = px.histogram(df, x='war_2026', color='Status', nbins=30,
                        title='Distribution of Projected WAR (Top 300)',
                        barmode='overlay', opacity=0.7,
                        color_discrete_map={'Retained': 'red', 'Auction': 'blue'})
    fig2.write_html(output_dir / 'war_distribution.html')
    try:
        fig2.write_image(output_dir / 'war_distribution.png', scale=2)
    except:
        pass
        
if __name__ == "__main__":
    main()
