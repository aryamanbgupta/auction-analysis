"""
Interactive Visualization: Base Price vs Projected WAR for IPL 2026 Auction.

Creates a Plotly scatter plot showing each player's base price on X-axis
and projected WAR on Y-axis, with color coding by prediction source.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


def create_visualization():
    """Create interactive scatter plot of WAR vs Base Price."""
    # Load data
    project_root = Path(__file__).parent.parent
    results = pd.read_csv(
        project_root / 'results' / 'WARprojections' / 'auction_2026_v9prod' / 'auction_war_projections_v9prod.csv'
    )
    
    # Clean and prepare data
    results = results[results['projected_war_2026'] > 0].copy()  # Remove replacement level
    results['base_price'] = pd.to_numeric(results['base_price'], errors='coerce')
    results = results.dropna(subset=['base_price', 'projected_war_2026'])
    
    # Add value metric (WAR per Crore at base price)
    results['value_ratio'] = results['projected_war_2026'] / (results['base_price'] / 100)
    
    # Clean role for display
    results['role_clean'] = results['role'].fillna('Unknown').replace('NaN', 'Unknown')
    
    # Create color mapping by source
    color_map = {
        'V9_Production': '#2ecc71',  # Green - best model
        'Marcel': '#3498db',  # Blue
        'V8_Domestic': '#e74c3c',  # Red
        'Global_Only': '#9b59b6',  # Purple
    }
    
    # Create the figure
    fig = px.scatter(
        results,
        x='base_price',
        y='projected_war_2026',
        color='prediction_source',
        hover_name='player',
        hover_data={
            'country': True,
            'role_clean': True,
            'base_price': ':.0f',
            'projected_war_2026': ':.2f',
            'value_ratio': ':.2f',
            'prediction_source': True
        },
        color_discrete_map=color_map,
        title='IPL 2026 Auction Pool: Base Price vs Projected WAR',
        labels={
            'base_price': 'Base Price (₹ Lakh)',
            'projected_war_2026': 'Projected WAR (2026)',
            'prediction_source': 'Model Source',
            'role_clean': 'Role',
            'value_ratio': 'WAR per Crore'
        }
    )
    
    # Update layout
    fig.update_layout(
        font=dict(family='Inter, sans-serif', size=12),
        title=dict(
            font=dict(size=20, color='#2c3e50'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title_font=dict(size=14),
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='rgba(0,0,0,0.3)'
        ),
        yaxis=dict(
            title_font=dict(size=14),
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='rgba(0,0,0,0.3)'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            title=None
        ),
        hovermode='closest',
        margin=dict(l=60, r=40, t=100, b=60)
    )
    
    # Add annotations for top players
    top_players = results.nlargest(5, 'projected_war_2026')
    for _, row in top_players.iterrows():
        fig.add_annotation(
            x=row['base_price'],
            y=row['projected_war_2026'],
            text=row['player'].split()[-1],  # Last name only
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor='#34495e',
            ax=30,
            ay=-30,
            font=dict(size=10, color='#34495e')
        )
    
    # Update marker styling
    fig.update_traces(
        marker=dict(size=10, opacity=0.7, line=dict(width=1, color='white')),
    )
    
    # Save
    output_dir = project_root / 'results' / 'WARprojections' / 'auction_2026_v9prod'
    fig.write_html(output_dir / 'auction_war_visualization.html')
    
    print(f"✓ Interactive visualization saved to:")
    print(f"  {output_dir / 'auction_war_visualization.html'}")
    
    # Also show summary stats
    print(f"\n=== Summary ===")
    print(f"Players shown: {len(results)}")
    print(f"Base price range: ₹{results['base_price'].min():.0f}L - ₹{results['base_price'].max():.0f}L")
    print(f"WAR range: {results['projected_war_2026'].min():.2f} - {results['projected_war_2026'].max():.2f}")
    
    # Best value players (high WAR per price)
    print(f"\n=== Top 10 Value Players (WAR per Crore at Base) ===")
    top_value = results.nlargest(10, 'value_ratio')[['player', 'base_price', 'projected_war_2026', 'value_ratio']]
    print(top_value.to_string(index=False))
    
    return fig


if __name__ == "__main__":
    create_visualization()
