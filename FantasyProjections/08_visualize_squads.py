"""
Interactive visualization of IPL 2026 squad fantasy point projections.

Charts:
1. Team-by-team horizontal bar charts (25 players each)
2. Team strength comparison (avg + total projected FP)
3. League-wide top 30 players
4. Value analysis (FP per crore spent)

OUTPUT: results/FantasyProjections/squad_2026/squad_fantasy_visualization.html
"""

from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


# Team colors matching IPL branding
TEAM_COLORS = {
    'CSK': '#FFCB05',
    'DC': '#004C93',
    'GT': '#1B2133',
    'KKR': '#3A225D',
    'LSG': '#A72056',
    'MI': '#004BA0',
    'PBKS': '#DD1F2D',
    'RR': '#EA1A85',
    'RCB': '#EC1C24',
    'SRH': '#FF822A',
}

SOURCE_COLORS = {
    'Production': '#2ecc71',
    'Marcel': '#f39c12',
    'Replacement': '#95a5a6',
}


def main():
    project_root = Path(__file__).parent.parent
    output_dir = project_root / 'results' / 'FantasyProjections' / 'squad_2026'
    df = pd.read_csv(output_dir / 'squad_fantasy_projections_2026.csv')

    print(f"Loaded {len(df)} players from {df['Team'].nunique()} teams")

    # ── 1. Team Strength Comparison ───────────────────────────────────
    team_stats = df.groupby('Team').agg(
        avg_fp=('projected_avg_fp_2026', 'mean'),
        total_fp=('projected_avg_fp_2026', 'sum'),
        max_fp=('projected_avg_fp_2026', 'max'),
        total_spend=('Price_Cr', 'sum'),
        n_players=('Player', 'count'),
    ).sort_values('avg_fp', ascending=True)

    fig_team = go.Figure()
    fig_team.add_trace(go.Bar(
        y=team_stats.index,
        x=team_stats['avg_fp'],
        orientation='h',
        marker_color=[TEAM_COLORS.get(t, '#666') for t in team_stats.index],
        text=[f"{v:.1f}" for v in team_stats['avg_fp']],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Avg FP: %{x:.1f}<br>Total FP: %{customdata[0]:.0f}<br>Best: %{customdata[1]:.1f}<br>Spend: ₹%{customdata[2]:.1f}Cr<extra></extra>',
        customdata=team_stats[['total_fp', 'max_fp', 'total_spend']].values,
    ))
    fig_team.update_layout(
        title='IPL 2026 Team Strength — Avg Projected Fantasy Pts/Match',
        xaxis_title='Avg Projected Fantasy Pts/Match',
        yaxis_title='',
        height=500,
        template='plotly_white',
        margin=dict(l=60, r=40, t=60, b=40),
    )

    # ── 2. Top 30 Players ────────────────────────────────────────────
    top30 = df.nlargest(30, 'projected_avg_fp_2026').iloc[::-1]  # reverse for bottom-up bar
    fig_top = go.Figure()
    fig_top.add_trace(go.Bar(
        y=[f"{r['Player']} ({r['Team']})" for _, r in top30.iterrows()],
        x=top30['projected_avg_fp_2026'],
        orientation='h',
        marker_color=[TEAM_COLORS.get(r['Team'], '#666') for _, r in top30.iterrows()],
        text=[f"{v:.1f}" for v in top30['projected_avg_fp_2026']],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>FP: %{x:.1f}<br>Price: ₹%{customdata[0]:.1f}Cr<br>Source: %{customdata[1]}<extra></extra>',
        customdata=top30[['Price_Cr', 'prediction_source']].values,
    ))
    fig_top.update_layout(
        title='Top 30 IPL 2026 Players — Projected Fantasy Pts/Match',
        xaxis_title='Projected Avg Fantasy Pts/Match',
        height=800,
        template='plotly_white',
        margin=dict(l=200, r=40, t=60, b=40),
    )

    # ── 3. Team-by-Team Breakdown ────────────────────────────────────
    teams_sorted = df.groupby('Team')['projected_avg_fp_2026'].mean().sort_values(ascending=False).index.tolist()
    fig_teams = make_subplots(
        rows=5, cols=2,
        subplot_titles=teams_sorted,
        horizontal_spacing=0.12,
        vertical_spacing=0.04,
    )
    for i, team in enumerate(teams_sorted):
        row, col = i // 2 + 1, i % 2 + 1
        team_df = df[df['Team'] == team].sort_values('projected_avg_fp_2026', ascending=True)
        colors = [SOURCE_COLORS.get(s, '#666') for s in team_df['prediction_source']]
        fig_teams.add_trace(
            go.Bar(
                y=team_df['Player'],
                x=team_df['projected_avg_fp_2026'],
                orientation='h',
                marker_color=colors,
                text=[f"{v:.0f}" for v in team_df['projected_avg_fp_2026']],
                textposition='outside',
                textfont=dict(size=9),
                hovertemplate='<b>%{y}</b><br>FP: %{x:.1f}<br>₹%{customdata[0]:.1f}Cr<br>%{customdata[1]}<br>%{customdata[2]}<extra></extra>',
                customdata=team_df[['Price_Cr', 'prediction_source', 'playing_role']].values,
                showlegend=False,
            ),
            row=row, col=col,
        )
        fig_teams.update_xaxes(range=[0, 75], row=row, col=col)
        fig_teams.update_yaxes(tickfont=dict(size=8), row=row, col=col)

    fig_teams.update_layout(
        title='IPL 2026 Squad Projections by Team (Green=Production, Orange=Marcel, Grey=Replacement)',
        height=3000,
        template='plotly_white',
        margin=dict(l=150, r=30, t=80, b=30),
    )

    # ── 4. Value Analysis (FP per Crore) ─────────────────────────────
    df_val = df[df['Price_Cr'] > 0].copy()
    df_val['fp_per_cr'] = df_val['projected_avg_fp_2026'] / df_val['Price_Cr']
    top_value = df_val.nlargest(30, 'fp_per_cr').iloc[::-1]

    fig_value = go.Figure()
    fig_value.add_trace(go.Bar(
        y=[f"{r['Player']} ({r['Team']})" for _, r in top_value.iterrows()],
        x=top_value['fp_per_cr'],
        orientation='h',
        marker_color=[TEAM_COLORS.get(r['Team'], '#666') for _, r in top_value.iterrows()],
        text=[f"{v:.1f}" for v in top_value['fp_per_cr']],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>FP/Cr: %{x:.1f}<br>FP: %{customdata[0]:.1f}<br>Price: ₹%{customdata[1]:.1f}Cr<extra></extra>',
        customdata=top_value[['projected_avg_fp_2026', 'Price_Cr']].values,
    ))
    fig_value.update_layout(
        title='Top 30 Value Picks — Fantasy Pts per Crore Spent',
        xaxis_title='Projected FP / Price (Cr)',
        height=800,
        template='plotly_white',
        margin=dict(l=200, r=40, t=60, b=40),
    )

    # ── 5. Scatter: FP vs Price ───────────────────────────────────────
    df_val['size_col'] = df_val['projected_avg_fp_2026'].clip(lower=1)
    fig_scatter = px.scatter(
        df_val,
        x='Price_Cr',
        y='projected_avg_fp_2026',
        color='Team',
        color_discrete_map=TEAM_COLORS,
        hover_name='Player',
        hover_data={'prediction_source': True, 'playing_role': True, 'Price_Cr': ':.1f'},
        size='size_col',
        size_max=15,
        template='plotly_white',
        title='Fantasy Points vs Price — All Squad Players',
        labels={'projected_avg_fp_2026': 'Projected Avg FP/Match', 'Price_Cr': 'Price (₹ Cr)'},
    )
    fig_scatter.update_layout(height=600)

    # ── Combine into single HTML ──────────────────────────────────────
    output_path = output_dir / 'squad_fantasy_visualization.html'
    with open(output_path, 'w') as f:
        f.write('<!DOCTYPE html><html><head><meta charset="utf-8">')
        f.write('<title>IPL 2026 Fantasy Projections</title>')
        f.write('<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>')
        f.write('<style>body{font-family:Arial,sans-serif;max-width:1200px;margin:0 auto;padding:20px;background:#fafafa}'
                '.chart{margin-bottom:40px;background:white;padding:20px;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.1)}'
                'h1{text-align:center;color:#333}p.subtitle{text-align:center;color:#666}</style>')
        f.write('</head><body>')
        f.write('<h1>IPL 2026 Squad Fantasy Point Projections</h1>')
        f.write('<p class="subtitle">Dream11 scoring system &mdash; Production model + Marcel fallback</p>')

        for i, (fig, title) in enumerate([
            (fig_team, 'Team Strength'),
            (fig_top, 'Top 30 Players'),
            (fig_scatter, 'FP vs Price'),
            (fig_value, 'Best Value Picks'),
            (fig_teams, 'Team Rosters'),
        ]):
            f.write(f'<div class="chart" id="chart{i}"></div>')

        f.write('<script>')
        for i, (fig, _) in enumerate([
            (fig_team, ''), (fig_top, ''), (fig_scatter, ''),
            (fig_value, ''), (fig_teams, ''),
        ]):
            f.write(f'Plotly.newPlot("chart{i}",{fig.to_json()}.data,{fig.to_json()}.layout);')
        f.write('</script></body></html>')

    # More efficient: write using plotly's to_html with include_plotlyjs once
    with open(output_path, 'w') as f:
        f.write('<!DOCTYPE html><html><head><meta charset="utf-8">\n')
        f.write('<title>IPL 2026 Fantasy Projections</title>\n')
        f.write('<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>\n')
        f.write('<style>body{font-family:Arial,sans-serif;max-width:1200px;margin:0 auto;padding:20px;background:#fafafa}\n'
                '.chart{margin-bottom:40px;background:white;padding:20px;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.1)}\n'
                'h1{text-align:center;color:#333}p.subtitle{text-align:center;color:#666}</style>\n')
        f.write('</head><body>\n')
        f.write('<h1>IPL 2026 Squad Fantasy Point Projections</h1>\n')
        f.write('<p class="subtitle">Dream11 scoring &mdash; Ensemble model (XGBoost + RF + Marcel)</p>\n')

        figs = [
            (fig_team, 'Team Strength Comparison'),
            (fig_top, 'Top 30 Players League-Wide'),
            (fig_scatter, 'Fantasy Points vs Price'),
            (fig_value, 'Best Value Picks (FP per Crore)'),
            (fig_teams, 'Full Squad Breakdowns by Team'),
        ]
        for fig, title in figs:
            f.write(fig.to_html(full_html=False, include_plotlyjs=False))
            f.write('\n<hr>\n')

        f.write('</body></html>')

    print(f"\n✓ Saved visualization to {output_path}")
    print(f"  Open in browser: file://{output_path.resolve()}")


if __name__ == '__main__':
    main()
