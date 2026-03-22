"""
Interactive visualization of IPL 2026 squad fantasy point projections.

Sections:
1. Team strength comparison bar chart
2. Top 30 players bar chart
3. FP vs Price scatter
4. Value picks bar chart
5. Full sortable/filterable player table (250 players) with role/team/source filters
6. Team-by-team squad tables (25 players each)
7. Team-by-team bar charts

OUTPUT: results/FantasyProjections/squad_2026/squad_fantasy_visualization.html
"""

from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import json


TEAM_COLORS = {
    'CSK': '#FFCB05', 'DC': '#004C93', 'GT': '#1B2133', 'KKR': '#3A225D',
    'LSG': '#A72056', 'MI': '#004BA0', 'PBKS': '#DD1F2D', 'RR': '#EA1A85',
    'RCB': '#EC1C24', 'SRH': '#FF822A',
}

TEAM_TEXT_COLORS = {
    'CSK': '#000', 'DC': '#fff', 'GT': '#fff', 'KKR': '#fff',
    'LSG': '#fff', 'MI': '#fff', 'PBKS': '#fff', 'RR': '#fff',
    'RCB': '#fff', 'SRH': '#fff',
}

SOURCE_COLORS = {
    'Production': '#2ecc71', 'Marcel': '#f39c12', 'Replacement': '#95a5a6',
}


def build_full_table_html(df):
    """Build a filterable/sortable HTML table of all 250 players."""
    teams = sorted(df['Team'].unique())
    roles = sorted([r for r in df['playing_role'].dropna().unique() if r not in ('Unknown', 'Error', '')])
    sources = sorted(df['prediction_source'].unique())

    # Clean role display
    df = df.copy()
    df['playing_role'] = df['playing_role'].fillna('Unknown').replace({'Error': 'Unknown', '': 'Unknown'})

    html = '''
    <div class="section" id="full-table-section">
    <h2>All 250 Players — Sortable &amp; Filterable</h2>
    <div class="filters">
        <label>Team: <select id="filter-team" onchange="filterTable()">
            <option value="">All Teams</option>
    '''
    for t in teams:
        html += f'        <option value="{t}">{t}</option>\n'
    html += '''    </select></label>
        <label>Role: <select id="filter-role" onchange="filterTable()">
            <option value="">All Roles</option>
    '''
    for r in roles:
        html += f'        <option value="{r}">{r}</option>\n'
    html += '''    </select></label>
        <label>Source: <select id="filter-source" onchange="filterTable()">
            <option value="">All Sources</option>
    '''
    for s in sources:
        html += f'        <option value="{s}">{s}</option>\n'
    html += '''    </select></label>
        <label>Search: <input type="text" id="filter-search" oninput="filterTable()" placeholder="Player name..."></label>
        <button onclick="resetFilters()">Reset</button>
        <span id="row-count" style="margin-left:15px;color:#666;"></span>
    </div>
    <table id="player-table">
    <thead>
        <tr>
            <th onclick="sortTable(0)" class="sortable">#</th>
            <th onclick="sortTable(1)" class="sortable">Player</th>
            <th onclick="sortTable(2)" class="sortable">Team</th>
            <th onclick="sortTable(3,'num')" class="sortable">Projected FP</th>
            <th onclick="sortTable(4,'num')" class="sortable">Price (Cr)</th>
            <th onclick="sortTable(5)" class="sortable">Role</th>
            <th onclick="sortTable(6)" class="sortable">Source</th>
            <th onclick="sortTable(7)" class="sortable">Acquisition</th>
            <th onclick="sortTable(8)" class="sortable">Type</th>
        </tr>
    </thead>
    <tbody>
    '''

    sorted_df = df.sort_values('projected_avg_fp_2026', ascending=False).reset_index(drop=True)
    for i, (_, row) in enumerate(sorted_df.iterrows()):
        source = row['prediction_source']
        source_color = SOURCE_COLORS.get(source, '#666')
        team = row['Team']
        team_bg = TEAM_COLORS.get(team, '#666')
        team_txt = TEAM_TEXT_COLORS.get(team, '#fff')
        role = row['playing_role']
        acq = row.get('Acquisition', '') or ''
        ptype = row.get('Player_Type', '') or ''
        price = row['Price_Cr']
        fp = row['projected_avg_fp_2026']

        html += f'''        <tr data-team="{team}" data-role="{role}" data-source="{source}">
            <td>{i+1}</td>
            <td class="player-name">{row['Player']}</td>
            <td><span class="team-badge" style="background:{team_bg};color:{team_txt}">{team}</span></td>
            <td class="fp-cell"><strong>{fp:.1f}</strong></td>
            <td>{price:.1f}</td>
            <td class="role-cell">{role}</td>
            <td><span class="source-dot" style="background:{source_color}"></span>{source}</td>
            <td>{acq}</td>
            <td>{ptype}</td>
        </tr>
'''

    html += '''    </tbody>
    </table>
    </div>
    '''
    return html


def build_squad_tables_html(df):
    """Build per-team squad tables."""
    teams_sorted = df.groupby('Team')['projected_avg_fp_2026'].mean().sort_values(ascending=False).index.tolist()

    html = '<div class="section"><h2>Squad Rosters by Team</h2>\n'

    for team in teams_sorted:
        team_df = df[df['Team'] == team].sort_values('projected_avg_fp_2026', ascending=False)
        team_bg = TEAM_COLORS.get(team, '#666')
        team_txt = TEAM_TEXT_COLORS.get(team, '#fff')
        avg_fp = team_df['projected_avg_fp_2026'].mean()
        total_spend = team_df['Price_Cr'].sum()

        html += f'''
    <div class="squad-card">
        <div class="squad-header" style="background:{team_bg};color:{team_txt}">
            <h3>{team} <span class="squad-meta">Avg FP: {avg_fp:.1f} | Spend: ₹{total_spend:.1f}Cr | 25 players</span></h3>
        </div>
        <table class="squad-table">
        <thead><tr>
            <th>#</th><th>Player</th><th>FP</th><th>Price</th><th>Role</th><th>Source</th><th>Acquisition</th>
        </tr></thead>
        <tbody>
'''
        for i, (_, row) in enumerate(team_df.iterrows()):
            source = row['prediction_source']
            sc = SOURCE_COLORS.get(source, '#666')
            role = row.get('playing_role', '') or ''
            if role in ('Unknown', 'Error'):
                role = ''
            acq = row.get('Acquisition', '') or ''
            html += f'''        <tr>
            <td>{i+1}</td>
            <td>{row['Player']}</td>
            <td><strong>{row['projected_avg_fp_2026']:.1f}</strong></td>
            <td>₹{row['Price_Cr']:.1f}Cr</td>
            <td>{role}</td>
            <td><span class="source-dot" style="background:{sc}"></span>{source}</td>
            <td>{acq}</td>
        </tr>
'''
        html += '    </tbody></table></div>\n'

    html += '</div>\n'
    return html


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
        height=500, template='plotly_white',
        margin=dict(l=60, r=40, t=60, b=40),
    )

    # ── 2. Top 30 Players ────────────────────────────────────────────
    top30 = df.nlargest(30, 'projected_avg_fp_2026').iloc[::-1]
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
        height=800, template='plotly_white',
        margin=dict(l=200, r=40, t=60, b=40),
    )

    # ── 3. Scatter ────────────────────────────────────────────────────
    df_val = df[df['Price_Cr'] > 0].copy()
    df_val['size_col'] = df_val['projected_avg_fp_2026'].clip(lower=1)
    fig_scatter = px.scatter(
        df_val, x='Price_Cr', y='projected_avg_fp_2026',
        color='Team', color_discrete_map=TEAM_COLORS,
        hover_name='Player',
        hover_data={'prediction_source': True, 'playing_role': True, 'Price_Cr': ':.1f'},
        size='size_col', size_max=15,
        template='plotly_white',
        title='Fantasy Points vs Price — All Squad Players',
        labels={'projected_avg_fp_2026': 'Projected Avg FP/Match', 'Price_Cr': 'Price (₹ Cr)'},
    )
    fig_scatter.update_layout(height=600)

    # ── 4. Value Analysis ─────────────────────────────────────────────
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
        height=800, template='plotly_white',
        margin=dict(l=200, r=40, t=60, b=40),
    )

    # ── Build HTML tables ─────────────────────────────────────────────
    full_table_html = build_full_table_html(df)
    squad_tables_html = build_squad_tables_html(df)

    # ── Write HTML ────────────────────────────────────────────────────
    output_path = output_dir / 'squad_fantasy_visualization.html'
    with open(output_path, 'w') as f:
        f.write('''<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>IPL 2026 Fantasy Projections</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; color: #333; }
h1 { text-align: center; color: #1a1a2e; margin-bottom: 5px; }
h2 { color: #1a1a2e; border-bottom: 2px solid #e0e0e0; padding-bottom: 8px; margin-top: 40px; }
p.subtitle { text-align: center; color: #666; margin-top: 0; }
.section { background: white; padding: 25px; border-radius: 10px;
           box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 30px; }
nav { background: white; padding: 15px 25px; border-radius: 10px; margin-bottom: 25px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center; }
nav a { margin: 0 12px; text-decoration: none; color: #004BA0; font-weight: 500; }
nav a:hover { text-decoration: underline; }

/* Filters */
.filters { margin-bottom: 15px; display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }
.filters label { font-size: 14px; color: #555; }
.filters select, .filters input { padding: 6px 10px; border: 1px solid #ccc; border-radius: 5px;
                                    font-size: 14px; }
.filters button { padding: 6px 14px; background: #e0e0e0; border: none; border-radius: 5px;
                   cursor: pointer; font-size: 14px; }
.filters button:hover { background: #ccc; }

/* Tables */
#player-table, .squad-table { width: 100%; border-collapse: collapse; font-size: 14px; }
#player-table th, .squad-table th { background: #f8f9fa; padding: 10px 8px; text-align: left;
                                     border-bottom: 2px solid #dee2e6; font-weight: 600; }
#player-table td, .squad-table td { padding: 8px; border-bottom: 1px solid #eee; }
#player-table tbody tr:hover, .squad-table tbody tr:hover { background: #f0f7ff; }
.sortable { cursor: pointer; user-select: none; }
.sortable:hover { color: #004BA0; }
.fp-cell { font-variant-numeric: tabular-nums; }
.team-badge { padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; white-space: nowrap; }
.source-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%;
              margin-right: 5px; vertical-align: middle; }
.role-cell { font-size: 13px; color: #555; }
.player-name { font-weight: 500; }

/* Squad cards */
.squad-card { margin-bottom: 25px; border-radius: 8px; overflow: hidden;
              border: 1px solid #e0e0e0; }
.squad-header { padding: 12px 18px; }
.squad-header h3 { margin: 0; font-size: 18px; }
.squad-meta { font-size: 13px; font-weight: 400; opacity: 0.85; margin-left: 12px; }
</style>
</head>
<body>
<h1>IPL 2026 Squad Fantasy Point Projections</h1>
<p class="subtitle">Dream11 scoring &mdash; XGBoost + RF + Marcel ensemble model</p>

<nav>
    <a href="#charts">Charts</a>
    <a href="#full-table-section">All Players Table</a>
    <a href="#squad-rosters">Squad Rosters</a>
</nav>
''')

        # Charts section
        f.write('<div id="charts">\n')
        for fig in [fig_team, fig_top, fig_scatter, fig_value]:
            f.write('<div class="section">\n')
            f.write(fig.to_html(full_html=False, include_plotlyjs=False))
            f.write('</div>\n')
        f.write('</div>\n')

        # Full filterable table
        f.write(full_table_html)

        # Per-team squad tables
        f.write('<div id="squad-rosters">\n')
        f.write(squad_tables_html)
        f.write('</div>\n')

        # JavaScript for filtering and sorting
        f.write('''
<script>
function filterTable() {
    const team = document.getElementById('filter-team').value;
    const role = document.getElementById('filter-role').value;
    const source = document.getElementById('filter-source').value;
    const search = document.getElementById('filter-search').value.toLowerCase();
    const rows = document.querySelectorAll('#player-table tbody tr');
    let visible = 0;
    rows.forEach((row, i) => {
        const matchTeam = !team || row.dataset.team === team;
        const matchRole = !role || row.dataset.role === role;
        const matchSource = !source || row.dataset.source === source;
        const name = row.querySelector('.player-name').textContent.toLowerCase();
        const matchSearch = !search || name.includes(search);
        const show = matchTeam && matchRole && matchSource && matchSearch;
        row.style.display = show ? '' : 'none';
        if (show) { visible++; row.cells[0].textContent = visible; }
    });
    document.getElementById('row-count').textContent = visible + ' of 250 players';
}

function resetFilters() {
    document.getElementById('filter-team').value = '';
    document.getElementById('filter-role').value = '';
    document.getElementById('filter-source').value = '';
    document.getElementById('filter-search').value = '';
    filterTable();
}

let sortCol = -1, sortAsc = true;
function sortTable(col, type) {
    const tbody = document.querySelector('#player-table tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    if (sortCol === col) { sortAsc = !sortAsc; } else { sortCol = col; sortAsc = true; }
    rows.sort((a, b) => {
        let va = a.cells[col].textContent.trim();
        let vb = b.cells[col].textContent.trim();
        if (type === 'num') { va = parseFloat(va) || 0; vb = parseFloat(vb) || 0; }
        if (va < vb) return sortAsc ? -1 : 1;
        if (va > vb) return sortAsc ? 1 : -1;
        return 0;
    });
    rows.forEach((r, i) => { tbody.appendChild(r); r.cells[0].textContent = i + 1; });
}

// Initialize count
filterTable();
</script>
</body></html>
''')

    print(f"\n✓ Saved visualization to {output_path}")
    print(f"  Open in browser: file://{output_path.resolve()}")


if __name__ == '__main__':
    main()
