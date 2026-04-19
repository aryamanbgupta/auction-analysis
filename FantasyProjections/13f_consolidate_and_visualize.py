"""
Consolidate Marcel + XGBoost projections into one table and emit an
interactive HTML dashboard.

Inputs:
  - results/FantasyProjections/ipl2026_custom/marcel_projections_2026.csv
  - results/FantasyProjections/ipl2026_custom/xgb_projections_2026.csv
  - results/FantasyProjections/ipl2026_custom/xgb_loso_backtest.csv

Outputs:
  - results/FantasyProjections/ipl2026_custom/projections_2026.csv
  - results/FantasyProjections/ipl2026_custom/projections_2026.html
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS = PROJECT_ROOT / 'results' / 'FantasyProjections' / 'ipl2026_custom'
MARCEL = RESULTS / 'marcel_projections_2026.csv'
XGB = RESULTS / 'xgb_projections_2026.csv'
BACKTEST = RESULTS / 'xgb_loso_backtest.csv'
OUT_CSV = RESULTS / 'projections_2026.csv'
OUT_HTML = RESULTS / 'projections_2026.html'


def main():
    marcel = pd.read_csv(MARCEL)
    xgb = pd.read_csv(XGB)
    bt = pd.read_csv(BACKTEST)

    # Use XGB as the base (wider coverage) and merge Marcel's full_season
    m = marcel[['player', 'marcel_full_season']].rename(
        columns={'player': 'player_name', 'marcel_full_season': 'marcel_projection'}
    )
    combined = xgb.merge(m, on='player_name', how='left')

    # Blend: simple average as an ensemble estimate when both exist
    combined['ensemble_projection'] = combined[['final_projection', 'marcel_projection']].mean(axis=1)
    combined['ensemble_projection'] = combined['ensemble_projection'].fillna(
        combined['final_projection']
    )

    # Round
    for c in ['marcel_projection', 'ensemble_projection']:
        combined[c] = combined[c].round(1)

    # Sort by XGB final_projection
    combined = combined.sort_values('final_projection', ascending=False).reset_index(drop=True)
    combined['rank'] = combined.index + 1

    keep = ['rank', 'player_name', 'owner', 'ipl_team', 'role',
            'price_L', 'app_points',
            'ytd_matches', 'ytd_total',
            'team_games_played', 'team_games_left',
            'career_matches', 'career_ppm',
            'lag1_matches', 'lag1_ppm',
            'early_matches', 'early_ppm',
            'marcel_ppm', 'role_prior_ppm',
            'xgb_full_season', 'xgb_remaining',
            'marcel_projection',
            'ensemble_projection',
            'is_rookie', 'final_projection']
    combined = combined[keep]

    combined.to_csv(OUT_CSV, index=False)
    print(f"✓ {OUT_CSV} ({len(combined)} rows)")

    # Backtest summary stats for header display
    bt_summary = bt.groupby('season').apply(
        lambda g: pd.Series({
            'n': len(g),
            'xgb_mae': (g['xgb_pred'] - g['target_total']).abs().mean(),
            'marcel_mae': (g['marcel_pred'] - g['target_total']).abs().mean(),
        }), include_groups=False
    ).reset_index()
    overall_xgb = bt_summary['xgb_mae'].mean()
    overall_marcel = bt_summary['marcel_mae'].mean()

    # Build HTML
    # Data for JS
    display_cols = ['rank', 'player_name', 'owner', 'ipl_team', 'role',
                    'price_L', 'app_points',
                    'ytd_matches', 'ytd_total',
                    'final_projection', 'marcel_projection', 'ensemble_projection',
                    'xgb_remaining', 'is_rookie']
    # JSON-friendly: replace NaN/inf
    rows = combined[display_cols].copy()
    for c in rows.columns:
        if rows[c].dtype == object:
            rows[c] = rows[c].fillna('')
        else:
            rows[c] = rows[c].replace([np.inf, -np.inf], np.nan).fillna(0)
    data = rows.to_dict(orient='records')

    owners = sorted([o for o in combined['owner'].dropna().unique() if o])
    teams = sorted([t for t in combined['ipl_team'].dropna().unique() if t])
    roles = ['BAT', 'BOWL', 'AR', 'WK']

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>IPL 2026 Fantasy Projections — Custom Rules</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 20px; color: #222; background: #fafafa; }}
  h1 {{ margin-bottom: 4px; }}
  .sub {{ color: #666; margin-bottom: 18px; }}
  .stats {{ background: #eef; padding: 10px 14px; border-radius: 6px; margin-bottom: 16px; font-size: 14px; }}
  .controls {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }}
  .controls select, .controls input {{ padding: 6px 10px; font-size: 14px; border: 1px solid #ccc; border-radius: 4px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; background: white; }}
  th, td {{ padding: 6px 10px; border-bottom: 1px solid #eee; text-align: left; }}
  th {{ background: #333; color: white; cursor: pointer; user-select: none; position: sticky; top: 0; }}
  th:hover {{ background: #555; }}
  tr:hover {{ background: #f6f6f6; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .rookie {{ color: #a40; font-style: italic; }}
  .owner-unsold {{ color: #888; }}
  .pagination {{ margin-top: 12px; color: #666; font-size: 13px; }}
</style>
</head>
<body>
<h1>IPL 2026 Fantasy Projections — Custom Rules</h1>
<div class="sub">
Boston IPL 26 league · {len(combined)} players (drafted + unsold + full squads) ·
Full group-stage (14 games) projection
</div>

<div class="stats">
<b>Model backtest (LOSO, 2020-2025):</b>
XGBoost MAE = <b>{overall_xgb:.1f}</b> pts ·
Marcel MAE = <b>{overall_marcel:.1f}</b> pts ·
Features: career, 3-season lag, early-season (pre-6-games) stats, role prior ·
Final XGB model trained on 2012-2025
</div>

<div class="controls">
  <input id="search" type="text" placeholder="Search player name…">
  <select id="owner">
    <option value="">All owners</option>
    <option value="__drafted__">Any Boston manager</option>
    <option value="Unsold">Unsold</option>
    <option value="__nobid__">Not listed</option>
    {''.join(f'<option value="{o}">{o}</option>' for o in owners)}
  </select>
  <select id="team">
    <option value="">All IPL teams</option>
    {''.join(f'<option value="{t}">{t}</option>' for t in teams)}
  </select>
  <select id="role">
    <option value="">All roles</option>
    {''.join(f'<option value="{r}">{r}</option>' for r in roles)}
  </select>
  <select id="sort">
    <option value="final_projection">Sort: XGB projection</option>
    <option value="marcel_projection">Sort: Marcel projection</option>
    <option value="ensemble_projection">Sort: Ensemble</option>
    <option value="ytd_total">Sort: YTD pts</option>
    <option value="price_L">Sort: Price (₹L)</option>
  </select>
</div>

<table id="t">
<thead><tr>
<th data-k="rank" class="num">#</th>
<th data-k="player_name">Player</th>
<th data-k="owner">Owner</th>
<th data-k="ipl_team">IPL Team</th>
<th data-k="role">Role</th>
<th data-k="price_L" class="num">Price (₹L)</th>
<th data-k="ytd_matches" class="num">YTD M</th>
<th data-k="ytd_total" class="num">YTD Pts</th>
<th data-k="final_projection" class="num">XGB Full</th>
<th data-k="marcel_projection" class="num">Marcel</th>
<th data-k="ensemble_projection" class="num">Ensemble</th>
<th data-k="xgb_remaining" class="num">Rest</th>
</tr></thead>
<tbody id="body"></tbody>
</table>
<div id="pag" class="pagination"></div>

<script>
const DATA = {json.dumps(data)};

function fmt(v, k) {{
  if (v === null || v === undefined || v === '') return '';
  if (['price_L','ytd_total','final_projection','marcel_projection','ensemble_projection','xgb_remaining'].includes(k)) {{
    return typeof v === 'number' ? v.toFixed(1) : v;
  }}
  if (['rank','ytd_matches'].includes(k)) {{
    return typeof v === 'number' ? Math.round(v) : v;
  }}
  return v;
}}

function render(rows) {{
  const body = document.getElementById('body');
  body.innerHTML = rows.map(r => {{
    const numCls = ['rank','price_L','ytd_matches','ytd_total','final_projection','marcel_projection','ensemble_projection','xgb_remaining'];
    const cells = ['rank','player_name','owner','ipl_team','role','price_L','ytd_matches','ytd_total','final_projection','marcel_projection','ensemble_projection','xgb_remaining']
      .map(k => {{
        let cls = numCls.includes(k) ? 'num' : '';
        if (k === 'owner' && r[k] === 'Unsold') cls += ' owner-unsold';
        if (r.is_rookie && k === 'player_name') cls += ' rookie';
        return `<td class="${{cls}}">${{fmt(r[k], k)}}</td>`;
      }}).join('');
    return `<tr>${{cells}}</tr>`;
  }}).join('');
  document.getElementById('pag').textContent = `Showing ${{rows.length}} of ${{DATA.length}} players`;
}}

function filtered() {{
  const q = document.getElementById('search').value.toLowerCase();
  const o = document.getElementById('owner').value;
  const t = document.getElementById('team').value;
  const rl = document.getElementById('role').value;
  return DATA.filter(r => {{
    if (q && !(r.player_name || '').toLowerCase().includes(q)) return false;
    if (o === '__drafted__') {{ if (!r.owner || r.owner === 'Unsold' || r.owner === '') return false; }}
    else if (o === '__nobid__') {{ if (r.owner) return false; }}
    else if (o && r.owner !== o) return false;
    if (t && r.ipl_team !== t) return false;
    if (rl && r.role !== rl) return false;
    return true;
  }});
}}

function sorted(rows) {{
  const key = document.getElementById('sort').value;
  return rows.slice().sort((a, b) => (b[key] || 0) - (a[key] || 0));
}}

function refresh() {{
  render(sorted(filtered()));
}}

['search','owner','team','role','sort'].forEach(id => {{
  document.getElementById(id).addEventListener('input', refresh);
  document.getElementById(id).addEventListener('change', refresh);
}});

// Click-to-sort on headers
document.querySelectorAll('#t thead th').forEach(th => {{
  th.addEventListener('click', () => {{
    const k = th.dataset.k;
    const body = document.getElementById('body');
    const rows = [...body.querySelectorAll('tr')].map(r => [...r.children].map(c => c.textContent));
    const headers = [...th.parentNode.children].map(c => c.dataset.k);
    const i = headers.indexOf(k);
    const asc = th.dataset.asc !== 'true';
    rows.sort((a, b) => {{
      const va = parseFloat(a[i]); const vb = parseFloat(b[i]);
      if (!isNaN(va) && !isNaN(vb)) return asc ? va - vb : vb - va;
      return asc ? a[i].localeCompare(b[i]) : b[i].localeCompare(a[i]);
    }});
    th.parentNode.parentNode.parentNode.querySelector('tbody').innerHTML =
      rows.map(r => `<tr>${{r.map((c, ci) => `<td class="${{th.parentNode.children[ci].classList.contains('num') ? 'num' : ''}}">${{c}}</td>`).join('')}}</tr>`).join('');
    th.dataset.asc = asc;
  }});
}});

refresh();
</script>
</body></html>"""

    OUT_HTML.write_text(html)
    print(f"✓ {OUT_HTML}")

    print("\n── Owner summary ──")
    summary = combined[combined['owner'].notna() & (combined['owner'] != 'Unsold')
                       & (combined['owner'] != 'Awards')]
    print(summary.groupby('owner').agg(
        n=('player_name', 'count'),
        xgb_sum=('final_projection', 'sum'),
        marcel_sum=('marcel_projection', 'sum'),
        ensemble_sum=('ensemble_projection', 'sum'),
    ).sort_values('ensemble_sum', ascending=False).round(1).to_string())

    print(f"\nTop 10 unsold (XGB):")
    unsold = combined[combined['owner'] == 'Unsold'].nlargest(10, 'final_projection')
    print(unsold[['player_name', 'ipl_team', 'role', 'final_projection']].to_string(index=False))


if __name__ == '__main__':
    main()
