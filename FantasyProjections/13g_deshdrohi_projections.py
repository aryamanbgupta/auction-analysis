"""
Deshdrohi Babes league — re-owner the XGB 2026 projections and build an HTML
dashboard with tabs for rankings, team mapping, and unsold players.

Inputs:
  - data/Deshdrohi Babes Auction_Results (1).xlsx
  - results/FantasyProjections/ipl2026_custom/xgb_projections_2026.csv
  - results/FantasyProjections/ipl2026_custom/marcel_projections_2026.csv
  - results/FantasyProjections/ipl2026_custom/xgb_loso_backtest.csv

Outputs:
  - results/FantasyProjections/ipl2026_custom/projections_deshdrohi.csv
  - results/FantasyProjections/ipl2026_custom/projections_deshdrohi.html
"""
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS = PROJECT_ROOT / 'results' / 'FantasyProjections' / 'ipl2026_custom'
DESHD_FILE = PROJECT_ROOT / 'data' / 'Deshdrohi Babes Auction_Results (1).xlsx'
XGB_FILE = RESULTS / 'xgb_projections_2026.csv'
MARCEL_FILE = RESULTS / 'marcel_projections_2026.csv'
BACKTEST_FILE = RESULTS / 'xgb_loso_backtest.csv'
OUT_CSV = RESULTS / 'projections_deshdrohi.csv'
OUT_HTML = RESULTS / 'projections_deshdrohi.html'

SKIP_NAME_TAGS = ('TOTAL', 'PURSE', 'REMAINING', 'BOUGHT', 'SPENT', 'RATING', 'SQUAD')


def load_deshdrohi():
    x = pd.read_excel(DESHD_FILE, sheet_name=None)
    drafted_rows, unsold_rows = [], []
    for sheet, d in x.items():
        if sheet.lower() == 'awards' or d is None or d.empty:
            continue
        if 'Player' not in d.columns:
            continue
        is_unsold = 'unsold' in sheet.lower()
        for _, r in d.iterrows():
            p = r.get('Player')
            if not isinstance(p, str) or not p.strip():
                continue
            upper = p.upper()
            if any(t in upper for t in SKIP_NAME_TAGS):
                continue
            rec = {
                'player_name': p.strip(),
                'ipl_team_deshd': r.get('Team'),
                'role_deshd': r.get('Role'),
                'base_L': r.get('Base (₹L)'),
                'price_L_deshd': r.get('Price (₹L)') if not is_unsold else None,
                'app_points_deshd': r.get('Points') if not is_unsold else None,
            }
            if is_unsold:
                rec['owner_deshd'] = 'Unsold'
                unsold_rows.append(rec)
            else:
                rec['owner_deshd'] = sheet
                drafted_rows.append(rec)
    drafted = pd.DataFrame(drafted_rows)
    unsold = pd.DataFrame(unsold_rows)
    return drafted, unsold


def main():
    xgb = pd.read_csv(XGB_FILE)
    marcel = pd.read_csv(MARCEL_FILE)
    bt = pd.read_csv(BACKTEST_FILE)

    drafted, unsold = load_deshdrohi()
    print(f"Deshdrohi drafted: {len(drafted)}, unsold listed: {len(unsold)}")

    # Union by player name (preserve drafted first → they take precedence on conflicts)
    deshd = pd.concat([drafted, unsold], ignore_index=True)
    deshd = deshd.drop_duplicates('player_name', keep='first')

    # Strip the Boston-specific owner/price/app_points from xgb, add Deshdrohi's
    xgb_clean = xgb.drop(columns=['owner', 'price_L', 'app_points'], errors='ignore')

    # Merge: xgb ∪ deshd by player_name. Keep xgb as base; pull in deshd fields.
    df = xgb_clean.merge(deshd, on='player_name', how='left')
    df['owner'] = df['owner_deshd'].fillna('Not listed')
    # Prefer Deshdrohi's team label if provided
    df['ipl_team'] = df['ipl_team_deshd'].fillna(df['ipl_team'])
    df['price_L'] = df['price_L_deshd']
    df['app_points'] = df['app_points_deshd']

    # Marcel blend
    m = marcel[['player', 'marcel_full_season']].rename(
        columns={'player': 'player_name', 'marcel_full_season': 'marcel_projection'}
    )
    df = df.merge(m, on='player_name', how='left')
    df['ensemble_projection'] = df[['final_projection', 'marcel_projection']].mean(axis=1)
    df['ensemble_projection'] = df['ensemble_projection'].fillna(df['final_projection'])

    for c in ['marcel_projection', 'ensemble_projection']:
        df[c] = df[c].round(1)

    df = df.sort_values('final_projection', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1

    # For players that appear only in Deshdrohi's lists but not in our 259-player
    # projection universe, they'd be absent. Surface them with NaN projections.
    deshd_names = set(deshd['player_name'])
    xgb_names = set(xgb_clean['player_name'])
    missing = deshd_names - xgb_names
    if missing:
        print(f"  {len(missing)} Deshdrohi players missing from XGB output — adding with NaN projection")
        extra = deshd[deshd['player_name'].isin(missing)].copy()
        extra['owner'] = extra['owner_deshd']
        extra['ipl_team'] = extra['ipl_team_deshd']
        extra['price_L'] = extra['price_L_deshd']
        extra['app_points'] = extra['app_points_deshd']
        # Fill projection columns with 0/None
        for c in ['final_projection', 'marcel_projection', 'ensemble_projection',
                  'xgb_full_season', 'xgb_remaining', 'ytd_matches', 'ytd_total',
                  'career_matches', 'career_ppm', 'lag1_matches', 'lag1_ppm',
                  'early_matches', 'early_ppm', 'marcel_ppm', 'role_prior_ppm',
                  'team_games_played', 'team_games_left']:
            if c in df.columns and c not in extra.columns:
                extra[c] = 0
        # Role fallback: normalize from role_deshd
        def norm(r):
            if pd.isna(r): return 'AR'
            s = str(r).upper()
            if 'WICKET' in s: return 'WK'
            if 'BATSMAN' in s or s.strip() == 'BAT': return 'BAT'
            if 'BOWLER' in s: return 'BOWL'
            if 'ROUNDER' in s or 'AR' in s: return 'AR'
            return 'AR'
        extra['role'] = extra['role_deshd'].apply(norm)
        extra['is_rookie'] = True
        df = pd.concat([df, extra[df.columns.intersection(extra.columns)]], ignore_index=True)
        df = df.sort_values('final_projection', ascending=False, na_position='last').reset_index(drop=True)
        df['rank'] = df.index + 1

    keep = ['rank', 'player_name', 'owner', 'ipl_team', 'role',
            'price_L', 'app_points',
            'ytd_matches', 'ytd_total',
            'team_games_played', 'team_games_left',
            'career_matches', 'career_ppm',
            'lag1_matches', 'lag1_ppm',
            'early_matches', 'early_ppm',
            'marcel_ppm', 'role_prior_ppm',
            'xgb_full_season', 'xgb_remaining',
            'marcel_projection', 'ensemble_projection',
            'is_rookie', 'final_projection']
    out = df[[c for c in keep if c in df.columns]].copy()
    out.to_csv(OUT_CSV, index=False)
    print(f"✓ {OUT_CSV}  ({len(out)} players)")

    # Backtest metrics for header
    bt_by = bt.groupby('season').apply(
        lambda g: pd.Series({
            'xgb_mae': (g['xgb_pred'] - g['target_total']).abs().mean(),
            'marcel_mae': (g['marcel_pred'] - g['target_total']).abs().mean(),
        }), include_groups=False
    ).reset_index()
    xgb_mae = bt_by['xgb_mae'].mean()
    marcel_mae = bt_by['marcel_mae'].mean()

    # ── Owner summary + unsold list for tabs ────────────────────────────
    drafted_df = out[out['owner'].isin(drafted['owner_deshd'].unique())].copy()
    unsold_df = out[out['owner'] == 'Unsold'].copy()
    not_listed_df = out[out['owner'] == 'Not listed'].copy()

    owner_sum = drafted_df.groupby('owner').agg(
        players=('player_name', 'count'),
        total_proj=('final_projection', 'sum'),
        mean_proj=('final_projection', 'mean'),
        total_price_L=('price_L', 'sum'),
        avg_ppm=('final_projection', lambda x: x.mean() / 14),
    ).round(1).reset_index().sort_values('total_proj', ascending=False)

    print("\n── Deshdrohi Owner Rankings (by total XGB projection) ──")
    print(owner_sum.to_string(index=False))

    # Top 10 unsold
    print("\n── Top 10 Unsold (Deshdrohi) ──")
    print(unsold_df.nlargest(10, 'final_projection')[
        ['player_name', 'ipl_team', 'role', 'final_projection']
    ].to_string(index=False))

    # ── HTML ────────────────────────────────────────────────────────────
    rank_cols = ['rank', 'player_name', 'owner', 'ipl_team', 'role',
                 'price_L', 'app_points',
                 'ytd_matches', 'ytd_total',
                 'final_projection', 'marcel_projection', 'ensemble_projection',
                 'xgb_remaining', 'is_rookie']
    rows = out[rank_cols].copy()
    for c in rows.columns:
        if rows[c].dtype == object:
            rows[c] = rows[c].fillna('')
        else:
            rows[c] = rows[c].replace([np.inf, -np.inf], np.nan).fillna(0)
    data = rows.to_dict(orient='records')

    # Team mapping data: per-owner roster
    roster = drafted_df.sort_values(['owner', 'final_projection'],
                                    ascending=[True, False])
    team_data = {}
    for owner, grp in roster.groupby('owner'):
        team_data[owner] = grp[['player_name', 'ipl_team', 'role',
                                'price_L', 'app_points', 'final_projection',
                                'ytd_total']].round(1).to_dict(orient='records')

    unsold_data = unsold_df.sort_values('final_projection', ascending=False)[
        ['player_name', 'ipl_team', 'role', 'base_L' if 'base_L' in unsold_df.columns else 'price_L',
         'final_projection', 'ytd_total']
    ].copy()
    unsold_data = unsold_data.fillna('')
    unsold_records = unsold_data.to_dict(orient='records')

    owner_sum_records = owner_sum.to_dict(orient='records')

    owners = sorted(out['owner'].dropna().unique())
    teams = sorted([t for t in out['ipl_team'].dropna().unique() if t])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Deshdrohi Babes — IPL 2026 Fantasy Projections</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 20px; color: #222; background: #fafafa; }}
  h1 {{ margin-bottom: 4px; }}
  h2 {{ margin-top: 24px; margin-bottom: 8px; color: #333; }}
  .sub {{ color: #666; margin-bottom: 18px; }}
  .stats {{ background: #eef; padding: 10px 14px; border-radius: 6px; margin-bottom: 16px; font-size: 14px; }}
  .tabs {{ display: flex; gap: 0; margin-bottom: 0; border-bottom: 2px solid #333; }}
  .tab {{ padding: 10px 18px; cursor: pointer; border: 1px solid #ccc; border-bottom: none; background: #ddd;
          font-weight: 500; border-radius: 6px 6px 0 0; margin-right: 2px; }}
  .tab.active {{ background: #333; color: white; border-color: #333; }}
  .panel {{ display: none; padding-top: 16px; }}
  .panel.active {{ display: block; }}
  .controls {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }}
  .controls select, .controls input {{ padding: 6px 10px; font-size: 14px; border: 1px solid #ccc; border-radius: 4px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; background: white; }}
  th, td {{ padding: 6px 10px; border-bottom: 1px solid #eee; text-align: left; }}
  th {{ background: #333; color: white; position: sticky; top: 0; cursor: pointer; user-select: none; }}
  th:hover {{ background: #555; }}
  tr:hover {{ background: #f6f6f6; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .rookie {{ color: #a40; font-style: italic; }}
  .owner-unsold {{ color: #888; }}
  .team-card {{ background: white; border: 1px solid #ddd; border-radius: 8px; padding: 14px 18px;
                margin-bottom: 18px; }}
  .team-card h3 {{ margin: 0 0 6px 0; display: flex; justify-content: space-between; align-items: baseline; }}
  .team-card .tsub {{ color: #666; font-size: 13px; font-weight: normal; }}
  .team-card table {{ margin-top: 6px; }}
  .pagination {{ margin-top: 12px; color: #666; font-size: 13px; }}
</style>
</head>
<body>
<h1>Deshdrohi Babes — IPL 2026 Fantasy Projections</h1>
<div class="sub">6 managers · XGBoost projections under published custom-scoring rules</div>
<div class="stats">
<b>Backtest (LOSO, 2020-2025):</b> XGB MAE = <b>{xgb_mae:.1f}</b> pts · Marcel MAE = <b>{marcel_mae:.1f}</b> pts.
Every player is projected for the full 14-game group stage.
</div>

<div class="tabs">
  <div class="tab active" data-tab="rank">Rankings</div>
  <div class="tab" data-tab="team">Team Mapping</div>
  <div class="tab" data-tab="unsold">Unsold Players</div>
  <div class="tab" data-tab="standings">Manager Standings</div>
</div>

<div id="panel-rank" class="panel active">
<div class="controls">
  <input id="search" type="text" placeholder="Search player name…">
  <select id="owner">
    <option value="">All owners</option>
    <option value="__drafted__">Any Deshdrohi manager</option>
    {''.join(f'<option value="{o}">{o}</option>' for o in owners)}
  </select>
  <select id="team">
    <option value="">All IPL teams</option>
    {''.join(f'<option value="{t}">{t}</option>' for t in teams)}
  </select>
  <select id="role">
    <option value="">All roles</option>
    <option value="BAT">BAT</option><option value="BOWL">BOWL</option>
    <option value="AR">AR</option><option value="WK">WK</option>
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
<th class="num">#</th><th>Player</th><th>Owner</th><th>IPL Team</th><th>Role</th>
<th class="num">Price (₹L)</th><th class="num">YTD M</th><th class="num">YTD Pts</th>
<th class="num">XGB Full</th><th class="num">Marcel</th><th class="num">Ensemble</th><th class="num">Rest</th>
</tr></thead>
<tbody id="body"></tbody>
</table>
<div id="pag" class="pagination"></div>
</div>

<div id="panel-team" class="panel"></div>

<div id="panel-unsold" class="panel">
<p>Top-value unsold players (re-auction targets), sorted by XGB full-season projection.</p>
<table>
<thead><tr>
<th>Player</th><th>IPL Team</th><th>Role</th><th class="num">Base (₹L)</th>
<th class="num">YTD Pts</th><th class="num">XGB Full</th>
</tr></thead>
<tbody id="unsold-body"></tbody>
</table>
</div>

<div id="panel-standings" class="panel">
<p>Projected team strength based on XGB full-season totals (ranked by total projection).</p>
<table>
<thead><tr>
<th>Owner</th><th class="num">Players</th><th class="num">Total Proj</th>
<th class="num">Mean Proj</th><th class="num">Avg PPM</th><th class="num">Total Spent (₹L)</th>
</tr></thead>
<tbody id="standings-body"></tbody>
</table>
</div>

<script>
const DATA = {json.dumps(data)};
const TEAM_DATA = {json.dumps(team_data)};
const UNSOLD = {json.dumps(unsold_records)};
const STANDINGS = {json.dumps(owner_sum_records)};

function fmt(v, k) {{
  if (v === null || v === undefined || v === '') return '';
  if (['price_L','ytd_total','final_projection','marcel_projection','ensemble_projection','xgb_remaining','app_points','mean_proj','total_proj','avg_ppm','total_price_L'].includes(k)) {{
    return typeof v === 'number' ? v.toFixed(1) : v;
  }}
  if (['rank','ytd_matches','players'].includes(k)) {{
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
    if (o === '__drafted__') {{ if (!r.owner || r.owner === 'Unsold' || r.owner === 'Not listed' || r.owner === '') return false; }}
    else if (o && r.owner !== o) return false;
    if (t && r.ipl_team !== t) return false;
    if (rl && r.role !== rl) return false;
    return true;
  }});
}}

function sortedData(rows) {{
  const key = document.getElementById('sort').value;
  return rows.slice().sort((a, b) => (b[key] || 0) - (a[key] || 0));
}}

function refresh() {{ render(sortedData(filtered())); }}

['search','owner','team','role','sort'].forEach(id => {{
  document.getElementById(id).addEventListener('input', refresh);
  document.getElementById(id).addEventListener('change', refresh);
}});

// Team mapping tab
function renderTeams() {{
  const panel = document.getElementById('panel-team');
  const ranked = STANDINGS.slice().sort((a, b) => b.total_proj - a.total_proj);
  panel.innerHTML = ranked.map(s => {{
    const owner = s.owner;
    const roster = TEAM_DATA[owner] || [];
    const rows = roster.map(p => `
      <tr>
        <td>${{p.player_name}}</td>
        <td>${{p.ipl_team || ''}}</td>
        <td>${{p.role || ''}}</td>
        <td class="num">${{fmt(p.price_L, 'price_L')}}</td>
        <td class="num">${{fmt(p.ytd_total, 'ytd_total')}}</td>
        <td class="num">${{fmt(p.final_projection, 'final_projection')}}</td>
      </tr>`).join('');
    return `<div class="team-card">
      <h3>${{owner}}
        <span class="tsub">${{s.players}} players · total proj ${{fmt(s.total_proj, 'total_proj')}} ·
        mean ${{fmt(s.mean_proj, 'mean_proj')}} · spent ₹${{fmt(s.total_price_L, 'total_price_L')}}L</span>
      </h3>
      <table>
        <thead><tr><th>Player</th><th>IPL Team</th><th>Role</th>
        <th class="num">Price (₹L)</th><th class="num">YTD Pts</th><th class="num">XGB Full</th></tr></thead>
        <tbody>${{rows}}</tbody>
      </table>
    </div>`;
  }}).join('');
}}

// Unsold tab
function renderUnsold() {{
  const body = document.getElementById('unsold-body');
  body.innerHTML = UNSOLD.map(p => `
    <tr>
      <td>${{p.player_name}}</td>
      <td>${{p.ipl_team || ''}}</td>
      <td>${{p.role || ''}}</td>
      <td class="num">${{fmt(p.base_L || p.price_L, 'price_L')}}</td>
      <td class="num">${{fmt(p.ytd_total, 'ytd_total')}}</td>
      <td class="num">${{fmt(p.final_projection, 'final_projection')}}</td>
    </tr>`).join('');
}}

// Standings tab
function renderStandings() {{
  const body = document.getElementById('standings-body');
  const ranked = STANDINGS.slice().sort((a, b) => b.total_proj - a.total_proj);
  body.innerHTML = ranked.map(s => `
    <tr>
      <td>${{s.owner}}</td>
      <td class="num">${{s.players}}</td>
      <td class="num">${{fmt(s.total_proj, 'total_proj')}}</td>
      <td class="num">${{fmt(s.mean_proj, 'mean_proj')}}</td>
      <td class="num">${{fmt(s.avg_ppm, 'avg_ppm')}}</td>
      <td class="num">${{fmt(s.total_price_L, 'total_price_L')}}</td>
    </tr>`).join('');
}}

// Tab switching
document.querySelectorAll('.tab').forEach(tab => {{
  tab.addEventListener('click', () => {{
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('panel-' + tab.dataset.tab).classList.add('active');
  }});
}});

refresh();
renderTeams();
renderUnsold();
renderStandings();
</script>
</body></html>"""

    OUT_HTML.write_text(html)
    print(f"\n✓ {OUT_HTML}")


if __name__ == '__main__':
    main()
