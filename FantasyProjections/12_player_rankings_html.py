"""
Build an interactive HTML table of all players ranked by season-to-date
fantasy points (IPL 2026 mid-season, 24 matches played).

Columns: rank, player, IPL team, IPL role, matches, points, owner (Boston
manager or 'Unsold' or 'Not in auction'), price.

Filters: IPL team, IPL role, owner (any manager / Unsold / Drafted / All).
Search: free-text on player name.
Sort: click any column header.

Points source:
  * Use APP points when the player is drafted or listed unsold in the
    Boston file (the app is the ground truth for those players).
  * Fall back to computed V_NO_SR total (published-rules approximation)
    for cricsheet players not in the Boston file.

Output: results/FantasyProjections/ipl2026_custom/player_rankings.html
"""
import json
import re
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
BOSTON_FILE = PROJECT_ROOT / 'data' / 'Boston IPL 26_Results (1).xlsx'
VARIANTS_FILE = PROJECT_ROOT / 'results' / 'FantasyProjections' / 'ipl2026_custom' / 'season_to_date_totals.csv'
OUT_HTML = PROJECT_ROOT / 'results' / 'FantasyProjections' / 'ipl2026_custom' / 'player_rankings.html'

# Same cricsheet → Boston mapping used in reconciliation (reverse direction)
MANUAL = {
    'Vaibhav Sooryavanshi':'V Suryavanshi','Philip Salt':'PD Salt','Jos Buttler':'JC Buttler',
    'Yashasvi Jaiswal':'YBK Jaiswal','Prabhsimran Singh':'P Simran Singh','Sanju Samson':'SV Samson',
    'Suryakumar Yadav':'SA Yadav','Angkrish Raghuvanshi':'A Raghuvanshi','Ayush Mhatre':'A Mhatre',
    'Shivam Dube':'SM Dube','Shubman Gill':'Shubman Gill','Virat Kohli':'V Kohli','Ishan Kishan':'Ishan Kishan',
    'Heinrich Klaasen':'H Klaasen','Rajat Patidar':'RM Patidar','Shreyas Iyer':'SS Iyer',
    'KL Rahul':'KL Rahul','Travis Head':'TM Head','Jofra Archer':'JC Archer','Arshdeep Singh':'Arshdeep Singh',
    'Noor Ahmad':'Noor Ahmad','Mohammed Siraj':'Mohammed Siraj','Devdutt Padikkal':'D Padikkal',
    'Quinton de Kock':'Q de Kock','Hardik Pandya':'HH Pandya','Jasprit Bumrah':'JJ Bumrah',
    'Rishabh Pant':'RR Pant','Abhishek Sharma':'Abhishek Sharma','Ruturaj Gaikwad':'RD Gaikwad',
    'Sai Sudharsan':'B Sai Sudharsan','Rohit Sharma':'RG Sharma','Nicholas Pooran':'N Pooran',
    'Priyansh Arya':'Priyansh Arya','Ajinkya Rahane':'AM Rahane','Pathum Nissanka':'WPN Nissanka',
    'Josh Hazlewood':'JR Hazlewood','Jitesh Sharma':'Jitesh Sharma','Cameron Green':'CD Green',
    'Mitchell Marsh':'MR Marsh','Kuldeep Yadav':'Kuldeep Yadav','Tilak Varma':'Tilak Varma',
    'Dewald Brevis':'DR Brevis','Tim David':'TH David','Nehal Wadhera':'N Wadhera','Naman Dhir':'N Dhir',
    'Shimron Hetmyer':'SO Hetmyer','Prasidh Krishna':'M Prasidh Krishna','Bhuvneshwar Kumar':'B Kumar',
    'Marco Jansen':'M Jansen','Sunil Narine':'SP Narine','Rinku Singh':'RK Singh','Aiden Markram':'AK Markram',
    'Varun Chakaravarthy':'V Chakravarthy','Manish Pandey':'MK Pandey','Rovman Powell':'R Powell',
    'Nitish Rana':'N Rana','Finn Allen':'FH Allen','Trent Boult':'TA Boult',
    'Lockie Ferguson':'LH Ferguson','Kyle Jamieson':'KA Jamieson','Pat Cummins':'PJ Cummins',
    'Rashid Khan':'Rashid Khan','Ravindra Jadeja':'RA Jadeja','Axar Patel':'AR Patel','David Miller':'DA Miller',
    'Yuzvendra Chahal':'YS Chahal','Prashant Veer':'Prashant Veer','Jacob Bethell':'JG Bethell',
    'Matheesha Pathirana':'M Pathirana',
}


def load_boston() -> pd.DataFrame:
    x = pd.ExcelFile(BOSTON_FILE)
    rows = []
    for s in x.sheet_names:
        if s == 'Awards':
            continue
        d = pd.read_excel(x, sheet_name=s)
        if 'Player' not in d.columns:
            continue
        d = d[d['Player'].notna()].copy()
        d['Player'] = d['Player'].astype(str).str.strip()
        d['Manager'] = 'Unsold' if s == 'Unsold Players' else s
        keep = ['Manager', 'Player']
        for c in ('Role', 'Team', 'Price (₹L)', 'Points'):
            if c in d.columns:
                keep.append(c)
        rows.append(d[keep])
    boston = pd.concat(rows, ignore_index=True)
    # Remove non-player rows like 'TOTAL POINTS'
    boston = boston[~boston['Player'].str.contains('TOTAL', case=False, na=False)].copy()
    return boston


def main():
    variants = pd.read_csv(VARIANTS_FILE)
    boston = load_boston()

    # Build cricsheet <-> boston mapping
    boston['cricsheet_name'] = boston['Player'].map(MANUAL)

    # Outer-join so everyone appears (Boston drafted/unsold + cricsheet-only players).
    # Left side: Boston roster (owner is known). Right side: computed stats.
    merged = boston.merge(
        variants[['player_name', 'matches', 'V_NO_SR_total']],
        left_on='cricsheet_name', right_on='player_name', how='left'
    )

    # Pick the points to display: app Points if present, else computed V_NO_SR
    merged['computed_total'] = merged['V_NO_SR_total']
    merged['display_points'] = merged.apply(
        lambda r: r['Points'] if pd.notna(r.get('Points')) else r['computed_total'], axis=1
    )
    merged['display_points'] = pd.to_numeric(merged['display_points'], errors='coerce').fillna(0).astype(int)
    merged['matches'] = pd.to_numeric(merged['matches'], errors='coerce').fillna(0).astype(int)

    # Cricsheet-only players not in Boston file (e.g., overseas swapped-in, depth)
    in_boston = set(merged['cricsheet_name'].dropna())
    extras = variants[~variants['player_name'].isin(in_boston)].copy()
    extras = extras.rename(columns={'player_name': 'Player'})
    extras['Manager'] = 'Not in auction'
    extras['Role'] = ''
    extras['Team'] = ''
    extras['Price (₹L)'] = 0
    extras['Points'] = pd.NA
    extras['display_points'] = extras['V_NO_SR_total'].astype(int)
    extras['matches'] = extras['matches'].astype(int)
    extras['cricsheet_name'] = extras['Player']

    cols = ['Player', 'cricsheet_name', 'Team', 'Role', 'Manager',
            'Price (₹L)', 'matches', 'display_points', 'Points', 'V_NO_SR_total']
    all_players = pd.concat([merged[cols], extras[cols]], ignore_index=True)
    all_players = all_players.sort_values('display_points', ascending=False).reset_index(drop=True)
    all_players['rank'] = all_players.index + 1

    # Build JSON payload for the HTML
    records = []
    for _, r in all_players.iterrows():
        records.append({
            'rank': int(r['rank']),
            'player': r['Player'],
            'team': r['Team'] if pd.notna(r['Team']) else '',
            'role': r['Role'] if pd.notna(r['Role']) else '',
            'manager': r['Manager'] if pd.notna(r['Manager']) else '',
            'price': float(r['Price (₹L)']) if pd.notna(r['Price (₹L)']) else 0,
            'matches': int(r['matches']) if pd.notna(r['matches']) else 0,
            'points': int(r['display_points']) if pd.notna(r['display_points']) else 0,
            'app_points': int(r['Points']) if pd.notna(r['Points']) else None,
            'computed_points': int(r['V_NO_SR_total']) if pd.notna(r['V_NO_SR_total']) else None,
        })

    # Distinct filter options
    teams = sorted({rec['team'] for rec in records if rec['team']})
    roles = sorted({rec['role'] for rec in records if rec['role']})
    managers = sorted({rec['manager'] for rec in records if rec['manager']})

    html = build_html(records, teams, roles, managers)
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding='utf-8')
    print(f'✓ Saved: {OUT_HTML}')
    print(f'  {len(records)} players total')
    print(f'  {sum(1 for r in records if r["manager"] not in ("Unsold","Not in auction"))} drafted')
    print(f'  {sum(1 for r in records if r["manager"] == "Unsold")} unsold')
    print(f'  {sum(1 for r in records if r["manager"] == "Not in auction")} not listed in Boston file')


def build_html(records, teams, roles, managers) -> str:
    data_json = json.dumps(records)
    team_opts = ''.join(f'<option value="{t}">{t}</option>' for t in teams)
    role_opts = ''.join(f'<option value="{r}">{r}</option>' for r in roles)
    mgr_opts = ''.join(f'<option value="{m}">{m}</option>' for m in managers)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>IPL 2026 Fantasy — Player Rankings</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 0; padding: 20px; background: #f7f8fa; color: #222; }}
  h1 {{ margin: 0 0 4px 0; font-size: 22px; }}
  .subtitle {{ color: #666; margin-bottom: 16px; font-size: 13px; }}
  .filters {{ background: white; padding: 14px; border-radius: 8px;
              box-shadow: 0 1px 3px rgba(0,0,0,.08); margin-bottom: 16px;
              display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }}
  .filters label {{ font-size: 12px; color: #555; display: flex;
                     flex-direction: column; gap: 4px; }}
  .filters select, .filters input {{
    padding: 6px 10px; border: 1px solid #ccc; border-radius: 5px;
    font-size: 13px; min-width: 140px;
  }}
  .filters input {{ min-width: 200px; }}
  .stats {{ font-size: 12px; color: #666; margin-left: auto; }}
  table {{ width: 100%; border-collapse: collapse; background: white;
           border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
  thead {{ background: #2c3e50; color: white; }}
  th {{ padding: 10px 12px; text-align: left; font-size: 12px;
        font-weight: 600; text-transform: uppercase; letter-spacing: .3px;
        cursor: pointer; user-select: none; white-space: nowrap; }}
  th:hover {{ background: #34495e; }}
  th .arrow {{ opacity: .4; margin-left: 4px; }}
  th.active .arrow {{ opacity: 1; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #eee; font-size: 13px; }}
  tbody tr:hover {{ background: #f5f7fa; }}
  .rank {{ font-weight: 600; color: #777; width: 40px; }}
  .player {{ font-weight: 600; }}
  .team {{ font-size: 12px; color: #555; }}
  .role {{ font-size: 11px; padding: 2px 6px; border-radius: 3px;
           background: #e8eaf0; color: #333; }}
  .role-BAT {{ background: #dbeafe; color: #1e40af; }}
  .role-BOWL {{ background: #fee2e2; color: #991b1b; }}
  .role-AR {{ background: #fef3c7; color: #92400e; }}
  .role-WK {{ background: #ede9fe; color: #5b21b6; }}
  .mgr {{ font-size: 12px; }}
  .mgr-unsold {{ color: #999; font-style: italic; }}
  .mgr-none {{ color: #bbb; font-style: italic; }}
  .points {{ font-weight: 600; color: #111; text-align: right; }}
  .pts-source {{ font-size: 10px; color: #999; margin-left: 4px; }}
  .hidden {{ display: none; }}
</style>
</head>
<body>
<h1>IPL 2026 Fantasy — Player Rankings</h1>
<div class="subtitle">
  Season-to-date points (24 matches). App points used where available;
  computed points otherwise.
</div>

<div class="filters">
  <label>Search
    <input type="text" id="search" placeholder="Player name...">
  </label>
  <label>IPL team
    <select id="team">
      <option value="">All teams</option>
      {team_opts}
    </select>
  </label>
  <label>Role
    <select id="role">
      <option value="">All roles</option>
      {role_opts}
    </select>
  </label>
  <label>Owner
    <select id="manager">
      <option value="">All</option>
      <option value="__DRAFTED__">Drafted (any)</option>
      <option value="Unsold">Unsold</option>
      <option value="Not in auction">Not in auction</option>
      <option disabled>──────</option>
      {mgr_opts}
    </select>
  </label>
  <span class="stats" id="stats"></span>
</div>

<table id="tbl">
  <thead>
    <tr>
      <th data-key="rank">#<span class="arrow">▲</span></th>
      <th data-key="player">Player<span class="arrow"></span></th>
      <th data-key="team">IPL Team<span class="arrow"></span></th>
      <th data-key="role">Role<span class="arrow"></span></th>
      <th data-key="matches">M<span class="arrow"></span></th>
      <th data-key="points" class="active">Points<span class="arrow">▼</span></th>
      <th data-key="manager">Owner<span class="arrow"></span></th>
      <th data-key="price">Price (₹L)<span class="arrow"></span></th>
    </tr>
  </thead>
  <tbody id="tbody"></tbody>
</table>

<script>
const DATA = {data_json};
let sortKey = 'points';
let sortDir = -1;  // -1 = desc, 1 = asc

const $search = document.getElementById('search');
const $team   = document.getElementById('team');
const $role   = document.getElementById('role');
const $mgr    = document.getElementById('manager');
const $tbody  = document.getElementById('tbody');
const $stats  = document.getElementById('stats');

function roleClass(role) {{
  const r = (role || '').toLowerCase();
  if (r.includes('wicket')) return 'role-WK';
  if (r.includes('bowl'))   return 'role-BOWL';
  if (r.includes('all'))    return 'role-AR';
  if (r.includes('bat'))    return 'role-BAT';
  return '';
}}

function mgrClass(m) {{
  if (m === 'Unsold') return 'mgr-unsold';
  if (m === 'Not in auction') return 'mgr-none';
  return '';
}}

function render() {{
  const q = $search.value.trim().toLowerCase();
  const t = $team.value;
  const ro = $role.value;
  const mg = $mgr.value;

  const filtered = DATA.filter(r => {{
    if (q && !r.player.toLowerCase().includes(q)) return false;
    if (t && r.team !== t) return false;
    if (ro && r.role !== ro) return false;
    if (mg === '__DRAFTED__') {{
      if (r.manager === 'Unsold' || r.manager === 'Not in auction' || !r.manager) return false;
    }} else if (mg && r.manager !== mg) return false;
    return true;
  }});

  filtered.sort((a, b) => {{
    let av = a[sortKey], bv = b[sortKey];
    if (typeof av === 'string') av = av.toLowerCase();
    if (typeof bv === 'string') bv = bv.toLowerCase();
    if (av < bv) return -1 * sortDir;
    if (av > bv) return  1 * sortDir;
    return 0;
  }});

  $tbody.innerHTML = filtered.map(r => {{
    const priceStr = r.price > 0 ? r.price.toFixed(0) : '—';
    const srcHint = r.app_points !== null ? '' :
      `<span class="pts-source" title="Computed under published rules (app score unavailable)">*</span>`;
    return `
      <tr>
        <td class="rank">${{r.rank}}</td>
        <td class="player">${{r.player}}</td>
        <td class="team">${{r.team || '—'}}</td>
        <td><span class="role ${{roleClass(r.role)}}">${{r.role || '—'}}</span></td>
        <td>${{r.matches}}</td>
        <td class="points">${{r.points}}${{srcHint}}</td>
        <td class="mgr ${{mgrClass(r.manager)}}">${{r.manager || '—'}}</td>
        <td>${{priceStr}}</td>
      </tr>`;
  }}).join('');

  $stats.textContent = `${{filtered.length}} of ${{DATA.length}} players`;
}}

document.querySelectorAll('th').forEach(th => {{
  th.addEventListener('click', () => {{
    const k = th.dataset.key;
    if (sortKey === k) sortDir *= -1;
    else {{ sortKey = k; sortDir = (k === 'player' || k === 'team' || k === 'role' || k === 'manager') ? 1 : -1; }}
    document.querySelectorAll('th').forEach(x => {{
      x.classList.remove('active');
      x.querySelector('.arrow').textContent = '';
    }});
    th.classList.add('active');
    th.querySelector('.arrow').textContent = sortDir === -1 ? '▼' : '▲';
    render();
  }});
}});

[$search, $team, $role, $mgr].forEach(el => el.addEventListener('input', render));
render();
</script>
</body>
</html>
"""


if __name__ == '__main__':
    main()
