"""
Fantasy Auction Valuation System.

Computes optimal auction credit values for a fantasy cricket league:
  - 8 managers, 100 credits each
  - 250 IPL 2026 squad players available
  - Each manager drafts 10 players via English auction
  - Scoring: Dream11 fantasy points (total season)

Valuation approach:
  1. Project total season FP (avg × 14 matches)
  2. VORP: value over the 80th-best player (replacement level)
  3. Positional scarcity adjustment (scarce roles get premium)
  4. Consistency premium (low-variance players worth more in season-long)
  5. Convert to auction credits (Rotisserie-style dollar values)
  6. Game theory: bid ranges, nomination strategy, budget allocation

OUTPUT: results/FantasyProjections/fantasy_auction/
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

sys.path.insert(0, str(Path(__file__).parent.parent / 'WARprojections'))
from utils import save_dataframe

# ── Configuration ─────────────────────────────────────────────────────────

N_TEAMS = 8
BUDGET_PER_TEAM = 100
ROSTER_SIZE = 10
MATCHES_PER_SEASON = 14
MIN_BID = 1

# Demand per role (how many of each role across all teams)
DEMAND_PER_ROLE = {'BAT': 24, 'BOWL': 24, 'AR': 16, 'WK': 16}  # total = 80

# Role mapping: various sources → simplified 4-role system
ROLE_MAP = {
    # From player_metadata role_category
    'Pacer': 'BOWL', 'Spinner': 'BOWL',
    'Top-order Batter': 'BAT', 'Middle-order Batter': 'BAT',
    'Allrounder': 'AR',
    'Wicketkeeper': 'WK',
    # From enriched data playing_role
    'Batter': 'BAT', 'Top-order batter': 'BAT', 'Opening batter': 'BAT',
    'Middle-order batter': 'BAT', 'Middle order Batter': 'BAT',
    'Bowler': 'BOWL',
    'Batting allrounder': 'AR', 'Bowling allrounder': 'AR',
    'Wicketkeeper batter': 'WK', 'Wicketkeeper Batter': 'WK',
}

ROLE_COLORS = {
    'BAT': '#e74c3c', 'BOWL': '#3498db', 'AR': '#2ecc71', 'WK': '#f39c12',
}

TIER_COLORS = {
    'Tier 1 — Franchise': '#8e44ad',
    'Tier 2 — Premium': '#2980b9',
    'Tier 3 — Solid': '#27ae60',
    'Tier 4 — Bargain': '#f39c12',
    'Tier 5 — Min Bid': '#95a5a6',
}


# ── Data Loading ──────────────────────────────────────────────────────────

def load_and_merge_data():
    """Load squad projections, metadata, and historical stats."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    fp_results = project_root / 'results' / 'FantasyProjections'

    # Squad projections (250 players)
    proj = pd.read_csv(fp_results / 'squad_2026' / 'squad_fantasy_projections_2026.csv')

    # Metadata for dob, player type from enriched squads
    enriched = pd.read_csv(data_dir / 'ipl_2026_squads_enriched.csv')
    enriched_cols = ['Player', 'Team', 'playing_role', 'Player_Type', 'dob', 'country',
                     'batting_style', 'bowling_style', 'cricsheet_id']
    enriched_slim = enriched[[c for c in enriched_cols if c in enriched.columns]].drop_duplicates(
        subset=['Player', 'Team'])

    # Merge (projections already have Player, Team)
    df = proj.merge(enriched_slim, on=['Player', 'Team'], how='left', suffixes=('', '_enr'))

    # Player metadata role via cricsheet_id (best coverage: 188/250)
    player_meta = pd.read_csv(data_dir / 'player_metadata.csv')
    meta_id_roles = player_meta[['player_id', 'role_category']].drop_duplicates(subset='player_id')
    meta_id_roles = meta_id_roles.rename(columns={'player_id': 'cricsheet_id', 'role_category': 'meta_role'})
    df = df.merge(meta_id_roles, on='cricsheet_id', how='left')

    # Also try name-based metadata match as fallback
    meta_name_roles = player_meta[['player_name', 'role_category']].drop_duplicates(subset='player_name')
    meta_name_roles = meta_name_roles.rename(columns={'player_name': 'Player', 'role_category': 'meta_role_name'})
    df = df.merge(meta_name_roles, on='Player', how='left')

    # Role from fantasy features (derived from batting/bowling point shares)
    features = pd.read_csv(data_dir / 'fantasy_features.csv')
    latest_roles = features[features['season'] == features['season'].max()][
        ['player_name', 'role']].drop_duplicates(subset='player_name')
    latest_roles = latest_roles.rename(columns={'player_name': 'Player', 'role': 'feature_role'})
    df = df.merge(latest_roles, on='Player', how='left')

    # Resolve role with priority:
    # 1. meta_role (via cricsheet_id) — best: 188 players, proper WK detection
    # 2. meta_role_name (via name) — fallback for unmatched IDs
    # 3. feature_role (batting/bowling shares) — for players in fantasy features
    # 4. playing_role from enriched data
    # 5. BAT default
    def resolve_role(row):
        for col in ['meta_role', 'meta_role_name']:
            val = row.get(col)
            if pd.notna(val) and val in ROLE_MAP:
                return ROLE_MAP[val]
        fr = row.get('feature_role')
        if pd.notna(fr) and fr in ('BAT', 'BOWL', 'AR', 'WK'):
            return fr
        pr = row.get('playing_role')
        if pd.notna(pr) and pr in ROLE_MAP:
            return ROLE_MAP[pr]
        return 'BAT'

    df['role'] = df.apply(resolve_role, axis=1)

    # Historical volatility (most recent season per player)
    season_hist = pd.read_csv(data_dir / 'fantasy_points_per_season.csv')
    # Get latest season per player
    latest_season = season_hist.sort_values('season').groupby('player_name').last().reset_index()
    vol = latest_season[['player_name', 'avg_fantasy_pts', 'std_fantasy_pts', 'matches']].rename(
        columns={'player_name': 'Player', 'avg_fantasy_pts': 'hist_avg_fp',
                 'std_fantasy_pts': 'hist_std_fp', 'matches': 'hist_matches'})
    df = df.merge(vol, on='Player', how='left')

    # 2024 and 2025 actual fantasy points per match (for reference on dashboard)
    for yr in [2024, 2025]:
        yr_data = season_hist[season_hist['season'] == yr][
            ['player_id', 'avg_fantasy_pts', 'matches']].rename(
            columns={'player_id': 'cricsheet_id',
                     'avg_fantasy_pts': f'actual_fp_{yr}',
                     'matches': f'matches_{yr}'})
        yr_data = yr_data.sort_values(f'matches_{yr}', ascending=False).drop_duplicates(
            subset='cricsheet_id', keep='first')
        df = df.merge(yr_data, on='cricsheet_id', how='left')
    matched_24 = df['actual_fp_2024'].notna().sum()
    matched_25 = df['actual_fp_2025'].notna().sum()
    print(f"Historical FP matched: 2024={matched_24}, 2025={matched_25} (of {len(df)})")

    # Compute projected total season FP
    df['projected_total_fp'] = df['projected_avg_fp_2026'] * MATCHES_PER_SEASON

    # Age
    if 'dob' in df.columns:
        df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
        df['age'] = 2026 - df['dob'].dt.year
        df.loc[(df['age'] < 15) | (df['age'] > 50), 'age'] = np.nan

    print(f"Loaded {len(df)} players")
    print(f"Role distribution: {df['role'].value_counts().to_dict()}")
    return df


# ── Valuation Engine ──────────────────────────────────────────────────────

def compute_replacement_level(df, n_drafted=None):
    """The projected_total_fp of the Nth-best player (replacement level)."""
    if n_drafted is None:
        n_drafted = N_TEAMS * ROSTER_SIZE
    sorted_fp = df['projected_total_fp'].sort_values(ascending=False).values
    if n_drafted >= len(sorted_fp):
        return sorted_fp[-1]
    return sorted_fp[n_drafted - 1]  # 80th best player


def compute_vorp(df, replacement_level):
    """Value Over Replacement Player."""
    df = df.copy()
    df['replacement_level'] = replacement_level
    df['VORP'] = (df['projected_total_fp'] - replacement_level).clip(lower=0)
    return df


def compute_positional_scarcity(df, demand=None):
    """Scarcity multiplier per role based on supply vs demand."""
    if demand is None:
        demand = DEMAND_PER_ROLE

    df = df.copy()
    scarcity = {}

    for role, dem in demand.items():
        role_players = df[df['role'] == role]
        supply_above_repl = (role_players['VORP'] > 0).sum()
        supply_above_repl = max(supply_above_repl, 1)  # avoid div by zero
        scarcity[role] = dem / supply_above_repl

    # Normalize so average multiplier = 1.0
    avg_scarcity = np.mean(list(scarcity.values()))
    for role in scarcity:
        scarcity[role] /= avg_scarcity

    df['scarcity_multiplier'] = df['role'].map(scarcity).fillna(1.0)
    df['scarcity_VORP'] = df['VORP'] * df['scarcity_multiplier']

    print(f"\nPositional Scarcity Multipliers:")
    for role, mult in sorted(scarcity.items()):
        dem = demand[role]
        sup = (df[df['role'] == role]['VORP'] > 0).sum()
        print(f"  {role:4s}: demand={dem:2d}, supply_above_repl={sup:2d}, multiplier={mult:.3f}")

    return df


def compute_consistency_adjustment(df):
    """Premium for consistent players, discount for volatile ones."""
    df = df.copy()

    # Coefficient of variation from historical data
    df['cv'] = np.where(
        (df['hist_avg_fp'] > 0) & df['hist_std_fp'].notna(),
        df['hist_std_fp'] / df['hist_avg_fp'],
        np.nan
    )

    # consistency_score: low CV → premium (up to 1.15), high CV → discount (down to 0.85)
    # Formula: 1 - min(cv/2, 0.15)  → range [0.85, 1.0] for players with data
    # But we also want to reward very consistent players above 1.0
    # Use: 1.15 - min(cv, 0.30)  → cv=0 → 1.15, cv=0.3+ → 0.85
    df['consistency_score'] = np.where(
        df['cv'].notna(),
        (1.15 - df['cv'].clip(upper=0.30)),
        1.0  # no data → neutral
    )

    df['adjusted_VORP'] = df['scarcity_VORP'] * df['consistency_score']

    n_with_cv = df['cv'].notna().sum()
    print(f"\nConsistency adjustment: {n_with_cv} players with historical data")
    print(f"  CV range: {df['cv'].min():.3f} - {df['cv'].max():.3f}")
    print(f"  Consistency score range: {df['consistency_score'].min():.3f} - {df['consistency_score'].max():.3f}")

    return df


def compute_auction_values(df, n_teams=N_TEAMS, budget=BUDGET_PER_TEAM, min_bid=MIN_BID):
    """Convert adjusted VORP to auction credit values."""
    df = df.copy()

    total_credits = n_teams * budget
    n_drafted = n_teams * ROSTER_SIZE
    reserve_credits = n_drafted * min_bid
    distributable = total_credits - reserve_credits

    total_adj_vorp = df.loc[df['adjusted_VORP'] > 0, 'adjusted_VORP'].sum()

    df['auction_value'] = np.where(
        df['adjusted_VORP'] > 0,
        (df['adjusted_VORP'] / total_adj_vorp) * distributable + min_bid,
        min_bid
    )

    # Round to nearest 0.5 for usability
    df['auction_value'] = (df['auction_value'] * 2).round() / 2

    # Ensure minimum
    df['auction_value'] = df['auction_value'].clip(lower=min_bid)

    print(f"\nCredit Conversion:")
    print(f"  Total pool: {total_credits} credits ({n_teams} × {budget})")
    print(f"  Reserved: {reserve_credits} credits ({n_drafted} × {min_bid} min bid)")
    print(f"  Distributable: {distributable} credits")
    print(f"  Sum of top {n_drafted} auction values: {df.nlargest(n_drafted, 'auction_value')['auction_value'].sum():.1f}")

    return df


# ── Tier Classification ───────────────────────────────────────────────────

def classify_tiers(df):
    """Assign tier based on auction value."""
    df = df.copy()

    def tier(v):
        if v >= 15:
            return 'Tier 1 — Franchise'
        elif v >= 8:
            return 'Tier 2 — Premium'
        elif v >= 4:
            return 'Tier 3 — Solid'
        elif v >= 2:
            return 'Tier 4 — Bargain'
        else:
            return 'Tier 5 — Min Bid'

    df['tier'] = df['auction_value'].apply(tier)
    print(f"\nTier Distribution:")
    for t in ['Tier 1 — Franchise', 'Tier 2 — Premium', 'Tier 3 — Solid',
              'Tier 4 — Bargain', 'Tier 5 — Min Bid']:
        n = (df['tier'] == t).sum()
        print(f"  {t}: {n} players")

    return df


# ── Game Theory ───────────────────────────────────────────────────────────

def compute_bid_ranges(df):
    """Floor / fair / max bid prices."""
    df = df.copy()
    df['floor_price'] = (df['auction_value'] * 0.7).round(1).clip(lower=MIN_BID)
    df['fair_price'] = df['auction_value']
    df['max_bid'] = (df['auction_value'] * 1.3).round(1).clip(lower=MIN_BID)
    return df


def compute_inflation_risk(df):
    """Estimate how likely a player is to be overbid."""
    df = df.copy()

    def inflation(row):
        # High: Tier 1, Indian stars, well-known
        if row['tier'] == 'Tier 1 — Franchise':
            return 'High'
        if row['tier'] == 'Tier 2 — Premium' and row.get('Player_Type') == 'Indian':
            return 'High'
        if row['tier'] == 'Tier 2 — Premium':
            return 'Medium'
        if row['tier'] == 'Tier 3 — Solid':
            return 'Medium'
        if row['prediction_source'] == 'Replacement':
            return 'Low'
        return 'Low'

    df['inflation_risk'] = df.apply(inflation, axis=1)
    return df


def generate_nomination_strategy(df):
    """Identify players to nominate to drain opponent budgets, and quiet value picks."""
    # Nominate to inflate: expensive players likely to trigger bidding wars
    nominate = df[df['tier'].isin(['Tier 1 — Franchise', 'Tier 2 — Premium'])].nlargest(
        15, 'auction_value').copy()
    nominate['strategy'] = 'Nominate to inflate'
    nominate['reasoning'] = nominate.apply(
        lambda r: f"Auction value {r['auction_value']:.1f} credits. "
                  f"High-profile {r['role']} — will trigger bidding war and drain budgets.",
        axis=1
    )

    # Quiet value: high value + low inflation risk
    quiet = df[(df['inflation_risk'] == 'Low') & (df['auction_value'] >= 3)].nlargest(
        10, 'auction_value').copy()
    quiet['strategy'] = 'Quiet value'
    quiet['reasoning'] = quiet.apply(
        lambda r: f"Auction value {r['auction_value']:.1f} credits but low profile — "
                  f"likely to go near floor price ({r['floor_price']:.1f}).",
        axis=1
    )

    strategy = pd.concat([nominate, quiet], ignore_index=True)
    return strategy[['Player', 'Team', 'role', 'auction_value', 'floor_price', 'max_bid',
                      'tier', 'strategy', 'reasoning']]


# ── Cheat Sheet ───────────────────────────────────────────────────────────

def generate_cheat_sheet(df):
    """Per-role ranked list with bid prices."""
    rows = []
    for role in ['BAT', 'BOWL', 'AR', 'WK']:
        role_df = df[df['role'] == role].sort_values('auction_value', ascending=False)
        for rank, (_, r) in enumerate(role_df.iterrows(), 1):
            notes = ''
            if r['inflation_risk'] == 'High':
                notes = 'Expect overbidding'
            elif r['inflation_risk'] == 'Low' and r['auction_value'] >= 3:
                notes = 'Potential bargain'
            if r['prediction_source'] == 'Replacement':
                notes = 'No historical data'

            rows.append({
                'Role': role,
                'Rank_in_Role': rank,
                'Player': r['Player'],
                'IPL_Team': r['Team'],
                'Auction_Value': r['auction_value'],
                'Floor': r['floor_price'],
                'Max': r['max_bid'],
                'Tier': r['tier'],
                'Proj_FP_per_Match': r['projected_avg_fp_2026'],
                'Notes': notes,
            })

    return pd.DataFrame(rows)


# ── Dashboard HTML ────────────────────────────────────────────────────────

def build_dashboard_html(df):
    """Build fully interactive Plotly HTML dashboard with client-side recalculation."""

    # Serialize player data as JSON for the JS valuation engine
    player_records = []
    for _, row in df.iterrows():
        fp24 = row.get('actual_fp_2024')
        fp25 = row.get('actual_fp_2025')
        player_records.append({
            'name': row['Player'],
            'team': row['Team'],
            'role': row['role'],
            'proj_avg_fp': round(row['projected_avg_fp_2026'], 2),
            'consistency': round(row.get('consistency_score', 1.0), 4),
            'source': row.get('prediction_source', ''),
            'ptype': row.get('Player_Type', ''),
            'fp24': round(fp24, 1) if pd.notna(fp24) else None,
            'fp25': round(fp25, 1) if pd.notna(fp25) else None,
        })
    players_json = json.dumps(player_records)

    html = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Fantasy Auction Valuation Dashboard — IPL 2026</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           background: #f5f5f5; margin: 0; padding: 20px; color: #333; }}
    .header {{ text-align: center; padding: 30px 20px; background: linear-gradient(135deg, #2c3e50, #3498db);
               color: white; border-radius: 12px; margin-bottom: 30px; }}
    .header h1 {{ margin: 0; font-size: 2.2em; }}
    .header p {{ margin: 8px 0 0; opacity: 0.9; font-size: 1.1em; }}
    .header #subtitle {{ }}
    .stats-bar {{ display: flex; gap: 15px; flex-wrap: wrap; margin-bottom: 25px; justify-content: center; }}
    .stat-card {{ background: white; padding: 15px 25px; border-radius: 8px;
                  box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center; min-width: 140px; }}
    .stat-card .number {{ font-size: 1.8em; font-weight: 700; color: #2c3e50; }}
    .stat-card .label {{ font-size: 0.85em; color: #666; margin-top: 4px; }}
    .section {{ background: white; padding: 25px; border-radius: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 25px; }}
    .section h2 {{ margin-top: 0; color: #2c3e50; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }}
    .chart-container {{ margin: 15px 0; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th {{ background: #2c3e50; color: white; padding: 10px 8px; text-align: left; cursor: pointer; }}
    th:hover {{ background: #34495e; }}
    td {{ padding: 8px; border-bottom: 1px solid #ecf0f1; }}
    tr:hover {{ background: #f8f9fa; }}
    .role-badge {{ padding: 3px 8px; border-radius: 4px; color: white; font-size: 11px; font-weight: 600; display:inline-block; }}
    .tier-badge {{ padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; display:inline-block; }}
    .filters {{ margin-bottom: 15px; display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }}
    .filters select, .filters input[type=text] {{ padding: 6px 10px; border: 1px solid #ddd; border-radius: 5px; }}
    .filters button {{ padding: 6px 14px; background: #3498db; color: white; border: none;
                       border-radius: 5px; cursor: pointer; }}
    .filters button:hover {{ background: #2980b9; }}
    #row-count {{ color: #666; font-size: 13px; }}
    .role-section {{ margin-top: 20px; }}
    .role-section h3 {{ cursor: pointer; padding: 10px; background: #ecf0f1; border-radius: 6px;
                        margin-bottom: 5px; }}
    .role-section h3:hover {{ background: #d5dbdb; }}
    .cfg-panel {{ display: flex; gap: 20px; flex-wrap: wrap; align-items: end; }}
    .cfg-panel label {{ font-size: 14px; font-weight: 500; }}
    .cfg-panel input[type=number] {{ padding: 6px 8px; border: 1px solid #ccc; border-radius: 5px;
                                      font-size: 14px; }}
    .cfg-panel input:disabled {{ background: #eee; color: #aaa; }}
    .recalc-btn {{ padding: 10px 28px; background: #e74c3c; color: white; border: none;
                   border-radius: 6px; font-weight: bold; font-size: 15px; cursor: pointer;
                   transition: background 0.2s; }}
    .recalc-btn:hover {{ background: #c0392b; }}
    .cfg-row {{ display: flex; gap: 18px; flex-wrap: wrap; align-items: center; margin-top: 12px; }}
    .cfg-check {{ font-size: 14px; cursor: pointer; }}
</style>
</head>
<body>

<div class="header">
    <h1>Fantasy Auction Valuation Dashboard</h1>
    <p id="subtitle">IPL 2026</p>
</div>

<!-- ── Configuration Panel ─────────────────────────────────────────── -->
<div class="section" id="config-panel">
    <h2>League Settings</h2>
    <div class="cfg-panel">
        <label>Teams: <input type="number" id="cfg-teams" value="8" min="2" max="20" style="width:60px"></label>
        <label>Players/Team: <input type="number" id="cfg-roster" value="10" min="1" max="25" style="width:60px"></label>
        <label>Budget/Team: <input type="number" id="cfg-budget" value="100" min="10" max="10000" style="width:80px"></label>
    </div>
    <div class="cfg-row">
        <label class="cfg-check"><input type="checkbox" id="cfg-no-roles" onchange="toggleRoles()"> No role requirements (pure value)</label>
    </div>
    <div class="cfg-row" id="role-inputs">
        <label>Min BAT: <input type="number" id="cfg-bat" value="3" min="0" max="15" class="role-input" style="width:55px"></label>
        <label>Min BOWL: <input type="number" id="cfg-bowl" value="3" min="0" max="15" class="role-input" style="width:55px"></label>
        <label>Min AR: <input type="number" id="cfg-ar" value="2" min="0" max="15" class="role-input" style="width:55px"></label>
        <label>Min WK: <input type="number" id="cfg-wk" value="2" min="0" max="15" class="role-input" style="width:55px"></label>
    </div>
    <div class="cfg-row">
        <button class="recalc-btn" onclick="recalculate()">Recalculate Valuations</button>
        <span id="cfg-warning" style="color:#e74c3c;font-size:13px;"></span>
    </div>
</div>

<!-- ── Stats Bar ───────────────────────────────────────────────────── -->
<div class="stats-bar">
    <div class="stat-card"><div class="number" id="stat-available">250</div><div class="label">Players Available</div></div>
    <div class="stat-card"><div class="number" id="stat-drafted">80</div><div class="label">Will Be Drafted</div></div>
    <div class="stat-card"><div class="number" id="stat-credits">800</div><div class="label">Total Credits</div></div>
    <div class="stat-card"><div class="number" id="stat-franchise">0</div><div class="label">Franchise Players</div></div>
    <div class="stat-card"><div class="number" id="stat-topvalue">0</div><div class="label">Top Value (Credits)</div></div>
</div>

<!-- ── Charts ──────────────────────────────────────────────────────── -->
<div class="section"><h2>Top 40 Players by Auction Value</h2><div class="chart-container" id="chart1"></div></div>
<div class="section"><h2>Tier Distribution by Role</h2><div class="chart-container" id="chart2"></div></div>
<div class="section"><h2>Positional Scarcity: Supply vs Demand</h2><div class="chart-container" id="chart3"></div></div>
<div class="section"><h2>Auction Value vs Projected Fantasy Points</h2><div class="chart-container" id="chart4"></div></div>

<!-- ── Strategy Cards ──────────────────────────────────────────────── -->
<div class="section">
<h2>Budget Allocation Strategies</h2>
<div style="display:flex;gap:20px;flex-wrap:wrap;">
<div style="flex:1;min-width:280px;border:2px solid #8e44ad;border-radius:10px;padding:20px;">
    <h3 style="color:#8e44ad;">Stars & Scrubs</h3>
    <p><strong>Risk: High variance</strong></p>
    <table style="width:100%;font-size:14px;"><tr><td>2-3 Tier 1-2</td><td style="text-align:right"><strong>55-65%</strong></td></tr><tr><td>2-3 Tier 3</td><td style="text-align:right">20-25%</td></tr><tr><td>4-5 Tier 4-5</td><td style="text-align:right">10-15%</td></tr></table>
    <p style="font-size:13px;color:#666;margin-top:10px;">Bet big on elite talent. High ceiling, high floor risk.</p>
</div>
<div style="flex:1;min-width:280px;border:2px solid #2980b9;border-radius:10px;padding:20px;">
    <h3 style="color:#2980b9;">Balanced</h3>
    <p><strong>Risk: Moderate</strong></p>
    <table style="width:100%;font-size:14px;"><tr><td>1-2 Tier 1-2</td><td style="text-align:right"><strong>25-35%</strong></td></tr><tr><td>4-5 Tier 3</td><td style="text-align:right">40-50%</td></tr><tr><td>3-4 Tier 4-5</td><td style="text-align:right">15-25%</td></tr></table>
    <p style="font-size:13px;color:#666;margin-top:10px;">Anchor star + strong supporting cast. Most forgiving.</p>
</div>
<div style="flex:1;min-width:280px;border:2px solid #27ae60;border-radius:10px;padding:20px;">
    <h3 style="color:#27ae60;">Value Hunter</h3>
    <p><strong>Risk: Low variance</strong></p>
    <table style="width:100%;font-size:14px;"><tr><td>0-1 Tier 2</td><td style="text-align:right"><strong>0-15%</strong></td></tr><tr><td>5-6 Tier 3</td><td style="text-align:right">50-60%</td></tr><tr><td>3-4 Tier 4</td><td style="text-align:right">25-40%</td></tr></table>
    <p style="font-size:13px;color:#666;margin-top:10px;">Let others overpay. Accumulate depth and consistency.</p>
</div>
</div></div>

<!-- ── Master Table ────────────────────────────────────────────────── -->
<div class="section" id="master-table-section">
    <h2>All Players — Master Valuation Table</h2>
    <div class="filters">
        <label>Role: <select id="f-role" onchange="filterMaster()">
            <option value="">All Roles</option>
            <option value="BAT">BAT</option><option value="BOWL">BOWL</option>
            <option value="AR">AR</option><option value="WK">WK</option>
        </select></label>
        <label>Tier: <select id="f-tier" onchange="filterMaster()">
            <option value="">All Tiers</option>
            <option value="Franchise">Franchise</option><option value="Premium">Premium</option>
            <option value="Solid">Solid</option><option value="Bargain">Bargain</option>
            <option value="Min Bid">Min Bid</option>
        </select></label>
        <label>Search: <input type="text" id="f-search" oninput="filterMaster()" placeholder="Player name..."></label>
        <button onclick="document.getElementById('f-role').value='';document.getElementById('f-tier').value='';document.getElementById('f-search').value='';filterMaster();">Reset</button>
        <span id="row-count"></span>
    </div>
    <table id="master-tbl">
    <thead><tr>
        <th onclick="doSort(0)">#</th>
        <th onclick="doSort(1)">Player</th>
        <th onclick="doSort(2)">IPL Team</th>
        <th onclick="doSort(3)">Role</th>
        <th onclick="doSort(4,'num')">Auction Value</th>
        <th onclick="doSort(5,'num')">Floor</th>
        <th onclick="doSort(6,'num')">Max Bid</th>
        <th onclick="doSort(7)">Tier</th>
        <th onclick="doSort(8,'num')">Proj FP/Match</th>
        <th onclick="doSort(9,'num')">2024 FP/M</th>
        <th onclick="doSort(10,'num')">2025 FP/M</th>
        <th onclick="doSort(11,'num')">Season FP</th>
        <th onclick="doSort(12)">Inflation</th>
        <th onclick="doSort(13)">Source</th>
    </tr></thead>
    <tbody id="tbl-body"></tbody>
    </table>
</div>

<!-- ── Per-Role Rankings ───────────────────────────────────────────── -->
<div class="section"><h2>Per-Role Rankings</h2><div id="role-lists"></div></div>

<!-- ════════════════════════════════════════════════════════════════════
     JAVASCRIPT VALUATION ENGINE
     ════════════════════════════════════════════════════════════════════ -->
<script>
var PLAYERS = {players_json};
var ROLE_COLORS = {json.dumps(ROLE_COLORS, ensure_ascii=False)};
var TIER_COLORS = {json.dumps(TIER_COLORS, ensure_ascii=False)};
var MATCHES = 14;
var MIN_BID = 1;

// ── Helpers ──────────────────────────────────────────────────────────
function $(id) {{ return document.getElementById(id); }}

function getTier(v) {{
    if (v >= 15) return 'Franchise';
    if (v >= 8) return 'Premium';
    if (v >= 4) return 'Solid';
    if (v >= 2) return 'Bargain';
    return 'Min Bid';
}}

function getInflation(tier, ptype, source) {{
    if (tier === 'Franchise') return 'High';
    if (tier === 'Premium' && ptype === 'Indian') return 'High';
    if (tier === 'Premium') return 'Medium';
    if (tier === 'Solid') return 'Medium';
    if (source === 'Replacement') return 'Low';
    return 'Low';
}}

var tierFullName = {{
    'Franchise': 'Tier 1 — Franchise', 'Premium': 'Tier 2 — Premium',
    'Solid': 'Tier 3 — Solid', 'Bargain': 'Tier 4 — Bargain', 'Min Bid': 'Tier 5 — Min Bid'
}};

// ── Toggle section expand/collapse ───────────────────────────────────
function toggleSection(el) {{
    var next = el.nextElementSibling;
    next.style.display = next.style.display === 'none' ? '' : 'none';
}}

// ── Toggle role inputs ───────────────────────────────────────────────
function toggleRoles() {{
    var dis = $('cfg-no-roles').checked;
    document.querySelectorAll('.role-input').forEach(function(el) {{ el.disabled = dis; }});
    $('role-inputs').style.opacity = dis ? 0.4 : 1;
}}

// ── Main recalculation ───────────────────────────────────────────────
function recalculate() {{
  try {{
    var btn = document.querySelector('.recalc-btn');
    btn.textContent = 'Recalculating...';
    btn.disabled = true;

    var nTeams = +$('cfg-teams').value || 8;
    var roster = +$('cfg-roster').value || 10;
    var budget = +$('cfg-budget').value || 100;
    var noRoles = $('cfg-no-roles').checked;
    var nDrafted = nTeams * roster;
    var totalCredits = nTeams * budget;

    // Validate
    if (nDrafted > PLAYERS.length) {{
        $('cfg-warning').textContent = 'Warning: ' + nDrafted + ' slots but only ' + PLAYERS.length + ' players available.';
        nDrafted = PLAYERS.length;
    }} else {{
        $('cfg-warning').textContent = '';
    }}

    // Role demands
    var demand = {{}};
    if (noRoles) {{
        demand = {{BAT: 0, BOWL: 0, AR: 0, WK: 0}};
    }} else {{
        var bat = +$('cfg-bat').value || 0, bowl = +$('cfg-bowl').value || 0;
        var ar = +$('cfg-ar').value || 0, wk = +$('cfg-wk').value || 0;
        var assigned = (bat + bowl + ar + wk) * nTeams;
        if (assigned > nDrafted) {{
            $('cfg-warning').textContent = 'Warning: role minimums (' + assigned + ') exceed roster slots (' + nDrafted + '). Excess ignored.';
        }}
        demand = {{BAT: bat * nTeams, BOWL: bowl * nTeams, AR: ar * nTeams, WK: wk * nTeams}};
    }}

    // 1. Total FP
    PLAYERS.forEach(function(p) {{ p.total_fp = p.proj_avg_fp * MATCHES; }});

    // 2. Replacement level
    var sorted = PLAYERS.slice().sort(function(a, b) {{ return b.total_fp - a.total_fp; }});
    var replIdx = Math.min(nDrafted - 1, sorted.length - 1);
    var replLevel = sorted[replIdx].total_fp;

    // 3. VORP
    PLAYERS.forEach(function(p) {{ p.vorp = Math.max(0, p.total_fp - replLevel); }});

    // 4. Positional scarcity
    if (noRoles) {{
        PLAYERS.forEach(function(p) {{ p.scarcity = 1.0; }});
    }} else {{
        var roles = ['BAT', 'BOWL', 'AR', 'WK'];
        var scarcity = {{}};
        var totalDemand = 0;
        roles.forEach(function(r) {{
            var dem = demand[r] || 0;
            if (dem === 0) {{ scarcity[r] = 1.0; return; }}
            var supply = PLAYERS.filter(function(p) {{ return p.role === r && p.vorp > 0; }}).length;
            supply = Math.max(supply, 1);
            scarcity[r] = dem / supply;
            totalDemand++;
        }});
        // Normalize to avg = 1.0 (only across roles with demand > 0)
        var activeRoles = roles.filter(function(r) {{ return (demand[r] || 0) > 0; }});
        if (activeRoles.length > 0) {{
            var avgSc = activeRoles.reduce(function(s, r) {{ return s + scarcity[r]; }}, 0) / activeRoles.length;
            if (avgSc > 0) {{
                activeRoles.forEach(function(r) {{ scarcity[r] /= avgSc; }});
            }}
        }}
        // Roles with 0 demand get multiplier 1.0
        roles.forEach(function(r) {{ if (!scarcity[r] || (demand[r] || 0) === 0) scarcity[r] = 1.0; }});
        PLAYERS.forEach(function(p) {{ p.scarcity = scarcity[p.role] || 1.0; }});
    }}

    // 5. Adjusted VORP (scarcity × consistency)
    PLAYERS.forEach(function(p) {{ p.adj_vorp = p.vorp * p.scarcity * p.consistency; }});

    // 6. Credit conversion
    var reserveCredits = nDrafted * MIN_BID;
    var distributable = Math.max(0, totalCredits - reserveCredits);
    var totalAdjVorp = PLAYERS.reduce(function(s, p) {{ return s + (p.adj_vorp > 0 ? p.adj_vorp : 0); }}, 0);

    PLAYERS.forEach(function(p) {{
        if (p.adj_vorp > 0 && totalAdjVorp > 0) {{
            p.auction_value = Math.round(((p.adj_vorp / totalAdjVorp) * distributable + MIN_BID) * 2) / 2;
        }} else {{
            p.auction_value = MIN_BID;
        }}
        p.auction_value = Math.max(p.auction_value, MIN_BID);
    }});

    // 7. Tiers, bid ranges, inflation
    PLAYERS.forEach(function(p) {{
        p.tier = getTier(p.auction_value);
        p.floor = Math.max(MIN_BID, Math.round(p.auction_value * 0.7 * 10) / 10);
        p.max_bid = Math.max(MIN_BID, Math.round(p.auction_value * 1.3 * 10) / 10);
        p.inflation = getInflation(p.tier, p.ptype, p.source);
    }});

    // Update everything
    updateStats(nTeams, budget, roster, nDrafted, totalCredits);
    rebuildTable();
    rebuildCharts(demand, noRoles);
    rebuildRoleLists();
    $('subtitle').textContent = 'IPL 2026 — ' + nTeams + ' Teams × ' + budget + ' Credits — ' + roster + ' Players Each';

    btn.textContent = 'Recalculate Valuations';
    btn.disabled = false;
    $('cfg-warning').textContent = 'Updated!';
    $('cfg-warning').style.color = '#27ae60';
    setTimeout(function() {{ $('cfg-warning').textContent = ''; $('cfg-warning').style.color = '#e74c3c'; }}, 2000);
  }} catch (e) {{
    var btn = document.querySelector('.recalc-btn');
    btn.textContent = 'Recalculate Valuations';
    btn.disabled = false;
    $('cfg-warning').textContent = 'Error: ' + e.message;
    $('cfg-warning').style.color = '#e74c3c';
    console.error('Recalculate error:', e);
  }}
}}

// ── Stats update ─────────────────────────────────────────────────────
function updateStats(nTeams, budget, roster, nDrafted, totalCredits) {{
    $('stat-available').textContent = PLAYERS.length;
    $('stat-drafted').textContent = nDrafted;
    $('stat-credits').textContent = totalCredits;
    var franchise = PLAYERS.filter(function(p) {{ return p.tier === 'Franchise'; }}).length;
    $('stat-franchise').textContent = franchise;
    var topVal = Math.max.apply(null, PLAYERS.map(function(p) {{ return p.auction_value; }}));
    $('stat-topvalue').textContent = topVal.toFixed(1);
}}

// ── Table rebuild ────────────────────────────────────────────────────
function rebuildTable() {{
    var sorted = PLAYERS.slice().sort(function(a, b) {{ return b.auction_value - a.auction_value; }});
    var tbody = $('tbl-body');
    var rows = '';
    for (var i = 0; i < sorted.length; i++) {{
        var p = sorted[i];
        var rc = ROLE_COLORS[p.role] || '#666';
        var tc = TIER_COLORS[tierFullName[p.tier]] || '#666';
        rows += '<tr data-role="' + p.role + '" data-tier="' + p.tier + '">' +
            '<td>' + (i + 1) + '</td>' +
            '<td><strong>' + p.name + '</strong></td>' +
            '<td>' + p.team + '</td>' +
            '<td><span class="role-badge" style="background:' + rc + '">' + p.role + '</span></td>' +
            '<td><strong>' + p.auction_value.toFixed(1) + '</strong></td>' +
            '<td>' + p.floor.toFixed(1) + '</td>' +
            '<td>' + p.max_bid.toFixed(1) + '</td>' +
            '<td><span class="tier-badge" style="background:' + tc + ';color:white">' + p.tier + '</span></td>' +
            '<td>' + p.proj_avg_fp.toFixed(1) + '</td>' +
            '<td>' + (p.fp24 !== null ? p.fp24.toFixed(1) : '—') + '</td>' +
            '<td>' + (p.fp25 !== null ? p.fp25.toFixed(1) : '—') + '</td>' +
            '<td>' + p.total_fp.toFixed(0) + '</td>' +
            '<td>' + p.inflation + '</td>' +
            '<td>' + p.source + '</td>' +
            '</tr>';
    }}
    tbody.innerHTML = rows;
    filterMaster();
}}

// ── Chart rebuilds ───────────────────────────────────────────────────
function rebuildCharts(demand, noRoles) {{
    var roles = ['BAT', 'BOWL', 'AR', 'WK'];
    var sorted = PLAYERS.slice().sort(function(a, b) {{ return b.auction_value - a.auction_value; }});
    var top40 = sorted.slice(0, 40);

    // Chart 1: Top 40 bar
    var traces1 = roles.map(function(r) {{
        var rd = top40.filter(function(p) {{ return p.role === r; }});
        return {{
            x: rd.map(function(p) {{ return p.name; }}),
            y: rd.map(function(p) {{ return p.auction_value; }}),
            text: rd.map(function(p) {{ return p.auction_value.toFixed(1); }}),
            textposition: 'outside', type: 'bar', name: r, marker: {{color: ROLE_COLORS[r]}},
            hovertemplate: '%{{x}}<br>Value: %{{y:.1f}}<br>Role: ' + r + '<extra></extra>'
        }};
    }});
    Plotly.react('chart1', traces1, {{
        title: 'Top 40 Players by Auction Value', barmode: 'group', height: 500,
        xaxis: {{tickangle: -45}}, yaxis: {{title: 'Auction Value (Credits)'}},
        legend: {{orientation: 'h', yanchor: 'bottom', y: 1.02}}
    }}, {{responsive: true}});

    // Chart 2: Tier × Role stacked bar
    var tierOrder = ['Franchise', 'Premium', 'Solid', 'Bargain', 'Min Bid'];
    var traces2 = tierOrder.map(function(t) {{
        var counts = roles.map(function(r) {{
            return PLAYERS.filter(function(p) {{ return p.role === r && p.tier === t; }}).length;
        }});
        return {{
            x: roles, y: counts, text: counts, textposition: 'inside',
            type: 'bar', name: t, marker: {{color: TIER_COLORS[tierFullName[t]]}}
        }};
    }});
    Plotly.react('chart2', traces2, {{
        title: 'Tier Distribution by Role', barmode: 'stack', height: 450,
        xaxis: {{title: 'Role'}}, yaxis: {{title: 'Number of Players'}},
        legend: {{orientation: 'h', yanchor: 'bottom', y: 1.02}}
    }}, {{responsive: true}});

    // Chart 3: Supply vs Demand
    var demandVals = roles.map(function(r) {{ return demand[r] || 0; }});
    var supplyVals = roles.map(function(r) {{
        return PLAYERS.filter(function(p) {{ return p.role === r && p.vorp > 0; }}).length;
    }});
    var traces3 = [
        {{x: roles, y: demandVals, text: demandVals, textposition: 'outside', type: 'bar', name: 'Demand', marker: {{color: '#e74c3c'}}}},
        {{x: roles, y: supplyVals, text: supplyVals, textposition: 'outside', type: 'bar', name: 'Supply (above repl.)', marker: {{color: '#3498db'}}}}
    ];
    var scTitle = noRoles ? 'Positional Scarcity: No Role Requirements (all multipliers = 1.0)' : 'Positional Scarcity: Supply vs Demand';
    Plotly.react('chart3', traces3, {{
        title: scTitle, barmode: 'group', height: 400,
        xaxis: {{title: 'Role'}}, yaxis: {{title: 'Players'}}
    }}, {{responsive: true}});

    // Chart 4: Value vs FP scatter
    var traces4 = roles.map(function(r) {{
        var rd = PLAYERS.filter(function(p) {{ return p.role === r; }});
        return {{
            x: rd.map(function(p) {{ return p.proj_avg_fp; }}),
            y: rd.map(function(p) {{ return p.auction_value; }}),
            text: rd.map(function(p) {{ return p.name; }}),
            mode: 'markers', type: 'scatter', name: r,
            marker: {{color: ROLE_COLORS[r], size: 8, opacity: 0.7}},
            hovertemplate: '%{{text}}<br>FP/match: %{{x:.1f}}<br>Value: %{{y:.1f}}<extra></extra>'
        }};
    }});
    Plotly.react('chart4', traces4, {{
        title: 'Auction Value vs Projected Fantasy Points', height: 500,
        xaxis: {{title: 'Projected Avg FP/Match'}}, yaxis: {{title: 'Auction Value (Credits)'}}
    }}, {{responsive: true}});
}}

// ── Role lists rebuild ───────────────────────────────────────────────
function rebuildRoleLists() {{
    var roles = ['BAT', 'BOWL', 'AR', 'WK'];
    var roleNames = {{BAT: 'Batters', BOWL: 'Bowlers', AR: 'All-Rounders', WK: 'Wicketkeepers'}};
    var html = '';
    roles.forEach(function(role) {{
        var rd = PLAYERS.filter(function(p) {{ return p.role === role; }}).sort(function(a, b) {{ return b.auction_value - a.auction_value; }});
        var rc = ROLE_COLORS[role];
        html += '<div class="role-section">' +
            '<h3 onclick="toggleSection(this)" ' +
            'style="border-left:4px solid ' + rc + ';padding-left:12px;">' +
            roleNames[role] + ' (' + rd.length + ' players) — click to expand</h3>' +
            '<div style="display:none;"><table><thead><tr>' +
            '<th>#</th><th>Player</th><th>IPL Team</th><th>Value</th><th>Floor</th><th>Max</th><th>Tier</th><th>Proj FP/M</th><th>2024 FP/M</th><th>2025 FP/M</th><th>Inflation</th>' +
            '</tr></thead><tbody>';
        for (var i = 0; i < rd.length; i++) {{
            var p = rd[i];
            var tc = TIER_COLORS[tierFullName[p.tier]] || '#666';
            html += '<tr><td>' + (i + 1) + '</td><td><strong>' + p.name + '</strong></td><td>' + p.team + '</td>' +
                '<td><strong>' + p.auction_value.toFixed(1) + '</strong></td><td>' + p.floor.toFixed(1) + '</td>' +
                '<td>' + p.max_bid.toFixed(1) + '</td>' +
                '<td><span class="tier-badge" style="background:' + tc + ';color:white">' + p.tier + '</span></td>' +
                '<td>' + p.proj_avg_fp.toFixed(1) + '</td>' +
                '<td>' + (p.fp24 !== null ? p.fp24.toFixed(1) : '—') + '</td>' +
                '<td>' + (p.fp25 !== null ? p.fp25.toFixed(1) : '—') + '</td>' +
                '<td>' + p.inflation + '</td></tr>';
        }}
        html += '</tbody></table></div></div>';
    }});
    $('role-lists').innerHTML = html;
}}

// ── Table filtering & sorting ────────────────────────────────────────
function filterMaster() {{
    var role = $('f-role').value;
    var tier = $('f-tier').value;
    var search = $('f-search').value.toLowerCase();
    var rows = document.querySelectorAll('#master-tbl tbody tr');
    var count = 0;
    rows.forEach(function(row) {{
        var show = true;
        if (role && row.getAttribute('data-role') !== role) show = false;
        if (tier && row.getAttribute('data-tier') !== tier) show = false;
        if (search && !row.cells[1].textContent.toLowerCase().includes(search)) show = false;
        row.style.display = show ? '' : 'none';
        if (show) count++;
    }});
    $('row-count').textContent = count + ' players shown';
}}

var sortDir = {{}};
function doSort(col, type) {{
    var table = $('master-tbl');
    var rows = Array.from(table.tBodies[0].rows);
    var dir = sortDir[col] = !sortDir[col];
    rows.sort(function(a, b) {{
        var av = a.cells[col].textContent.trim();
        var bv = b.cells[col].textContent.trim();
        if (type === 'num') {{ av = parseFloat(av) || 0; bv = parseFloat(bv) || 0; }}
        if (av < bv) return dir ? -1 : 1;
        if (av > bv) return dir ? 1 : -1;
        return 0;
    }});
    rows.forEach(function(row) {{ table.tBodies[0].appendChild(row); }});
}}

// ── Initial calculation ──────────────────────────────────────────────
recalculate();
</script>

</body>
</html>'''
    return html


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    project_root = Path(__file__).parent.parent
    output_dir = project_root / 'results' / 'FantasyProjections' / 'fantasy_auction'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FANTASY AUCTION VALUATION SYSTEM")
    print(f"{N_TEAMS} teams × {BUDGET_PER_TEAM} credits — {ROSTER_SIZE} players each")
    print("=" * 70)

    # Step 1: Load data
    df = load_and_merge_data()

    # Step 2: Replacement level & VORP
    replacement = compute_replacement_level(df)
    print(f"\nReplacement level (80th player): {replacement:.1f} total FP "
          f"({replacement / MATCHES_PER_SEASON:.1f} FP/match)")
    df = compute_vorp(df, replacement)

    # Step 3: Positional scarcity
    df = compute_positional_scarcity(df)

    # Step 4: Consistency adjustment
    df = compute_consistency_adjustment(df)

    # Step 5: Credit conversion
    df = compute_auction_values(df)

    # Step 6: Tiers
    df = classify_tiers(df)

    # Step 7: Game theory
    df = compute_bid_ranges(df)
    df = compute_inflation_risk(df)

    # ── Print summary ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("TOP 20 PLAYERS BY AUCTION VALUE")
    print("=" * 70)
    top20 = df.nlargest(20, 'auction_value')
    for i, (_, r) in enumerate(top20.iterrows(), 1):
        print(f"  {i:2d}. {r['Player']:25s} ({r['role']:4s})  "
              f"Value: {r['auction_value']:5.1f}  "
              f"[{r['floor_price']:.1f} — {r['max_bid']:.1f}]  "
              f"FP/m: {r['projected_avg_fp_2026']:.1f}  "
              f"Tier: {r['tier'].split(' — ')[1]}")

    print(f"\n{'='*70}")
    print("TOP 5 PER ROLE")
    print("=" * 70)
    for role in ['BAT', 'BOWL', 'AR', 'WK']:
        role_top = df[df['role'] == role].nlargest(5, 'auction_value')
        print(f"\n  {role}:")
        for _, r in role_top.iterrows():
            print(f"    {r['Player']:25s}  Value: {r['auction_value']:5.1f}  FP/m: {r['projected_avg_fp_2026']:.1f}")

    # ── Save outputs ──────────────────────────────────────────────────

    # 8a. Master CSV
    master_cols = ['Player', 'Team', 'role', 'projected_avg_fp_2026', 'projected_total_fp',
                   'replacement_level', 'VORP', 'scarcity_multiplier', 'consistency_score',
                   'adjusted_VORP', 'auction_value', 'tier', 'floor_price', 'fair_price',
                   'max_bid', 'inflation_risk', 'prediction_source', 'Player_Type']
    master = df[[c for c in master_cols if c in df.columns]].sort_values(
        'auction_value', ascending=False)
    save_dataframe(master, output_dir / 'fantasy_auction_values.csv', format='csv')
    print(f"\n✓ Master CSV saved: {output_dir / 'fantasy_auction_values.csv'}")

    # 8b. Cheat sheet
    cheat = generate_cheat_sheet(df)
    save_dataframe(cheat, output_dir / 'auction_cheat_sheet.csv', format='csv')
    print(f"✓ Cheat sheet saved: {output_dir / 'auction_cheat_sheet.csv'}")

    # 8c. Nomination strategy
    strategy = generate_nomination_strategy(df)
    save_dataframe(strategy, output_dir / 'nomination_strategy.csv', format='csv')
    print(f"✓ Nomination strategy saved: {output_dir / 'nomination_strategy.csv'}")

    # 8d. Dashboard HTML
    dashboard = build_dashboard_html(df)
    with open(output_dir / 'fantasy_auction_dashboard.html', 'w') as f:
        f.write(dashboard)
    print(f"✓ Dashboard saved: {output_dir / 'fantasy_auction_dashboard.html'}")

    # ── Sanity checks ─────────────────────────────────────────────────
    top80_sum = df.nlargest(80, 'auction_value')['auction_value'].sum()
    print(f"\n{'='*70}")
    print("SANITY CHECKS")
    print("=" * 70)
    print(f"  Sum of top 80 auction values: {top80_sum:.1f} (target: ~{N_TEAMS * BUDGET_PER_TEAM})")
    print(f"  Tier 1 count: {(df['tier'] == 'Tier 1 — Franchise').sum()}")
    print(f"  Min bid players: {(df['auction_value'] <= 1).sum()}")
    print(f"  Roles in tiers: {df.groupby('tier')['role'].nunique().to_dict()}")


if __name__ == '__main__':
    main()
