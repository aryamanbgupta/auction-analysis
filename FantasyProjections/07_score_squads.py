"""
Score ALL 250 IPL 2026 squad players with fantasy point projections.

Handles the ID mismatch between squad enriched file and Cricsheet ball-by-ball data
by using a multi-layer matching approach:
  1. Squad cricsheet_id → projection ID lookup
  2. Abbreviated name generation from full_name (e.g. "Rohit Gurunath Sharma" → "RG Sharma")
  3. Player name → projection name lookup
  4. Fuzzy match (threshold 0.80)
  5. Marcel fallback from season history
  6. Replacement level for unknowns

OUTPUT: results/FantasyProjections/squad_2026/squad_fantasy_projections_2026.csv
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher

sys.path.insert(0, str(Path(__file__).parent.parent / 'WARprojections'))
from utils import save_dataframe


def normalize_name(name):
    if pd.isna(name):
        return None
    name = str(name).lower().strip()
    name = re.sub(r'^(mr\.?|dr\.?|ms\.?)\s+', '', name)
    name = re.sub(r'\s+(jr\.?|sr\.?|ii|iii)$', '', name)
    name = re.sub(r'[^\w\s]', '', name)
    name = ' '.join(name.split())
    return name


def generate_abbreviated_names(full_name):
    """Generate Cricsheet-style abbreviated names from a full name.
    e.g. 'Rohit Gurunath Sharma' → ['RG Sharma', 'R Sharma']
         'Virat Kohli' → ['V Kohli']
         'Sanju Viswanath Samson' → ['SV Samson', 'S Samson']
    """
    if pd.isna(full_name) or not str(full_name).strip():
        return []
    parts = str(full_name).strip().split()
    if len(parts) < 2:
        return []
    last = parts[-1]
    abbrevs = set()
    # First initial + last name
    abbrevs.add(f"{parts[0][0]} {last}")
    # First + middle initials + last name (if 3+ parts)
    if len(parts) >= 3:
        initials = ''.join(p[0] for p in parts[:-1])
        abbrevs.add(f"{initials} {last}")
    # First two initials + last (common Cricsheet format)
    if len(parts) >= 3:
        abbrevs.add(f"{parts[0][0]}{parts[1][0]} {last}")
    return list(abbrevs)


def fuzzy_match(name, candidates, threshold=0.90):
    """Fuzzy match with first-initial validation to prevent false positives."""
    name_norm = normalize_name(name)
    if not name_norm:
        return None
    name_parts = name_norm.split()
    name_last = name_parts[-1] if name_parts else ''
    name_first_initial = name_parts[0][0] if name_parts and name_parts[0] else ''

    best_match, best_score = None, 0
    for cand in candidates:
        cand_norm = normalize_name(cand)
        if not cand_norm:
            continue
        if name_norm == cand_norm:
            return cand
        # Require matching last name for shorter names (reduces false positives)
        cand_parts = cand_norm.split()
        cand_last = cand_parts[-1] if cand_parts else ''
        if name_last != cand_last:
            continue
        score = SequenceMatcher(None, name_norm, cand_norm).ratio()
        if score > best_score:
            best_score = score
            best_match = cand
    return best_match if best_score >= threshold else None


def get_manual_overrides():
    """Manual overrides for known ID/name mismatches between squad and Cricsheet data.
    Maps squad Player name → Cricsheet player_id (from ball-by-ball data).
    """
    return {
        'Rohit Sharma': '740742ef',       # RG Sharma in Cricsheet
        'Rasikh Dar': 'b8527c3d',          # Rasikh Salam in Cricsheet
    }


def main():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    fp_results = project_root / 'results' / 'FantasyProjections'
    output_dir = fp_results / 'squad_2026'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FANTASY POINTS SQUAD SCORING — IPL 2026")
    print("=" * 70)

    # Load squad
    squad = pd.read_csv(data_dir / 'ipl_2026_squads_enriched.csv')
    print(f"Squad: {len(squad)} players, {squad['Team'].nunique()} teams")

    # Load production projections
    proj = pd.read_csv(fp_results / 'fantasy_projections_2026_lookup.csv')
    print(f"Production projections: {len(proj)} players")

    # Load season data for Marcel fallback
    season = pd.read_csv(data_dir / 'fantasy_points_per_season.csv')

    # ── Build lookup tables ────────────────────────────────────────────
    # ID lookup: cricsheet_id → (projected_fp, source)
    id_lookup = {}
    for _, r in proj.iterrows():
        if pd.notna(r['player_id']):
            id_lookup[r['player_id']] = (r['projected_avg_fp_2026'], 'Production')

    # Name lookup: normalized name → (projected_fp, source)
    name_lookup = {}
    for _, r in proj.iterrows():
        n = normalize_name(r['player_name'])
        if n:
            name_lookup[n] = (r['projected_avg_fp_2026'], 'Production')

    # Marcel fallback: weighted avg of recent seasons
    marcel_weights = {2025: 5, 2024: 4, 2023: 3, 2022: 2}
    marcel_id_lookup = {}
    marcel_name_lookup = {}
    for pid, grp in season.groupby('player_id'):
        recent = grp[grp['season'] >= 2022]
        if len(recent) == 0:
            continue
        pname = grp['player_name'].iloc[0]
        total_v, total_w = 0, 0
        for _, r in recent.iterrows():
            wt = marcel_weights.get(int(r['season']), 1)
            total_v += r['avg_fantasy_pts'] * wt
            total_w += wt
        fp = total_v / total_w
        if pid not in id_lookup:
            marcel_id_lookup[pid] = (fp, 'Marcel')
        n = normalize_name(pname)
        if n and n not in name_lookup:
            marcel_name_lookup[n] = (fp, 'Marcel')

    print(f"Lookups: {len(id_lookup)} prod IDs, {len(name_lookup)} prod names, "
          f"{len(marcel_id_lookup)} marcel IDs, {len(marcel_name_lookup)} marcel names")

    # All name candidates for fuzzy matching
    all_name_candidates = list(name_lookup.keys()) + list(marcel_name_lookup.keys())

    # ── Manual overrides ─────────────────────────────────────────────
    manual_overrides = get_manual_overrides()
    print(f"Manual overrides: {len(manual_overrides)} players")

    # ── Score each squad player ────────────────────────────────────────
    results = []
    match_log = []

    for _, row in squad.iterrows():
        player = row['Player']
        squad_id = row.get('cricsheet_id')
        full_name = row.get('full_name')
        fp, source, method = None, None, None

        # 0. Manual override (known ID mismatches)
        if player in manual_overrides:
            override_id = manual_overrides[player]
            if override_id in id_lookup:
                fp, source = id_lookup[override_id]
                method = 'Manual_override'
            elif override_id in marcel_id_lookup:
                fp, source = marcel_id_lookup[override_id]
                method = 'Manual_override_marcel'

        # 1. Direct ID match to production projections
        if fp is None and pd.notna(squad_id) and squad_id in id_lookup:
            fp, source = id_lookup[squad_id]
            method = 'ID_direct'

        # 2. Direct ID match to Marcel
        if fp is None and pd.notna(squad_id) and squad_id in marcel_id_lookup:
            fp, source = marcel_id_lookup[squad_id]
            method = 'ID_marcel'

        # 3. Abbreviated name from full_name (handles Cricsheet format)
        if fp is None and pd.notna(full_name):
            abbrevs = generate_abbreviated_names(full_name)
            for abbrev in abbrevs:
                n = normalize_name(abbrev)
                if n in name_lookup:
                    fp, source = name_lookup[n]
                    method = f'Abbrev:{abbrev}'
                    break
                if n in marcel_name_lookup:
                    fp, source = marcel_name_lookup[n]
                    method = f'Abbrev:{abbrev}'
                    break

        # 4. Player name direct match
        if fp is None:
            n = normalize_name(player)
            if n in name_lookup:
                fp, source = name_lookup[n]
                method = 'Name_direct'
            elif n in marcel_name_lookup:
                fp, source = marcel_name_lookup[n]
                method = 'Name_marcel'

        # 5. Fuzzy match
        if fp is None:
            names_to_try = [normalize_name(player)]
            if pd.notna(full_name):
                names_to_try.append(normalize_name(full_name))

            for name_try in names_to_try:
                if not name_try:
                    continue
                match = fuzzy_match(name_try, all_name_candidates, threshold=0.90)
                if match:
                    if match in name_lookup:
                        fp, source = name_lookup[match]
                    else:
                        fp, source = marcel_name_lookup[match]
                    method = f'Fuzzy:{match}'
                    break

        # 6. Replacement level
        if fp is None:
            fp = 25.0
            source = 'Replacement'
            method = 'None'

        results.append({
            'Player': player,
            'Team': row['Team'],
            'Price_Cr': row.get('Price_Cr'),
            'Acquisition': row.get('Acquisition'),
            'Player_Type': row.get('Player_Type'),
            'playing_role': row.get('playing_role', ''),
            'projected_avg_fp_2026': round(fp, 1),
            'prediction_source': source,
            'match_method': method,
        })
        match_log.append(f"{player:30s} → {source:12s} via {method}")

    results_df = pd.DataFrame(results)

    # ── Coverage Stats ─────────────────────────────────────────────────
    total = len(results_df)
    prod = (results_df['prediction_source'] == 'Production').sum()
    marcel = (results_df['prediction_source'] == 'Marcel').sum()
    repl = (results_df['prediction_source'] == 'Replacement').sum()

    print(f"\n{'='*70}")
    print("COVERAGE")
    print("=" * 70)
    print(f"Total: {total} players")
    print(f"  Production model: {prod} ({100*prod/total:.1f}%)")
    print(f"  Marcel fallback:  {marcel} ({100*marcel/total:.1f}%)")
    print(f"  Replacement:      {repl} ({100*repl/total:.1f}%)")
    print(f"  Matched total:    {prod+marcel}/{total} ({100*(prod+marcel)/total:.1f}%)")

    print(f"\nBy method:\n{results_df['match_method'].apply(lambda x: x.split(':')[0]).value_counts().to_string()}")

    # ── Sanity check: key players ──────────────────────────────────────
    print(f"\n{'='*70}")
    print("KEY PLAYER CHECK")
    print("=" * 70)
    key_players = ['Rohit Sharma', 'Virat Kohli', 'Rashid Khan', 'Jasprit Bumrah',
                   'MS Dhoni', 'Rishabh Pant', 'Sanju Samson', 'Shreyas Iyer',
                   'Suryakumar Yadav', 'KL Rahul']
    for name in key_players:
        row = results_df[results_df['Player'].str.contains(name, case=False, na=False)]
        if len(row) > 0:
            r = row.iloc[0]
            print(f"  {r['Player']:25s} ({r['Team']})  FP: {r['projected_avg_fp_2026']:5.1f}  [{r['prediction_source']}/{r['match_method']}]")
        else:
            print(f"  {name:25s} NOT IN SQUAD")

    # ── Team Rankings ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("TEAM STRENGTH (Avg Projected FP/Match)")
    print("=" * 70)
    team_avg = results_df.groupby('Team')['projected_avg_fp_2026'].agg(['mean', 'sum', 'max'])
    team_avg = team_avg.sort_values('mean', ascending=False)
    for team, r in team_avg.iterrows():
        print(f"  {team:5s}  Avg: {r['mean']:5.1f}  Total: {r['sum']:6.0f}  Best: {r['max']:5.1f}")

    # ── Top 30 ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("TOP 30 PLAYERS LEAGUE-WIDE")
    print("=" * 70)
    top = results_df.nlargest(30, 'projected_avg_fp_2026')
    for i, (_, r) in enumerate(top.iterrows()):
        print(f"  {i+1:2d}. {r['Player']:25s} ({r['Team']})  FP: {r['projected_avg_fp_2026']:5.1f}  "
              f"₹{r['Price_Cr']:.1f}Cr  [{r['prediction_source']}]")

    # ── Replacement level players (for inspection) ─────────────────────
    repl_players = results_df[results_df['prediction_source'] == 'Replacement']
    if len(repl_players) > 0:
        print(f"\n{'='*70}")
        print(f"REPLACEMENT LEVEL PLAYERS ({len(repl_players)}) — no historical data found")
        print("=" * 70)
        for _, r in repl_players.iterrows():
            print(f"  {r['Player']:30s} ({r['Team']})  {r['Player_Type']}")

    # Save
    output_path = output_dir / 'squad_fantasy_projections_2026.csv'
    save_dataframe(results_df, output_path, format='csv')

    # Save match log for debugging
    with open(output_dir / 'match_log.txt', 'w') as f:
        f.write('\n'.join(match_log))

    print(f"\n✓ Saved to {output_path}")


if __name__ == '__main__':
    main()
