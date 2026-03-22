"""
Score IPL 2026 Auction Pool with Fantasy Point Projections.

Matches auction players to fantasy projections using:
1. Cricsheet ID matching
2. Name matching (exact + variants)
3. Fuzzy matching (SequenceMatcher >= 0.65)

Fallback: Marcel-weighted historical avg for players not in production model.

OUTPUT: results/FantasyProjections/auction_fantasy_projections_2026.csv
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher

sys.path.insert(0, str(Path(__file__).parent.parent / 'WARprojections'))
from utils import save_dataframe


# ── Matching helpers (reused from WAR auction scorer) ──────────────────────

def normalize_name(name):
    if pd.isna(name):
        return None
    name = str(name).lower().strip()
    name = re.sub(r'^(mr\.?|dr\.?|ms\.?)\s+', '', name)
    name = re.sub(r'\s+(jr\.?|sr\.?|ii|iii)$', '', name)
    name = re.sub(r'[^\w\s]', '', name)
    name = ' '.join(name.split())
    return name


def generate_name_variants(row):
    variants = set()
    for col in ['unique_name', 'name', 'PLAYER', 'full_name']:
        if pd.notna(row.get(col)):
            name = str(row[col])
            variants.add(normalize_name(name))
            parts = name.split()
            if len(parts) >= 2:
                variants.add(normalize_name(f"{parts[0]} {parts[-1]}"))
                variants.add(normalize_name(f"{parts[0][0]} {parts[-1]}"))
    return [v for v in variants if v]


def fuzzy_match(name, candidates, threshold=0.78):
    """Fuzzy match with higher threshold to avoid false positives."""
    best_match, best_score = None, 0
    name_norm = normalize_name(name)
    if not name_norm:
        return None
    for candidate in candidates:
        cand_norm = normalize_name(candidate) if candidate else None
        if not cand_norm:
            continue
        if name_norm == cand_norm:
            return candidate
        score = SequenceMatcher(None, name_norm, cand_norm).ratio()
        if score > best_score:
            best_score = score
            best_match = candidate
    return best_match if best_score >= threshold else None


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    fp_results = project_root / 'results' / 'FantasyProjections'
    output_dir = fp_results / 'auction_2026'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FANTASY POINTS AUCTION SCORING")
    print("=" * 70)

    # Load auction list
    auction = pd.read_csv(data_dir / 'ipl_2026_auction_enriched.csv')
    print(f"Auction pool: {len(auction)} players")

    # Load production projections
    projections = pd.read_csv(fp_results / 'fantasy_projections_2026_lookup.csv')
    print(f"Fantasy projections: {len(projections)} players")

    # Load season-level fantasy data for Marcel fallback
    season_fp = pd.read_csv(data_dir / 'fantasy_points_per_season.csv')

    # ── Build Marcel fallback ──────────────────────────────────────────
    # For players not in production model, use Marcel-weighted avg of last 3 seasons
    recent = season_fp[season_fp['season'] >= 2022].copy()
    marcel_weights = {2025: 5, 2024: 4, 2023: 3, 2022: 2}

    marcel_records = []
    for pid, grp in recent.groupby('player_id'):
        pname = grp['player_name'].iloc[0]
        total_w, total_v = 0, 0
        for _, r in grp.iterrows():
            w = marcel_weights.get(int(r['season']), 1)
            total_v += r['avg_fantasy_pts'] * w
            total_w += w
        if total_w > 0:
            marcel_records.append({
                'player_id': pid, 'player_name': pname,
                'marcel_avg_fp': total_v / total_w,
            })
    marcel_df = pd.DataFrame(marcel_records)
    print(f"Marcel fallback: {len(marcel_df)} players")

    # ── Build lookups ──────────────────────────────────────────────────
    # Priority: Production > Marcel

    id_lookup = {}
    name_lookup = {}

    # Production projections (highest priority)
    for _, row in projections.iterrows():
        pid = row.get('player_id')
        if pd.notna(pid):
            id_lookup[pid] = (row['projected_avg_fp_2026'], 'Production')
        name = normalize_name(row.get('player_name'))
        if name:
            name_lookup[name] = (row['projected_avg_fp_2026'], 'Production')

    # Marcel (fallback)
    for _, row in marcel_df.iterrows():
        pid = row.get('player_id')
        if pd.notna(pid) and pid not in id_lookup:
            id_lookup[pid] = (row['marcel_avg_fp'], 'Marcel')
        name = normalize_name(row.get('player_name'))
        if name and name not in name_lookup:
            name_lookup[name] = (row['marcel_avg_fp'], 'Marcel')

    print(f"Lookup sizes: {len(id_lookup)} by ID, {len(name_lookup)} by name")

    # ── Score each auction player ──────────────────────────────────────
    results = []
    for _, row in auction.iterrows():
        cricsheet_id = row.get('cricsheet_id')
        fp, source, method = None, None, None

        # Step 1: ID matching
        if pd.notna(cricsheet_id) and cricsheet_id in id_lookup:
            fp, source = id_lookup[cricsheet_id]
            method = 'ID'

        # Step 2: Name matching
        if fp is None:
            variants = generate_name_variants(row)
            for name in variants:
                if name in name_lookup:
                    fp, source = name_lookup[name]
                    method = 'Name'
                    break

        # Step 3: Fuzzy matching
        if fp is None and variants:
            all_names = list(name_lookup.keys())
            for variant in variants:
                match = fuzzy_match(variant, all_names, threshold=0.65)
                if match:
                    fp, source = name_lookup[match]
                    method = 'Fuzzy'
                    break

        # Default: league average replacement level
        if fp is None:
            fp = 25.0  # approximate league-average fantasy pts/match
            source = 'Replacement_Level'
            method = 'None'

        results.append({
            'sr_no': row.get('SR. NO.'),
            'player': row.get('PLAYER') or row.get('name'),
            'country': row.get('COUNTRY') or row.get('country'),
            'role': row.get('playing_role'),
            'base_price': row.get('BASE PRICE (INR LAKH)'),
            'capped': row.get('C/U/A'),
            'projected_avg_fp_2026': round(fp, 1),
            'prediction_source': source,
            'match_method': method,
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('projected_avg_fp_2026', ascending=False)

    # ── Stats ──────────────────────────────────────────────────────────
    total = len(results_df)
    matched = (results_df['prediction_source'] != 'Replacement_Level').sum()

    print(f"\n{'='*70}")
    print("COVERAGE")
    print("=" * 70)
    print(f"Matched: {matched}/{total} ({100*matched/total:.1f}%)")
    print(f"\nBy source:\n{results_df['prediction_source'].value_counts().to_string()}")
    print(f"\nBy method:\n{results_df['match_method'].value_counts().to_string()}")

    # ── Top players ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("TOP 30 AUCTION PLAYERS BY PROJECTED FANTASY PTS/MATCH")
    print("=" * 70)
    top = results_df.head(30)
    for i, (_, r) in enumerate(top.iterrows()):
        print(f"  {i+1:2d}. {r['player']:25s} ({str(r['role'])[:20]:20s})  "
              f"FP: {r['projected_avg_fp_2026']:5.1f}  "
              f"Base: {r['base_price']}L  "
              f"[{r['prediction_source']}/{r['match_method']}]")

    # ── Role breakdown ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("AVERAGE PROJECTED FP BY ROLE")
    print("=" * 70)
    role_avg = results_df.groupby('role')['projected_avg_fp_2026'].agg(['mean', 'count', 'max'])
    print(role_avg.sort_values('mean', ascending=False).to_string())

    # Save
    output_path = output_dir / 'auction_fantasy_projections_2026.csv'
    save_dataframe(results_df, output_path, format='csv')
    print(f"\n✓ Saved to {output_path}")


if __name__ == '__main__':
    main()
