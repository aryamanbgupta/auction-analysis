"""
Reconcile Boston IPL 2026 fantasy points (ground truth from app) against
the 4 scoring variants computed in 10_ipl2026_custom_fantasy.py to
determine which rule set the app is actually using.

Match strategy: fuzzy match on last name + first initial, then full name.
Output: per-player comparison and overall fit (RMSE / mean abs error)
per variant. Flags players whose app total is between variant values.
"""

import sys, re
from pathlib import Path
from difflib import SequenceMatcher
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
BOSTON_FILE = PROJECT_ROOT / 'data' / 'Boston IPL 26_Results (1).xlsx'
VARIANTS_FILE = PROJECT_ROOT / 'results' / 'FantasyProjections' / 'ipl2026_custom' / 'season_to_date_totals.csv'
OUT_DIR = PROJECT_ROOT / 'results' / 'FantasyProjections' / 'ipl2026_custom'


def load_boston() -> pd.DataFrame:
    x = pd.ExcelFile(BOSTON_FILE)
    sheets = [s for s in x.sheet_names if s not in ('Unsold Players', 'Awards')]
    frames = []
    for s in sheets:
        df = pd.read_excel(x, sheet_name=s)
        df = df[df['Player'].notna() & df['Points'].notna()].copy()
        df['Manager'] = s
        frames.append(df[['Manager', 'Player', 'Role', 'Team', 'Points']])
    boston = pd.concat(frames, ignore_index=True)
    boston['Points'] = pd.to_numeric(boston['Points'], errors='coerce')
    boston = boston[boston['Points'].notna()].copy()
    boston['Player'] = boston['Player'].astype(str).str.strip()
    return boston


def load_variants() -> pd.DataFrame:
    return pd.read_csv(VARIANTS_FILE)


def normalize_name(n: str) -> str:
    n = re.sub(r'[^A-Za-z\s]', ' ', n).strip().lower()
    n = re.sub(r'\s+', ' ', n)
    return n


def last_name(n: str) -> str:
    parts = normalize_name(n).split()
    return parts[-1] if parts else ''


def match_players(boston: pd.DataFrame, variants: pd.DataFrame) -> pd.DataFrame:
    """Match Boston app names to cricsheet names in variants table."""
    variants = variants.copy()
    variants['_norm'] = variants['player_name'].apply(normalize_name)
    variants['_last'] = variants['player_name'].apply(last_name)

    results = []
    manual = {
        'Vaibhav Sooryavanshi': 'V Suryavanshi',
        'Philip Salt': 'PD Salt',
        'Jos Buttler': 'JC Buttler',
        'Yashasvi Jaiswal': 'YBK Jaiswal',
        'Prabhsimran Singh': 'P Simran Singh',
        'Sanju Samson': 'SV Samson',
        'Suryakumar Yadav': 'SA Yadav',
        'Angkrish Raghuvanshi': 'A Raghuvanshi',
        'Ayush Mhatre': 'A Mhatre',
        'Shivam Dube': 'SM Dube',
        'Shubman Gill': 'Shubman Gill',
        'Virat Kohli': 'V Kohli',
        'Ishan Kishan': 'Ishan Kishan',
        'Heinrich Klaasen': 'H Klaasen',
        'Rajat Patidar': 'RM Patidar',
        'Shreyas Iyer': 'SS Iyer',
        'KL Rahul': 'KL Rahul',
        'Travis Head': 'TM Head',
        'Jofra Archer': 'JC Archer',
        'Arshdeep Singh': 'Arshdeep Singh',
        'Noor Ahmad': 'Noor Ahmad',
        'Mohammed Siraj': 'Mohammed Siraj',
        'Devdutt Padikkal': 'D Padikkal',
        'Quinton de Kock': 'Q de Kock',
        'Hardik Pandya': 'HH Pandya',
        'Jasprit Bumrah': 'JJ Bumrah',
        'Rishabh Pant': 'RR Pant',
        'Abhishek Sharma': 'Abhishek Sharma',
        'Ruturaj Gaikwad': 'RD Gaikwad',
        'Sai Sudharsan': 'B Sai Sudharsan',
        'Rohit Sharma': 'RG Sharma',
        'Nicholas Pooran': 'N Pooran',
        'Priyansh Arya': 'Priyansh Arya',
        'Ajinkya Rahane': 'AM Rahane',
        'Pathum Nissanka': 'WPN Nissanka',
        'Josh Hazlewood': 'JR Hazlewood',
        'Jitesh Sharma': 'Jitesh Sharma',
        'Cameron Green': 'CD Green',
        'Mitchell Marsh': 'MR Marsh',
        'Kuldeep Yadav': 'Kuldeep Yadav',
        'Tilak Varma': 'Tilak Varma',
        'Dewald Brevis': 'DR Brevis',
        'Tim David': 'TH David',
        'Nehal Wadhera': 'N Wadhera',
        'Naman Dhir': 'N Dhir',
        'Shimron Hetmyer': 'SO Hetmyer',
        'Prasidh Krishna': 'M Prasidh Krishna',
        'Bhuvneshwar Kumar': 'B Kumar',
        'Marco Jansen': 'M Jansen',
        'Sunil Narine': 'SP Narine',
        'Rinku Singh': 'RK Singh',
        'Aiden Markram': 'AK Markram',
        'Varun Chakaravarthy': 'V Chakravarthy',
        'Manish Pandey': 'MK Pandey',
        'Rovman Powell': 'R Powell',
        'Harshit Rana': 'H Rana',
        'Nitish Rana': 'N Rana',
        'Finn Allen': 'FH Allen',
        'Trent Boult': 'TA Boult',
        'Lockie Ferguson': 'LH Ferguson',
        'Kyle Jamieson': 'KA Jamieson',
        'Pat Cummins': 'PJ Cummins',
        'Rashid Khan': 'Rashid Khan',
        'Ravindra Jadeja': 'RA Jadeja',
        'Axar Patel': 'AR Patel',
        'David Miller': 'DA Miller',
        'Yuzvendra Chahal': 'YS Chahal',
        'Prashant Veer': 'Prashant Veer',
        'Jacob Bethell': 'JG Bethell',
        'Matheesha Pathirana': 'M Pathirana',
    }

    for _, row in boston.iterrows():
        pname = row['Player']
        vname = None
        score = 0.0

        # Manual override
        manual_hit_missing = False
        if pname in manual:
            target = manual[pname]
            hit = variants[variants['player_name'] == target]
            if not hit.empty:
                vname = target
                score = 1.0
            else:
                # Manual target not found — player hasn't played this season.
                # Do NOT fuzzy-fall-through (that silently remaps to a different player, e.g. Harshit Rana -> N Rana).
                manual_hit_missing = True

        if vname is None and not manual_hit_missing:
            # Try fuzzy last-name + first-letter
            pn = normalize_name(pname)
            pl = last_name(pname)
            pfirst = pn.split()[0][0] if pn else ''
            cand = variants[variants['_last'] == pl]
            if len(cand) == 1:
                vname = cand.iloc[0]['player_name']
                score = 0.95
            elif len(cand) > 1:
                # Match first initial from cricsheet's initials-style name
                best = None; best_s = 0
                for _, c in cand.iterrows():
                    cname_norm = c['_norm']
                    s = SequenceMatcher(None, pn, cname_norm).ratio()
                    if s > best_s:
                        best_s = s; best = c['player_name']
                vname = best
                score = best_s

            if vname is None:
                # Full fuzzy match
                best = None; best_s = 0
                for _, c in variants.iterrows():
                    s = SequenceMatcher(None, pn, c['_norm']).ratio()
                    if s > best_s:
                        best_s = s; best = c['player_name']
                if best_s >= 0.7:
                    vname = best; score = best_s

        if vname:
            v = variants[variants['player_name'] == vname].iloc[0]
            results.append({
                'Manager': row['Manager'],
                'Player': pname,
                'Cricsheet_Name': vname,
                'Match_Quality': round(score, 2),
                'App_Points': row['Points'],
                'Matches': int(v['matches']),
                'V_BASE': v['V_BASE_total'],
                'V_HIGHEST': v['V_HIGHEST_total'],
                'V_NO_SR': v['V_NO_SR_total'],
                'V_NO_ECON': v['V_NO_ECON_total'],
            })
        else:
            results.append({
                'Manager': row['Manager'],
                'Player': pname,
                'Cricsheet_Name': None,
                'Match_Quality': 0,
                'App_Points': row['Points'],
                'Matches': 0,
                'V_BASE': np.nan, 'V_HIGHEST': np.nan, 'V_NO_SR': np.nan, 'V_NO_ECON': np.nan,
            })
    return pd.DataFrame(results)


def analyze(df: pd.DataFrame):
    m = df.dropna(subset=['V_BASE']).copy()
    for v in ['V_BASE', 'V_HIGHEST', 'V_NO_SR', 'V_NO_ECON']:
        m[f'{v}_diff'] = m['App_Points'] - m[v]

    print(f"\n{'='*80}")
    print(f"MATCHED: {len(m)} / {len(df)} Boston players")
    print('='*80)

    # Unmatched
    unmatched = df[df['Cricsheet_Name'].isna()]
    if not unmatched.empty:
        print(f"\nUNMATCHED ({len(unmatched)}):")
        for _, r in unmatched.iterrows():
            print(f"  {r['Player']} ({r['Manager']})")

    # Overall fit per variant (players that likely have complete data: matches >= 4)
    full = m[m['Matches'] >= 4].copy()
    print(f"\n── Overall fit across {len(full)} players with matches >= 4 ──")
    print(f"{'Variant':<12} {'MAE':>7} {'RMSE':>7} {'Mean Δ':>8} {'Median Δ':>9} {'Exact match':>12}")
    for v in ['V_BASE', 'V_HIGHEST', 'V_NO_SR', 'V_NO_ECON']:
        diff = full[f'{v}_diff']
        mae = diff.abs().mean()
        rmse = np.sqrt((diff**2).mean())
        exact = (diff == 0).sum()
        print(f"{v:<12} {mae:>7.1f} {rmse:>7.1f} {diff.mean():>+8.1f} {diff.median():>+9.1f} {exact:>6d} / {len(full)}")

    # Players with diff == 0 under each variant (strong signal)
    print(f"\n── Players with EXACT match (diff == 0) under each variant ──")
    for v in ['V_BASE', 'V_HIGHEST', 'V_NO_SR', 'V_NO_ECON']:
        exact = m[m[f'{v}_diff'] == 0]
        print(f"\n{v}: {len(exact)} exact matches")
        for _, r in exact.head(10).iterrows():
            print(f"  {r['Player']:25s}  app={r['App_Points']:5.0f}  matches={int(r['Matches'])}")

    # Per-player comparison: show residual pattern
    print(f"\n── Per-player residuals (App - V_BASE), sorted by absolute diff ──")
    m['abs_d'] = m['V_BASE_diff'].abs()
    for _, r in m.sort_values('abs_d', ascending=False).head(25).iterrows():
        print(f"  {r['Player']:25s} ({int(r['Matches'])}m) app={r['App_Points']:5.0f} "
              f"base={r['V_BASE']:5.0f} Δ={r['V_BASE_diff']:+5.0f}  "
              f"hi={r['V_HIGHEST']:5.0f} noSR={r['V_NO_SR']:5.0f} noEcon={r['V_NO_ECON']:5.0f}")

    # Players where V_BASE_diff > 0 AND V_NO_SR/V_NO_ECON diffs are smaller — hint missing matches
    # Players where V_BASE_diff closer to 0 vs variants — confirms V_BASE
    return m


def main():
    print("Loading Boston app results...")
    boston = load_boston()
    print(f"  {len(boston)} drafted players across {boston['Manager'].nunique()} managers")

    print("Loading computed variants...")
    variants = load_variants()
    print(f"  {len(variants)} players in variants table")

    matched = match_players(boston, variants)

    # Save matched table
    matched.to_csv(OUT_DIR / 'boston_reconciliation.csv', index=False)
    print(f"✓ Saved: {OUT_DIR / 'boston_reconciliation.csv'}")

    analyze(matched)


if __name__ == '__main__':
    main()
