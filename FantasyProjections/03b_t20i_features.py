"""
Engineer inter-season T20I features for each (player_id, ipl_season).

For each IPL season Y, computes features from T20I matches in the window
between the end of IPL season Y-1 and the start of IPL season Y.

OUTPUT: data/t20i_fantasy_features.csv
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'WARprojections'))
from utils import save_dataframe

# Elite T20I teams (same as WARprojections/06_league_strength.py)
STRONG_TEAMS = {
    'India', 'Australia', 'England', 'Pakistan', 'South Africa',
    'New Zealand', 'West Indies', 'Sri Lanka', 'Bangladesh', 'Afghanistan'
}


def compute_ipl_windows(ipl_parquet_path):
    """Compute IPL season start/end dates from actual match data."""
    ipl = pd.read_parquet(ipl_parquet_path, columns=['season', 'match_date'])
    ipl['match_date'] = pd.to_datetime(ipl['match_date'], errors='coerce')
    ipl = ipl.dropna(subset=['match_date'])

    windows = {}
    for season, grp in ipl.groupby('season'):
        windows[int(season)] = {
            'start': grp['match_date'].min(),
            'end': grp['match_date'].max(),
        }
    return windows


def get_t20i_window(ipl_season, ipl_windows):
    """Get (window_start, window_end) for T20I matches before an IPL season.

    Window = day after previous IPL ended → day before current IPL starts.
    """
    prev_season = ipl_season - 1

    if prev_season in ipl_windows:
        window_start = ipl_windows[prev_season]['end'] + pd.Timedelta(days=1)
    else:
        # No previous IPL data — use Jan 1 of current year as fallback
        window_start = pd.Timestamp(f'{ipl_season}-01-01')

    if ipl_season in ipl_windows:
        window_end = ipl_windows[ipl_season]['start'] - pd.Timedelta(days=1)
    else:
        # No current IPL data — use Mar 31 as default
        window_end = pd.Timestamp(f'{ipl_season}-03-31')

    return window_start, window_end


def decay_weighted_avg(values, decay=0.1):
    """Exponentially recency-weighted average (most recent = highest weight)."""
    if len(values) == 0:
        return np.nan
    n = len(values)
    weights = np.exp(-decay * np.arange(n)[::-1])  # recent gets higher weight
    return np.average(values, weights=weights)


def main():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'

    print("=" * 60)
    print("T20I INTER-SEASON FEATURE ENGINEERING")
    print("=" * 60)

    # Load T20I match-level fantasy points
    t20i = pd.read_csv(data_dir / 't20i_fantasy_points_per_match.csv')
    t20i['match_date'] = pd.to_datetime(t20i['match_date'], errors='coerce')
    t20i = t20i.dropna(subset=['match_date'])
    print(f"T20I match records: {len(t20i):,}")

    # Load IPL season data for the player-season index
    ipl_season = pd.read_csv(data_dir / 'fantasy_points_per_season.csv')
    print(f"IPL player-seasons: {len(ipl_season):,}")

    # Compute IPL windows from actual match dates
    ipl_windows = compute_ipl_windows(data_dir / 'ipl_matches_fantasy.parquet')
    print(f"IPL seasons with date info: {sorted(ipl_windows.keys())}")

    # Sort T20I by date for decay calculations
    t20i = t20i.sort_values('match_date')

    # Mark elite T20I matches
    t20i['is_elite'] = t20i.apply(
        lambda r: (str(r.get('team1', '')) in STRONG_TEAMS and
                   str(r.get('team2', '')) in STRONG_TEAMS),
        axis=1
    )

    # ── Build features for each (player_id, ipl_season) ─────────────
    all_ipl_seasons = sorted(ipl_season['season'].unique())
    # Only build features for seasons where we have a preceding T20I window
    target_seasons = [s for s in all_ipl_seasons if s >= 2009]

    records = []

    for ipl_s in target_seasons:
        window_start, window_end = get_t20i_window(ipl_s, ipl_windows)

        # Filter T20I matches in this window
        mask = (t20i['match_date'] >= window_start) & (t20i['match_date'] <= window_end)
        window_t20i = t20i[mask]

        if len(window_t20i) == 0:
            continue

        # Get IPL previous season avg for ratio features
        prev_ipl = ipl_season[ipl_season['season'] == ipl_s - 1]
        prev_avg_map = dict(zip(prev_ipl['player_id'], prev_ipl['avg_fantasy_pts']))

        # Group by player
        for pid, grp in window_t20i.groupby('player_id'):
            grp_sorted = grp.sort_values('match_date')
            n_matches = len(grp_sorted)
            fps = grp_sorted['total_fantasy_points'].values

            rec = {
                'player_id': pid,
                'season': ipl_s,
                # Volume
                't20i_matches_interseason': n_matches,
                # Fantasy point aggregates
                't20i_avg_fantasy_pts': fps.mean(),
                't20i_std_fantasy_pts': fps.std() if n_matches > 1 else 0,
                't20i_max_fantasy_pts': fps.max(),
                't20i_decay_form': decay_weighted_avg(fps),
                # Batting
                't20i_batting_avg_pts': grp_sorted['batting_points'].mean(),
                # Bowling
                't20i_bowling_avg_pts': grp_sorted['bowling_points'].mean(),
            }

            # Batting details (from raw ball stats)
            total_runs = grp_sorted['runs'].sum()
            total_balls = grp_sorted['balls_faced'].sum()
            total_fours = grp_sorted['fours'].sum()
            total_sixes = grp_sorted['sixes'].sum()
            if total_balls > 0:
                rec['t20i_batting_sr'] = (total_runs / total_balls) * 100
                rec['t20i_boundary_rate'] = (total_fours + total_sixes) / total_balls
            else:
                rec['t20i_batting_sr'] = np.nan
                rec['t20i_boundary_rate'] = np.nan

            # Bowling details
            total_overs = grp_sorted['overs'].sum()
            total_wickets = grp_sorted['wickets'].sum()
            total_runs_conceded = grp_sorted['runs_conceded'].sum()
            total_legal_balls = grp_sorted['legal_balls'].sum()
            if total_overs > 0:
                rec['t20i_bowling_econ'] = total_runs_conceded / total_overs
            else:
                rec['t20i_bowling_econ'] = np.nan
            if total_legal_balls > 0:
                rec['t20i_wicket_rate'] = total_wickets / total_legal_balls
            else:
                rec['t20i_wicket_rate'] = np.nan

            # Cross-competition signals
            prev_ipl_avg = prev_avg_map.get(pid)
            if prev_ipl_avg and prev_ipl_avg > 0:
                rec['t20i_vs_ipl_ratio'] = fps.mean() / prev_ipl_avg
                rec['t20i_form_momentum'] = fps.mean() - prev_ipl_avg
            else:
                rec['t20i_vs_ipl_ratio'] = np.nan
                rec['t20i_form_momentum'] = np.nan

            # Recency
            last_t20i_date = grp_sorted['match_date'].max()
            if ipl_s in ipl_windows:
                ipl_start = ipl_windows[ipl_s]['start']
                rec['t20i_recency_days'] = (ipl_start - last_t20i_date).days
            else:
                rec['t20i_recency_days'] = np.nan

            # Elite share
            elite_count = grp_sorted['is_elite'].sum()
            rec['t20i_elite_share'] = elite_count / n_matches

            records.append(rec)

    features_df = pd.DataFrame(records)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("T20I FEATURE SUMMARY")
    print("=" * 60)
    print(f"Total player-season records: {len(features_df)}")
    print(f"Unique players: {features_df['player_id'].nunique()}")
    print(f"Seasons covered: {sorted(features_df['season'].unique())}")

    # Coverage: how many IPL player-seasons have T20I data?
    ipl_ps = set(zip(ipl_season['player_id'], ipl_season['season']))
    t20i_ps = set(zip(features_df['player_id'], features_df['season']))
    overlap = ipl_ps & t20i_ps
    print(f"\nIPL player-seasons: {len(ipl_ps)}")
    print(f"With T20I data: {len(overlap)} ({100*len(overlap)/len(ipl_ps):.1f}%)")

    # Feature stats
    print(f"\nFeature stats:")
    feat_cols = [c for c in features_df.columns if c.startswith('t20i_')]
    for col in feat_cols:
        non_null = features_df[col].notna().sum()
        mean = features_df[col].mean()
        print(f"  {col:35s}  non-null: {non_null:5d}  mean: {mean:8.2f}")

    # Top T20I performers heading into recent IPL seasons
    recent = features_df[features_df['season'] >= 2023]
    if len(recent) > 0:
        # Merge names
        name_map = dict(zip(
            t20i[['player_id', 'player_name']].drop_duplicates()['player_id'],
            t20i[['player_id', 'player_name']].drop_duplicates()['player_name'],
        ))
        print(f"\nTop T20I performers (2023+ windows):")
        top = recent.nlargest(15, 't20i_avg_fantasy_pts')
        for _, r in top.iterrows():
            name = name_map.get(r['player_id'], r['player_id'])
            print(f"  {name:25s}  IPL {int(r['season'])}  "
                  f"T20I avg: {r['t20i_avg_fantasy_pts']:5.1f}  "
                  f"matches: {int(r['t20i_matches_interseason'])}  "
                  f"form: {r['t20i_decay_form']:5.1f}")

    output_path = data_dir / 't20i_fantasy_features.csv'
    save_dataframe(features_df, output_path, format='csv')
    print(f"\n✓ Saved to {output_path}")


if __name__ == '__main__':
    main()
