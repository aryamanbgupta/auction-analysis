"""
Build XGBoost training table for IPL custom-rules fantasy projection.

For each training season S in 2012-2025:
  - Features from all seasons < S (career, last-3 lag, role prior)
  - Features from early-S matches (where both teams entering the match
    have played fewer than 6 games — per user spec)
  - Target = player's full group-stage total points in season S

Group-stage matches are identified by chronological ordering per season:
a match is group-stage if neither team has already played 14 games before it.

Output:
  - data/training_table_custom.parquet  (one row per (season, player))
"""
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
PER_MATCH = PROJECT_ROOT / 'data' / 'historical_custom_points_per_match.parquet'
RAW_BALLS = PROJECT_ROOT / 'data' / 'ipl_matches_fantasy.parquet'
OUT = PROJECT_ROOT / 'data' / 'training_table_custom.parquet'

TARGET_SEASONS = list(range(2012, 2026))  # 2012..2025 inclusive
EARLY_CUTOFF_GAMES = 6                    # "every team has played <6 games"
GROUP_STAGE_GAMES_PER_TEAM = 14
MARCEL_PRIOR_N = 100
W_S1, W_S2, W_S3 = 5, 4, 3


def build_match_team_table(balls: pd.DataFrame) -> pd.DataFrame:
    """One row per match with the two teams."""
    mt = balls.groupby(['season', 'match_id', 'match_date'])['batting_team'].unique().reset_index()
    mt['team1'] = mt['batting_team'].str[0]
    mt['team2'] = mt['batting_team'].str[1]
    return mt.drop(columns='batting_team').sort_values(['season', 'match_date', 'match_id']).reset_index(drop=True)


def tag_group_and_early(mt: pd.DataFrame) -> pd.DataFrame:
    """Add is_group_stage and is_early columns."""
    out = []
    for s, grp in mt.groupby('season'):
        grp = grp.sort_values(['match_date', 'match_id']).copy()
        team_count: dict[str, int] = {}
        is_group = []
        is_early = []
        for _, r in grp.iterrows():
            t1, t2 = r['team1'], r['team2']
            c1 = team_count.get(t1, 0)
            c2 = team_count.get(t2, 0)
            # Group stage if both teams have <14 games before this match
            gs = (c1 < GROUP_STAGE_GAMES_PER_TEAM) and (c2 < GROUP_STAGE_GAMES_PER_TEAM)
            # Early if both teams have <EARLY_CUTOFF_GAMES before this match
            early = gs and (c1 < EARLY_CUTOFF_GAMES) and (c2 < EARLY_CUTOFF_GAMES)
            is_group.append(gs)
            is_early.append(early)
            team_count[t1] = c1 + 1
            team_count[t2] = c2 + 1
        grp['is_group_stage'] = is_group
        grp['is_early'] = is_early
        out.append(grp)
    return pd.concat(out, ignore_index=True)


def infer_role(career: pd.Series) -> str:
    """Infer role from career bat/bowl/field points."""
    bat = career.get('bat_pts', 0)
    bowl = career.get('bowl_pts', 0)
    stump = career.get('stumpings', 0)
    if stump >= 3:
        return 'WK'
    total_bb = bat + bowl
    if total_bb <= 0:
        return 'AR'
    bat_share = bat / total_bb if total_bb > 0 else 0
    if bowl >= 50 and bat_share > 0.6 and bat >= 100:
        return 'AR'
    if bat_share >= 0.75:
        return 'BAT'
    if bat_share <= 0.25:
        return 'BOWL'
    return 'AR'


def build_season_aggs(pm: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-match points to per (season, player) with role-relevant stats."""
    agg = pm.groupby(['season', 'player_name', 'player_id']).agg(
        matches=('match_id', 'nunique'),
        total_pts=('total_pts', 'sum'),
        bat_pts=('bat_pts', 'sum'),
        bowl_pts=('bowl_pts', 'sum'),
        field_pts=('field_pts', 'sum'),
        runs=('runs', 'sum'),
        balls_faced=('balls_faced', 'sum'),
        wickets=('wickets', 'sum'),
        overs=('overs', 'sum'),
        catches=('catches', 'sum'),
        stumpings=('stumpings', 'sum'),
    ).reset_index()
    agg['ppm'] = agg['total_pts'] / agg['matches'].clip(lower=1)
    return agg


def main():
    print(f"Loading per-match: {PER_MATCH}")
    pm = pd.read_parquet(PER_MATCH)
    print(f"  {len(pm):,} rows")

    print(f"Loading balls (for team → match mapping): {RAW_BALLS}")
    balls = pd.read_parquet(RAW_BALLS)
    mt = build_match_team_table(balls)
    mt = tag_group_and_early(mt)
    print(f"  {len(mt):,} matches tagged")
    print(mt.groupby('season').agg(
        total=('match_id', 'count'),
        group=('is_group_stage', 'sum'),
        early=('is_early', 'sum'),
    ))

    # Merge stage flags onto per-match, then filter to group-stage for ALL training
    pm = pm.merge(mt[['match_id', 'is_group_stage', 'is_early']], on='match_id', how='left')
    pm['is_group_stage'] = pm['is_group_stage'].fillna(False)
    pm['is_early'] = pm['is_early'].fillna(False)

    pm_group = pm[pm['is_group_stage']].copy()
    print(f"\n  {len(pm_group):,} player-match rows in group stage "
          f"({len(pm_group) / len(pm):.1%})")

    rows = []
    for S in TARGET_SEASONS:
        pre = pm_group[pm_group['season'] < S]
        this = pm_group[pm_group['season'] == S]
        if len(this) == 0:
            print(f"  season {S}: no data, skipping")
            continue

        # Early-season stats for S
        early = this[this['is_early']]
        full_season = this  # group stage = full season per user spec

        # Career aggregates pre-S
        career = pre.groupby(['player_name', 'player_id']).agg(
            career_matches=('match_id', 'nunique'),
            career_total=('total_pts', 'sum'),
            career_bat=('bat_pts', 'sum'),
            career_bowl=('bowl_pts', 'sum'),
            career_field=('field_pts', 'sum'),
            career_runs=('runs', 'sum'),
            career_balls=('balls_faced', 'sum'),
            career_wickets=('wickets', 'sum'),
            career_overs=('overs', 'sum'),
            career_catches=('catches', 'sum'),
            career_stumpings=('stumpings', 'sum'),
        ).reset_index()
        career['career_ppm'] = career['career_total'] / career['career_matches'].clip(lower=1)
        career['role'] = career.apply(
            lambda r: infer_role({
                'bat_pts': r['career_bat'], 'bowl_pts': r['career_bowl'],
                'stumpings': r['career_stumpings'],
            }), axis=1
        )

        # Per-season aggregates (for lag features)
        per_season = pre.groupby(['season', 'player_name', 'player_id']).agg(
            matches=('match_id', 'nunique'),
            total=('total_pts', 'sum'),
        ).reset_index()
        per_season['ppm'] = per_season['total'] / per_season['matches'].clip(lower=1)

        def lag_for(lag):
            lag_s = S - lag
            d = per_season[per_season['season'] == lag_s][['player_name', 'player_id', 'matches', 'ppm']]
            return d.rename(columns={'matches': f'lag{lag}_matches', 'ppm': f'lag{lag}_ppm'})

        lag1 = lag_for(1)
        lag2 = lag_for(2)
        lag3 = lag_for(3)

        # Role prior PPM: mean PPM across pre-S seasons, weighted by matches, per role
        # Use career aggregates grouped by inferred role
        role_prior = career.groupby('role').apply(
            lambda g: (g['career_total'].sum() / g['career_matches'].sum())
        ).to_dict()
        # Fallback if role missing
        default_prior = career['career_total'].sum() / max(career['career_matches'].sum(), 1)

        # Early-S per-player aggregates
        early_agg = early.groupby(['player_name', 'player_id']).agg(
            early_matches=('match_id', 'nunique'),
            early_total=('total_pts', 'sum'),
            early_bat=('bat_pts', 'sum'),
            early_bowl=('bowl_pts', 'sum'),
            early_field=('field_pts', 'sum'),
            early_runs=('runs', 'sum'),
            early_balls=('balls_faced', 'sum'),
            early_wickets=('wickets', 'sum'),
            early_overs=('overs', 'sum'),
        ).reset_index()
        early_agg['early_ppm'] = early_agg['early_total'] / early_agg['early_matches'].clip(lower=1)

        # Full-season target
        target = full_season.groupby(['player_name', 'player_id']).agg(
            target_matches=('match_id', 'nunique'),
            target_total=('total_pts', 'sum'),
        ).reset_index()
        target['target_ppm'] = target['target_total'] / target['target_matches'].clip(lower=1)

        # Union of players: anyone with pre-S history OR early-S appearance OR full-S appearance
        all_players = pd.concat([
            career[['player_name', 'player_id']],
            early_agg[['player_name', 'player_id']],
            target[['player_name', 'player_id']],
        ], ignore_index=True).drop_duplicates().reset_index(drop=True)

        df = all_players.merge(career, on=['player_name', 'player_id'], how='left')
        df = df.merge(lag1, on=['player_name', 'player_id'], how='left')
        df = df.merge(lag2, on=['player_name', 'player_id'], how='left')
        df = df.merge(lag3, on=['player_name', 'player_id'], how='left')
        df = df.merge(early_agg, on=['player_name', 'player_id'], how='left')
        df = df.merge(target, on=['player_name', 'player_id'], how='left')

        # Fill NaNs
        num_zero = [
            'career_matches', 'career_total', 'career_bat', 'career_bowl',
            'career_field', 'career_runs', 'career_balls', 'career_wickets',
            'career_overs', 'career_catches', 'career_stumpings',
            'lag1_matches', 'lag2_matches', 'lag3_matches',
            'early_matches', 'early_total', 'early_bat', 'early_bowl',
            'early_field', 'early_runs', 'early_balls', 'early_wickets',
            'early_overs',
        ]
        for c in num_zero:
            df[c] = df[c].fillna(0)
        df['career_ppm'] = df['career_ppm'].fillna(0)
        df['lag1_ppm'] = df['lag1_ppm'].fillna(0)
        df['lag2_ppm'] = df['lag2_ppm'].fillna(0)
        df['lag3_ppm'] = df['lag3_ppm'].fillna(0)
        df['early_ppm'] = df['early_ppm'].fillna(0)

        # For players with no pre-S role, reinfer from early-S
        miss = df['role'].isna()
        if miss.any():
            df.loc[miss, 'role'] = df.loc[miss].apply(
                lambda r: infer_role({
                    'bat_pts': r['early_bat'], 'bowl_pts': r['early_bowl'],
                    'stumpings': 0,
                }), axis=1
            )
        df['role'] = df['role'].fillna('AR')
        df['role_prior_ppm'] = df['role'].map(role_prior).fillna(default_prior)

        # Marcel PPM from lags + prior
        num = (W_S1 * df['lag1_ppm'] * df['lag1_matches']
               + W_S2 * df['lag2_ppm'] * df['lag2_matches']
               + W_S3 * df['lag3_ppm'] * df['lag3_matches']
               + MARCEL_PRIOR_N * df['role_prior_ppm'])
        denom = (W_S1 * df['lag1_matches']
                 + W_S2 * df['lag2_matches']
                 + W_S3 * df['lag3_matches']
                 + MARCEL_PRIOR_N)
        df['marcel_ppm'] = num / denom

        # Component rates (career)
        df['career_bat_ppm'] = df['career_bat'] / df['career_matches'].clip(lower=1)
        df['career_bowl_ppm'] = df['career_bowl'] / df['career_matches'].clip(lower=1)
        df['career_field_ppm'] = df['career_field'] / df['career_matches'].clip(lower=1)

        df['early_bat_ppm'] = df['early_bat'] / df['early_matches'].clip(lower=1)
        df['early_bowl_ppm'] = df['early_bowl'] / df['early_matches'].clip(lower=1)
        df['early_field_ppm'] = df['early_field'] / df['early_matches'].clip(lower=1)

        # Target: full-season total (drop players with NaN target — they didn't play in S)
        df['season'] = S
        df['played_in_S'] = df['target_matches'].notna()
        df['target_total'] = df['target_total'].fillna(0)
        df['target_matches'] = df['target_matches'].fillna(0)
        df['target_ppm'] = df['target_ppm'].fillna(0)

        rows.append(df)
        print(f"  season {S}: {len(df):,} player rows "
              f"(played={df['played_in_S'].sum()}, target_mean={df.loc[df['played_in_S'], 'target_total'].mean():.1f})")

    table = pd.concat(rows, ignore_index=True)
    table = table.sort_values(['season', 'target_total'], ascending=[True, False]).reset_index(drop=True)
    print(f"\nTotal training rows: {len(table):,}")
    print(f"Columns: {table.columns.tolist()}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUT, index=False)
    print(f"✓ {OUT}")

    # Sanity check: print top 5 per last few seasons
    print("\n── Top 5 target totals per season (sanity) ──")
    for s in sorted(table['season'].unique())[-4:]:
        top = table[(table['season'] == s) & table['played_in_S']].nlargest(5, 'target_total')
        names = ', '.join(f"{r.player_name}({int(r.target_total)})" for _, r in top.iterrows())
        print(f"  {s}: {names}")


if __name__ == '__main__':
    main()
