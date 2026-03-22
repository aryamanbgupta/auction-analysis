"""
Feature Engineering for Fantasy Points Projection.

Builds ML features per player-season to predict next season's avg fantasy pts/match.

Reuses WAR pipeline concepts (lags, Marcel weighting, phase splits, form, opponent quality)
and adds fantasy-specific features (boundary rate, SR bonus rate, economy bonus rate, etc.).

OUTPUT: data/fantasy_features.csv
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'WARprojections'))
from utils import save_dataframe


# ── Marcel weighting ───────────────────────────────────────────────────────

MARCEL_WEIGHTS = [5, 4, 3]  # most recent season first


def marcel_weighted_avg(lag1, lag2, lag3, w=MARCEL_WEIGHTS):
    """Weighted average with Marcel 5/4/3 weighting, ignoring NaN lags."""
    vals = [lag1, lag2, lag3]
    total_w, total_v = 0, 0
    for v, wt in zip(vals, w):
        if pd.notna(v):
            total_v += v * wt
            total_w += wt
    return total_v / total_w if total_w > 0 else np.nan


# ── Data loading ───────────────────────────────────────────────────────────

def load_data():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'

    season = pd.read_csv(data_dir / 'fantasy_points_per_season.csv')
    match = pd.read_csv(data_dir / 'fantasy_points_per_match.csv')
    ipl = pd.read_parquet(data_dir / 'ipl_matches_fantasy.parquet')

    # Player metadata for age
    meta_path = data_dir / 'player_metadata.csv'
    meta = pd.read_csv(meta_path) if meta_path.exists() else None

    return season, match, ipl, meta


# ── Career & lag features ──────────────────────────────────────────────────

def build_career_features(season_df: pd.DataFrame) -> pd.DataFrame:
    """Build lag, career trajectory, and Marcel-weighted features."""
    df = season_df.sort_values(['player_id', 'season']).copy()

    # Lag features (1, 2, 3 seasons back)
    for lag in [1, 2, 3]:
        shifted = df.groupby('player_id')[
            ['avg_fantasy_pts', 'total_fantasy_pts', 'matches',
             'batting_pts_share', 'bowling_pts_share', 'fielding_pts_share',
             'std_fantasy_pts']
        ].shift(lag)
        for col in shifted.columns:
            df[f'{col}_lag{lag}'] = shifted[col]

    # Marcel-weighted avg fantasy pts
    df['fantasy_pts_weighted'] = df.apply(
        lambda r: marcel_weighted_avg(
            r.get('avg_fantasy_pts_lag1'),
            r.get('avg_fantasy_pts_lag2'),
            r.get('avg_fantasy_pts_lag3'),
        ), axis=1
    )

    # Career cumulative (entering this season)
    df['career_fantasy_pts'] = df.groupby('player_id')['total_fantasy_pts'].cumsum() - df['total_fantasy_pts']
    df['career_matches'] = df.groupby('player_id')['matches'].cumsum() - df['matches']

    # Years played
    df['years_played'] = df.groupby('player_id').cumcount()

    # Career avg (entering this season)
    df['career_avg_fantasy'] = np.where(
        df['career_matches'] > 0,
        df['career_fantasy_pts'] / df['career_matches'],
        np.nan
    )

    return df


# ── Phase-specific features ────────────────────────────────────────────────

def build_phase_features(ipl: pd.DataFrame) -> pd.DataFrame:
    """Calculate phase-specific (powerplay/middle/death) fantasy-relevant stats per player-season."""
    ipl = ipl.copy()
    legal = ipl[ipl['wides'] == 0].copy()

    records = []

    # Batting phase stats
    for (pid, pname, season, phase), grp in legal.groupby(
        ['batter_id', 'batter_name', 'season', 'phase']
    ):
        balls = len(grp)
        runs = grp['batter_runs'].sum()
        fours = (grp['batter_runs'] == 4).sum()
        sixes = (grp['batter_runs'] == 6).sum()
        sr = (runs / balls * 100) if balls > 0 else 0

        records.append({
            'player_id': pid, 'player_name': pname, 'season': season,
            f'bat_{phase}_balls': balls,
            f'bat_{phase}_sr': sr,
            f'bat_{phase}_boundary_rate': (fours + sixes) / balls if balls > 0 else 0,
            f'bat_{phase}_six_rate': sixes / balls if balls > 0 else 0,
        })

    bat_phase = pd.DataFrame(records)
    # Pivot so each player-season has one row
    bat_phase = bat_phase.groupby(['player_id', 'player_name', 'season']).first().reset_index()

    # Bowling phase stats
    bowl_records = []
    bowl_legal = ipl[(ipl['wides'] == 0) & (ipl['noballs'] == 0)]
    for (pid, pname, season, phase), grp in ipl.groupby(
        ['bowler_id', 'bowler_name', 'season', 'phase']
    ):
        balls = len(grp[(grp['wides'] == 0) & (grp['noballs'] == 0)])
        total_balls_incl_extras = len(grp)
        runs_conceded = grp['total_runs'].sum() - grp['byes'].sum() - grp['legbyes'].sum()
        wickets = grp['is_wicket'].sum()
        dots = (grp['total_runs'] == 0).sum()

        overs = balls / 6 if balls > 0 else 0
        econ = runs_conceded / overs if overs > 0 else 0

        bowl_records.append({
            'player_id': pid, 'player_name': pname, 'season': season,
            f'bowl_{phase}_balls': balls,
            f'bowl_{phase}_econ': econ,
            f'bowl_{phase}_dot_rate': dots / total_balls_incl_extras if total_balls_incl_extras > 0 else 0,
            f'bowl_{phase}_wicket_rate': wickets / balls if balls > 0 else 0,
        })

    bowl_phase = pd.DataFrame(bowl_records)
    bowl_phase = bowl_phase.groupby(['player_id', 'player_name', 'season']).first().reset_index()

    # Merge bat + bowl phase
    phase = bat_phase.merge(bowl_phase, on=['player_id', 'player_name', 'season'], how='outer')
    return phase


# ── Fantasy-specific rate features ─────────────────────────────────────────

def build_fantasy_rate_features(match_df: pd.DataFrame, ipl: pd.DataFrame) -> pd.DataFrame:
    """Build fantasy-specific features: boundary rate, SR bonus rate, economy bonus rate, catches/match."""

    # ── From match-level fantasy data ──────────────────────────────────
    match = match_df.copy()
    match['strike_rate'] = np.where(
        match['balls_faced'] > 0,
        match['runs'] / match['balls_faced'] * 100,
        0
    )

    # SR bonus earned rate (batting, min 10 balls)
    match['sr_bonus_earned'] = 0
    mask_10 = match['balls_faced'] >= 10
    match.loc[mask_10 & (match['strike_rate'] >= 130), 'sr_bonus_earned'] = 1
    match.loc[mask_10 & (match['strike_rate'] < 70), 'sr_bonus_earned'] = -1

    # Economy bonus earned rate (bowling, min 2 overs)
    match['econ_bonus_earned'] = 0
    mask_2ov = match['overs'] >= 2
    econ = np.where(match['overs'] > 0, match['runs_conceded'] / match['overs'], 99)
    match.loc[mask_2ov & (econ <= 7), 'econ_bonus_earned'] = 1
    match.loc[mask_2ov & (econ > 10), 'econ_bonus_earned'] = -1

    season_rates = match.groupby(['season', 'player_name', 'player_id']).agg(
        boundary_rate=('fours', lambda x: x.sum() / max(match.loc[x.index, 'balls_faced'].sum(), 1)),
        six_rate=('sixes', lambda x: x.sum() / max(match.loc[x.index, 'balls_faced'].sum(), 1)),
        sr_bonus_rate=('sr_bonus_earned', 'mean'),
        econ_bonus_rate=('econ_bonus_earned', 'mean'),
        catches_per_match=('catches', 'mean'),
        wicket_rate_match=('wickets', lambda x: x.sum() / max(match.loc[x.index, 'overs'].sum() * 6, 1)),
        maiden_rate=('maidens', lambda x: x.sum() / max(match.loc[x.index, 'overs'].sum(), 1)),
        avg_balls_faced=('balls_faced', 'mean'),
        avg_overs_bowled=('overs', 'mean'),
    ).reset_index()

    return season_rates


# ── Rolling form features (match-level fantasy points) ─────────────────────

def build_form_features(match_df: pd.DataFrame, ipl: pd.DataFrame) -> pd.DataFrame:
    """Calculate rolling form using match-level fantasy points."""
    # Get match dates
    match_dates = ipl.groupby('match_id')['match_date'].first().reset_index()
    match_dates['match_date'] = pd.to_datetime(match_dates['match_date'])

    match = match_df.merge(match_dates, on='match_id', how='left')
    match = match.sort_values(['player_id', 'match_date'])

    results = []
    for (pid, pname, season), grp in match.groupby(['player_id', 'player_name', 'season']):
        season_start = grp['match_date'].min()
        history = match[
            (match['player_id'] == pid) & (match['match_date'] < season_start)
        ].sort_values('match_date', ascending=False)

        if len(history) == 0:
            results.append({
                'player_id': pid, 'player_name': pname, 'season': season,
                'fp_last_5': np.nan, 'fp_last_10': np.nan, 'fp_last_15': np.nan,
                'fp_decay_form': np.nan, 'fp_volatility': np.nan, 'fp_trend': np.nan,
            })
            continue

        fp = history['total_fantasy_points'].values
        last_5 = fp[:5].mean() if len(fp) >= 5 else np.nan
        last_10 = fp[:10].mean() if len(fp) >= 10 else np.nan
        last_15 = fp[:15].mean() if len(fp) >= 10 else np.nan

        # Exponential decay
        n = min(len(fp), 30)
        w = np.exp(-np.arange(n) * 0.1)
        w /= w.sum()
        decay_form = (fp[:n] * w).sum()

        volatility = fp[:10].std() if len(fp) >= 5 else np.nan

        if len(fp) >= 10:
            trend = fp[:5].mean() - fp[5:10].mean()
        else:
            trend = np.nan

        results.append({
            'player_id': pid, 'player_name': pname, 'season': season,
            'fp_last_5': last_5, 'fp_last_10': last_10, 'fp_last_15': last_15,
            'fp_decay_form': decay_form, 'fp_volatility': volatility, 'fp_trend': trend,
        })

    return pd.DataFrame(results)


# ── Opponent quality adjustment ────────────────────────────────────────────

def build_opponent_features(match_df: pd.DataFrame, ipl: pd.DataFrame) -> pd.DataFrame:
    """Opponent-adjusted fantasy points (against stronger teams = worth more)."""
    # Team strength per season (net run rate proxy)
    team_strength = {}
    for season in ipl['season'].unique():
        sd = ipl[ipl['season'] == season]
        for team in sd['batting_team'].unique():
            batting = sd[sd['batting_team'] == team]
            bowling = sd[sd['bowling_team'] == team]
            if len(batting) > 0 and len(bowling) > 0:
                sr = batting['total_runs'].sum() / max(len(batting), 1) * 6
                cr = bowling['total_runs'].sum() / max(len(bowling), 1) * 6
                strength = max(0.2, min(0.8, (sr - cr + 2) / 4))
            else:
                strength = 0.5
            team_strength[(team, season)] = strength

    # Get opponent for each match-player
    # For batters: opponent = bowling_team; for bowlers: opponent = batting_team
    # Since we have unified players, approximate by looking at which team they're on
    match_dates = ipl.groupby('match_id')[['match_date', 'season']].first().reset_index()
    match_dates['match_date'] = pd.to_datetime(match_dates['match_date'])

    # Get player teams from ball-by-ball
    player_teams = ipl.groupby(['match_id', 'batter_name', 'batter_id']).agg(
        team=('batting_team', 'first'),
        opponent=('bowling_team', 'first'),
    ).reset_index().rename(columns={'batter_name': 'player_name', 'batter_id': 'player_id'})

    bowler_teams = ipl.groupby(['match_id', 'bowler_name', 'bowler_id']).agg(
        team=('bowling_team', 'first'),
        opponent=('batting_team', 'first'),
    ).reset_index().rename(columns={'bowler_name': 'player_name', 'bowler_id': 'player_id'})

    all_teams = pd.concat([player_teams[['match_id', 'player_name', 'player_id', 'opponent']],
                           bowler_teams[['match_id', 'player_name', 'player_id', 'opponent']]]
                          ).drop_duplicates(subset=['match_id', 'player_id'])

    match_fp = match_df.merge(all_teams, on=['match_id', 'player_name', 'player_id'], how='left')
    match_fp = match_fp.merge(match_dates[['match_id', 'season']], on='match_id', how='left', suffixes=('', '_md'))

    # Use season from match if available
    if 'season_md' in match_fp.columns:
        match_fp['season'] = match_fp['season'].fillna(match_fp['season_md'])
        match_fp = match_fp.drop(columns=['season_md'])

    match_fp['opp_strength'] = match_fp.apply(
        lambda r: team_strength.get((r.get('opponent', ''), r.get('season', 0)), 0.5), axis=1
    )
    match_fp['opp_multiplier'] = 0.8 + match_fp['opp_strength'] * 0.5
    match_fp['opp_adj_fp'] = match_fp['total_fantasy_points'] * match_fp['opp_multiplier']

    opp_agg = match_fp.groupby(['player_id', 'player_name', 'season']).agg(
        opp_adj_fp_total=('opp_adj_fp', 'sum'),
        opp_adj_fp_avg=('opp_adj_fp', 'mean'),
    ).reset_index()

    return opp_agg


# ── Player role detection ──────────────────────────────────────────────────

def detect_role(row):
    """Detect player role from batting/bowling point shares."""
    bat_share = row.get('batting_pts_share', 0)
    bowl_share = row.get('bowling_pts_share', 0)
    stumpings = row.get('total_stumpings', 0)

    if stumpings > 0:
        return 'WK'
    if bat_share > 0.6 and bowl_share < 0.15:
        return 'BAT'
    if bowl_share > 0.5 and bat_share < 0.25:
        return 'BOWL'
    return 'AR'


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    project_root = Path(__file__).parent.parent

    print("=" * 60)
    print("FANTASY POINTS FEATURE ENGINEERING")
    print("=" * 60)

    season_df, match_df, ipl, meta = load_data()
    print(f"Loaded: {len(season_df)} player-seasons, {len(match_df)} player-matches")

    # 1. Career & lag features
    print("\n1. Building career & lag features...")
    features = build_career_features(season_df)
    print(f"   {len(features)} rows, {len(features.columns)} columns")

    # 2. Phase features
    print("2. Building phase features...")
    phase = build_phase_features(ipl)
    features = features.merge(phase, on=['player_id', 'player_name', 'season'], how='left')
    print(f"   → {len(features.columns)} columns")

    # 3. Fantasy rate features
    print("3. Building fantasy rate features...")
    rates = build_fantasy_rate_features(match_df, ipl)
    features = features.merge(rates, on=['season', 'player_name', 'player_id'], how='left')
    print(f"   → {len(features.columns)} columns")

    # 4. Form features
    print("4. Building rolling form features...")
    form = build_form_features(match_df, ipl)
    features = features.merge(form, on=['player_id', 'player_name', 'season'], how='left')
    print(f"   → {len(features.columns)} columns")

    # 5. Opponent quality
    print("5. Building opponent quality features...")
    opp = build_opponent_features(match_df, ipl)
    features = features.merge(opp, on=['player_id', 'player_name', 'season'], how='left')
    print(f"   → {len(features.columns)} columns")

    # 6. Role detection
    print("6. Detecting player roles...")
    features['role'] = features.apply(detect_role, axis=1)
    print(f"   Role distribution: {features['role'].value_counts().to_dict()}")

    # 7. Age (if metadata available)
    if meta is not None and 'dob' in meta.columns:
        print("7. Adding age feature...")
        meta_slim = meta[['player_id', 'dob']].drop_duplicates(subset='player_id')
        meta_slim['dob'] = pd.to_datetime(meta_slim['dob'], errors='coerce')
        features = features.merge(meta_slim, on='player_id', how='left')
        features['age'] = features['season'] - features['dob'].dt.year
        features.loc[features['age'] < 15, 'age'] = np.nan
        features.loc[features['age'] > 45, 'age'] = np.nan
    else:
        print("7. No metadata with dob found, skipping age")
        features['age'] = np.nan

    # 8. Target variable: avg fantasy pts NEXT season
    print("8. Creating target variable (avg_fantasy_pts_next_season)...")
    next_season = features[['player_id', 'season', 'avg_fantasy_pts']].copy()
    next_season['season'] = next_season['season'] - 1
    next_season = next_season.rename(columns={'avg_fantasy_pts': 'target_avg_fp_next'})
    features = features.merge(next_season, on=['player_id', 'season'], how='left')

    n_with_target = features['target_avg_fp_next'].notna().sum()
    print(f"   {n_with_target} rows with target (of {len(features)})")

    # 9. Summary
    print(f"\n{'='*60}")
    print(f"FEATURE ENGINEERING COMPLETE")
    print(f"{'='*60}")
    print(f"Total rows: {len(features)}")
    print(f"Total columns: {len(features.columns)}")
    print(f"Seasons: {sorted(features['season'].unique())}")

    # Key feature columns
    feature_cols = [c for c in features.columns if c not in
                    ['player_id', 'player_name', 'season', 'match_id', 'dob',
                     'target_avg_fp_next']]
    print(f"Feature columns ({len(feature_cols)}):")
    for c in sorted(feature_cols):
        non_null = features[c].notna().sum()
        print(f"  {c:40s}  non-null: {non_null:5d}  mean: {features[c].mean():8.2f}" if features[c].dtype in ['float64', 'int64', 'float32'] else f"  {c:40s}  non-null: {non_null:5d}")

    # Save
    output_path = project_root / 'data' / 'fantasy_features.csv'
    save_dataframe(features, output_path, format='csv')
    print(f"\n✓ Saved to {output_path}")


if __name__ == '__main__':
    main()
