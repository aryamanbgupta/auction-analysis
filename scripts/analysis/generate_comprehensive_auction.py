"""
Comprehensive 2026 IPL Auction Analytics.

Generates a master table combining:
- WAR projections (V9 Production, Marcel, Global)
- Price predictions (VOMAM, VOPE, XGBoost)
- Phase-by-phase batting stats (SR, runs, balls, RAA) for IPL and All Cricket
- Phase-by-phase bowling stats (Econ, wickets, SR, RAA) for IPL and All Cricket

Output: results/analysis/auction_2026_comprehensive.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import difflib
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Phase definitions (overs, 0-indexed)
PHASES = {
    'pp': (0, 6),      # Powerplay: overs 0-5 (ball overs 0-5.x)
    'mid': (6, 16),    # Middle: overs 6-15
    'death': (16, 20)  # Death: overs 16-19
}


# ============================================================================
# UTILITY FUNCTIONS  
# ============================================================================

def normalize_name(name):
    """Normalize player name for matching."""
    if pd.isna(name):
        return ""
    return str(name).strip().lower()


def fuzzy_match(name, candidates, threshold=0.8):
    """Find best fuzzy match for a name in candidates."""
    name_norm = normalize_name(name)
    best_match = None
    best_score = 0
    
    for candidate in candidates:
        cand_norm = normalize_name(candidate)
        score = difflib.SequenceMatcher(None, name_norm, cand_norm).ratio()
        if score > best_score and score >= threshold:
            best_score = score
            best_match = candidate
    
    return best_match, best_score


def get_phase(over):
    """Determine phase from over number."""
    if over < 6:
        return 'powerplay'
    elif over < 16:
        return 'middle'
    else:
        return 'death'


def normalize_phase(phase):
    """Normalize phase names to standard format."""
    phase_map = {
        'pp': 'powerplay',
        'mid': 'middle', 
        'death': 'death',
        'powerplay': 'powerplay',
        'middle': 'middle'
    }
    return phase_map.get(str(phase).lower(), phase)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_auction_pool():
    """Load 2026 auction player pool."""
    print("Loading auction pool...")
    path = PROJECT_ROOT / 'data' / 'ipl_2026_auction_enriched.csv'
    df = pd.read_csv(path)
    
    # Standardize column names
    df = df.rename(columns={
        'PLAYER': 'player',
        'COUNTRY': 'country_display',
        'C/U/A': 'capped',
        'BASE PRICE (INR LAKH)': 'base_price_lakh',
        'SR. NO.': 'sr_no',
        'playing_role': 'role'
    })
    
    # Handle missing cricsheet_id
    df['cricsheet_id'] = df['cricsheet_id'].fillna('')
    df['name_norm'] = df['player'].apply(normalize_name)
    
    print(f"  Loaded {len(df)} auction players")
    return df


def load_ipl_data():
    """Load IPL ball-by-ball data."""
    print("Loading IPL match data...")
    path = PROJECT_ROOT / 'data' / 'ipl_matches_all.parquet'
    df = pd.read_parquet(path)
    
    # Add phase if not present
    if 'phase' not in df.columns:
        df['phase'] = df['over'].apply(get_phase)
    
    print(f"  Loaded {len(df):,} IPL balls")
    return df


def load_global_data():
    """Load global T20 data."""
    print("Loading global T20 data...")
    path = PROJECT_ROOT / 'data' / 'global_t20_matches.parquet'
    df = pd.read_parquet(path)
    
    # Add phase column
    df['phase'] = df['over'].apply(get_phase)
    
    print(f"  Loaded {len(df):,} global T20 balls")
    return df


def load_war_projections():
    """Load WAR projections from all models, prioritizing auction-matched file."""
    print("Loading WAR projections...")
    proj_dir = PROJECT_ROOT / 'results' / 'WARprojections'
    
    projections = {}
    
    # PRIORITY 1: Load from pre-matched auction file (has best coverage)
    auction_path = proj_dir / 'auction_2026_v9prod' / 'auction_war_projections_v9prod.csv'
    if auction_path.exists():
        try:
            df = pd.read_csv(auction_path)
            for _, row in df.iterrows():
                name = row['player']
                war = row.get('projected_war_2026', 0)
                source = row.get('prediction_source', 'Unknown')
                if war and war > 0:
                    projections[name] = {'war': war, 'source': source}
            print(f"  Loaded {len(projections)} from auction V9 prod file")
        except Exception as e:
            print(f"  Error loading auction file: {e}")
    
    # PRIORITY 2: V9 Production (for any missing)
    for role in ['batter', 'bowler']:
        try:
            path = proj_dir / 'v9_production' / f'{role}_projections_2026_v9prod.csv'
            df = pd.read_csv(path)
            name_col = 'batter_name' if role == 'batter' else 'bowler_name'
            for _, row in df.iterrows():
                name = row[name_col]
                war = row['projected_war_2026']
                if name not in projections:
                    projections[name] = {'war': war, 'source': 'V9_Production'}
        except FileNotFoundError:
            pass
    
    # PRIORITY 3: Marcel (fallback)
    for role in ['batter', 'bowler']:
        try:
            path = proj_dir / 'marcel' / f'{role}_projections_2026.csv'
            df = pd.read_csv(path)
            name_col = 'player_name' if 'player_name' in df.columns else ('batter_name' if role == 'batter' else 'bowler_name')
            for _, row in df.iterrows():
                name = row[name_col]
                war = row['projected_war_2026']
                if name not in projections:
                    projections[name] = {'war': war, 'source': 'Marcel'}
        except FileNotFoundError:
            pass
    
    # PRIORITY 4: Global Only (final fallback)
    for role in ['batter', 'bowler']:
        try:
            path = proj_dir / 'global_only' / f'{role}_global_only_predictions.csv'
            df = pd.read_csv(path)
            name_col = 'batter_name' if role == 'batter' else 'bowler_name'
            if name_col not in df.columns:
                name_col = 'player_name'
            for _, row in df.iterrows():
                name = row[name_col]
                war = row.get('projected_war_2026', row.get('predicted_ipl_war', 0))
                if name not in projections:
                    projections[name] = {'war': war, 'source': 'Global_Only'}
        except FileNotFoundError:
            pass
    
    print(f"  Total projections loaded: {len(projections)} players")
    return projections


# ============================================================================
# EXPECTED RUNS MODEL (For RAA Calculation)
# ============================================================================

def train_expected_runs_model(match_data):
    """Train expected runs model for RAA calculation."""
    print("Training expected runs model...")
    
    # Calculate league averages by phase for baseline
    phase_stats = {}
    
    for phase in ['powerplay', 'middle', 'death']:
        phase_df = match_data[match_data['phase'] == phase]
        
        # Batting average (runs per ball)
        total_runs = phase_df['batter_runs'].sum()
        total_balls = len(phase_df[phase_df['wides'].fillna(0) == 0])  # Legal deliveries
        
        avg_runs_per_ball = total_runs / total_balls if total_balls > 0 else 1.2  # Default fallback
        
        phase_stats[phase] = {
            'avg_runs_per_ball': avg_runs_per_ball,
            'avg_sr': avg_runs_per_ball * 100
        }
    
    print(f"  Phase averages: PP={phase_stats['powerplay']['avg_sr']:.1f}, Mid={phase_stats['middle']['avg_sr']:.1f}, Death={phase_stats['death']['avg_sr']:.1f}")
    return phase_stats


# ============================================================================
# PHASE STATISTICS CALCULATION
# ============================================================================

def calculate_batter_phase_stats(player_id, player_name, match_data, phase_avgs, label_suffix):
    """Calculate batting statistics by phase for a player."""
    stats = {}
    
    # Try matching by ID first, then by name
    player_data = pd.DataFrame()
    
    if player_id and len(str(player_id)) > 2:
        player_data = match_data[match_data['batter_id'] == player_id]
    
    if len(player_data) == 0:
        # Try exact name match
        name_norm = normalize_name(player_name)
        player_data = match_data[match_data['batter_name'].apply(normalize_name) == name_norm]
    
    if len(player_data) == 0:
        # Try fuzzy name match
        all_batters = match_data['batter_name'].unique()
        matched_name, score = fuzzy_match(player_name, all_batters, threshold=0.80)
        if matched_name:
            player_data = match_data[match_data['batter_name'] == matched_name]
    
    for phase in ['powerplay', 'middle', 'death']:
        phase_short = {'powerplay': 'pp', 'middle': 'mid', 'death': 'death'}[phase]
        phase_data = player_data[player_data['phase'] == phase]
        
        # Remove wides from ball count
        legal_balls = len(phase_data[phase_data['wides'].fillna(0) == 0])
        runs = phase_data['batter_runs'].sum()
        
        # Strike rate
        sr = (runs / legal_balls * 100) if legal_balls > 0 else None
        
        # RAA = Actual runs - Expected runs (based on league average)
        if legal_balls > 0:
            expected_runs = legal_balls * phase_avgs[phase]['avg_runs_per_ball']
            raa = runs - expected_runs
        else:
            raa = None
        
        # Additional useful stats
        boundaries = phase_data[phase_data['batter_runs'] >= 4]['batter_runs'].count()
        sixes = phase_data[phase_data['batter_runs'] == 6]['batter_runs'].count()
        fours = phase_data[phase_data['batter_runs'] == 4]['batter_runs'].count()
        dots = len(phase_data[(phase_data['batter_runs'] == 0) & (phase_data['wides'].fillna(0) == 0)])
        dot_pct = (dots / legal_balls * 100) if legal_balls > 0 else None
        boundary_pct = (boundaries / legal_balls * 100) if legal_balls > 0 else None
        
        stats[f'bat_runs_{phase_short}_{label_suffix}'] = runs if runs > 0 else None
        stats[f'bat_balls_{phase_short}_{label_suffix}'] = legal_balls if legal_balls > 0 else None
        stats[f'bat_sr_{phase_short}_{label_suffix}'] = round(sr, 1) if sr else None
        stats[f'bat_raa_{phase_short}_{label_suffix}'] = round(raa, 1) if raa else None
        stats[f'bat_dot_pct_{phase_short}_{label_suffix}'] = round(dot_pct, 1) if dot_pct else None
        stats[f'bat_boundary_pct_{phase_short}_{label_suffix}'] = round(boundary_pct, 1) if boundary_pct else None
    
    return stats


def calculate_bowler_phase_stats(player_id, player_name, match_data, phase_avgs, label_suffix):
    """Calculate bowling statistics by phase for a player."""
    stats = {}
    
    # Try matching by ID first, then by name
    player_data = pd.DataFrame()
    
    if player_id and len(str(player_id)) > 2:
        player_data = match_data[match_data['bowler_id'] == player_id]
    
    if len(player_data) == 0:
        # Try exact name match
        name_norm = normalize_name(player_name)
        player_data = match_data[match_data['bowler_name'].apply(normalize_name) == name_norm]
    
    if len(player_data) == 0:
        # Try fuzzy name match
        all_bowlers = match_data['bowler_name'].unique()
        matched_name, score = fuzzy_match(player_name, all_bowlers, threshold=0.80)
        if matched_name:
            player_data = match_data[match_data['bowler_name'] == matched_name]
    
    for phase in ['powerplay', 'middle', 'death']:
        phase_short = {'powerplay': 'pp', 'middle': 'mid', 'death': 'death'}[phase]
        phase_data = player_data[player_data['phase'] == phase]
        
        # Legal deliveries (exclude wides and no-balls for ball count)
        legal_balls = len(phase_data[(phase_data['wides'].fillna(0) == 0) & (phase_data['noballs'].fillna(0) == 0)])
        
        # Runs conceded (total_runs - byes - legbyes for fair assessment)
        if 'total_runs' in phase_data.columns:
            runs_conceded = phase_data['total_runs'].sum()
            if 'byes' in phase_data.columns:
                runs_conceded -= phase_data['byes'].fillna(0).sum()
            if 'legbyes' in phase_data.columns:
                runs_conceded -= phase_data['legbyes'].fillna(0).sum()
        else:
            runs_conceded = phase_data['batter_runs'].sum() + phase_data['wides'].fillna(0).sum() + phase_data['noballs'].fillna(0).sum()
        
        # Wickets
        wickets = phase_data['is_wicket'].sum() if 'is_wicket' in phase_data.columns else 0
        
        # Economy rate (runs per over)
        overs = legal_balls / 6
        econ = (runs_conceded / overs) if overs > 0 else None
        
        # Bowling strike rate (balls per wicket)
        bowl_sr = (legal_balls / wickets) if wickets > 0 else None
        
        # RAA for bowling (negative is good - runs saved)
        if legal_balls > 0:
            expected_conceded = legal_balls * phase_avgs[phase]['avg_runs_per_ball']
            raa = expected_conceded - runs_conceded  # Positive = saved runs
        else:
            raa = None
        
        # Dot ball percentage
        dots = len(phase_data[(phase_data['batter_runs'] == 0) & 
                              (phase_data['wides'].fillna(0) == 0) & 
                              (phase_data['noballs'].fillna(0) == 0)])
        dot_pct = (dots / legal_balls * 100) if legal_balls > 0 else None
        
        stats[f'bowl_balls_{phase_short}_{label_suffix}'] = legal_balls if legal_balls > 0 else None
        stats[f'bowl_runs_{phase_short}_{label_suffix}'] = int(runs_conceded) if runs_conceded > 0 else None
        stats[f'bowl_wkts_{phase_short}_{label_suffix}'] = int(wickets) if wickets > 0 else None
        stats[f'bowl_econ_{phase_short}_{label_suffix}'] = round(econ, 2) if econ else None
        stats[f'bowl_sr_{phase_short}_{label_suffix}'] = round(bowl_sr, 1) if bowl_sr else None
        stats[f'bowl_raa_{phase_short}_{label_suffix}'] = round(raa, 1) if raa else None
        stats[f'bowl_dot_pct_{phase_short}_{label_suffix}'] = round(dot_pct, 1) if dot_pct else None
    
    return stats


# ============================================================================
# PRICE PREDICTION MODELS
# ============================================================================

def train_price_models():
    """Train VOMAM and VOPE price models."""
    print("Training price models...")
    
    # Load historical price data
    price_path = PROJECT_ROOT / 'results' / '11_war_vs_price' / 'war_vs_price_full.csv'
    if not price_path.exists():
        print("  Warning: Price training data not found, using defaults")
        return None, None, []
    
    price_data = pd.read_csv(price_path)
    
    # Load metadata for roles
    meta_path = PROJECT_ROOT / 'data' / 'player_metadata.csv'
    if meta_path.exists():
        metadata = pd.read_csv(meta_path)
        price_data = price_data.merge(
            metadata[['player_name', 'country', 'role_category']], 
            left_on='cricwar_name', right_on='player_name', how='left'
        )
        price_data['is_overseas'] = price_data['country'].apply(
            lambda x: 0 if str(x).lower() in ['india', 'ind'] else 1
        )
        price_data['role_category'] = price_data['role_category'].fillna('Unknown')
    else:
        price_data['is_overseas'] = 0
        price_data['role_category'] = 'Unknown'
    
    # VOMAM (Linear Model)
    df_encoded = pd.get_dummies(price_data, columns=['role_category'], prefix='role', drop_first=False)
    role_columns = [c for c in df_encoded.columns if c.startswith('role_')]
    
    features = ['total_WAR', 'is_overseas'] + role_columns
    target = 'price_cr'
    
    X = df_encoded[features].fillna(0)
    y = df_encoded[target].fillna(0)
    
    vomam_model = LinearRegression()
    vomam_model.fit(X, y)
    print(f"  VOMAM R²: {vomam_model.score(X, y):.3f}")
    
    # VOPE (Power Law)
    vope_data = price_data[(price_data['total_WAR'] > 0) & (price_data['price_cr'] > 0)].copy()
    
    def power_law(x, a, b):
        return a * np.power(x, b)
    
    try:
        popt, _ = curve_fit(power_law, vope_data['total_WAR'], vope_data['price_cr'], p0=[1, 1], maxfev=5000)
        vope_params = popt
        print(f"  VOPE params: a={popt[0]:.3f}, b={popt[1]:.3f}")
    except Exception as e:
        print(f"  VOPE fitting failed: {e}")
        vope_params = None
    
    return vomam_model, vope_params, role_columns


def load_xgb_prices():
    """Load XGBoost price predictions if available."""
    print("Loading XGBoost price predictions...")
    xgb_path = PROJECT_ROOT / 'results' / 'analysis' / 'price_model' / 'auction_prices_2026_xgb.csv'
    
    if not xgb_path.exists():
        print("  Warning: XGBoost predictions not found")
        return {}
    
    try:
        df = pd.read_csv(xgb_path)
        xgb_prices = {}
        for _, row in df.iterrows():
            player = row['player']
            price = row.get('predicted_price_cr', 0)
            if price and price > 0:
                xgb_prices[player] = price
        print(f"  Loaded XGBoost prices for {len(xgb_prices)} players")
        return xgb_prices
    except Exception as e:
        print(f"  Error loading XGBoost predictions: {e}")
        return {}


def predict_prices(war, is_overseas, role, vomam_model, vope_params, role_columns):
    """Predict prices using VOMAM and VOPE models."""
    # VOMAM
    if vomam_model is not None:
        input_data = pd.DataFrame({'total_WAR': [war], 'is_overseas': [int(is_overseas)]})
        for col in role_columns:
            role_name = col.replace('role_', '')
            input_data[col] = 1 if role == role_name else 0
        
        features = ['total_WAR', 'is_overseas'] + role_columns
        vomam_price = max(0.2, vomam_model.predict(input_data[features])[0])
    else:
        vomam_price = max(0.2, war * 3)  # Simple fallback
    
    # VOPE
    if vope_params is not None and war > 0:
        vope_price = max(0.2, vope_params[0] * (war ** vope_params[1]))
    else:
        vope_price = max(0.2, war * 2.5)  # Simple fallback
    
    return round(vomam_price, 2), round(vope_price, 2)


def standardize_role(role):
    """Map playing role to category for price model."""
    if pd.isna(role):
        return 'Unknown'
    
    role_lower = str(role).lower()
    
    if 'wicketkeeper' in role_lower or 'wk' in role_lower:
        return 'Wicketkeeper'
    elif 'allrounder' in role_lower or 'all-rounder' in role_lower:
        if 'batting' in role_lower:
            return 'Batting allrounder'
        elif 'bowling' in role_lower:
            return 'Bowling allrounder'
        return 'Allrounder'
    elif 'batter' in role_lower or 'batsman' in role_lower:
        if 'opening' in role_lower or 'top' in role_lower:
            return 'Top-order Batter'
        elif 'middle' in role_lower:
            return 'Middle-order Batter'
        return 'Batter'
    elif 'bowler' in role_lower:
        if 'spin' in role_lower or 'leg' in role_lower or 'off' in role_lower:
            return 'Spin Bowler'
        elif 'fast' in role_lower or 'seam' in role_lower or 'medium' in role_lower:
            return 'Pace Bowler'
        return 'Bowler'
    
    return 'Unknown'


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def calculate_age(dob, reference_date='2026-01-01'):
    """Calculate age from DOB."""
    if pd.isna(dob):
        return None
    try:
        birth = pd.to_datetime(dob)
        ref = pd.to_datetime(reference_date)
        age = (ref - birth).days / 365.25
        return round(age, 1)
    except:
        return None


def main():
    print("=" * 70)
    print("COMPREHENSIVE 2026 IPL AUCTION ANALYTICS")
    print("=" * 70)
    
    # Load all data
    auction_pool = load_auction_pool()
    ipl_data = load_ipl_data()
    global_data = load_global_data()
    war_projections = load_war_projections()
    
    # Train models
    phase_avgs_ipl = train_expected_runs_model(ipl_data)
    phase_avgs_global = train_expected_runs_model(global_data)
    vomam_model, vope_params, role_columns = train_price_models()
    xgb_prices = load_xgb_prices()
    
    # Combine IPL + Global data for "all cricket" stats
    # Need to align columns first
    common_cols = ['batter_name', 'batter_id', 'bowler_name', 'bowler_id', 
                   'batter_runs', 'total_runs', 'is_wicket', 'wides', 'noballs', 'over', 'phase']
    
    ipl_subset = ipl_data[[c for c in common_cols if c in ipl_data.columns]].copy()
    global_subset = global_data[[c for c in common_cols if c in global_data.columns]].copy()
    
    # Add missing columns with NaN
    for col in common_cols:
        if col not in ipl_subset.columns:
            ipl_subset[col] = np.nan
        if col not in global_subset.columns:
            global_subset[col] = np.nan
    
    all_cricket_data = pd.concat([ipl_subset, global_subset], ignore_index=True)
    print(f"Combined all cricket data: {len(all_cricket_data):,} balls")
    
    # Process each player
    print("\nProcessing players...")
    results = []
    
    for idx, row in auction_pool.iterrows():
        player = row['player']
        cricsheet_id = row.get('cricsheet_id', '')
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(auction_pool)} players...")
        
        result = {
            'sr_no': row.get('sr_no', idx + 1),
            'player': player,
            'country': row.get('country_display', row.get('country', '')),
            'role': row.get('role', ''),
            'base_price_lakh': row.get('base_price_lakh', ''),
            'capped': row.get('capped', ''),
            'age': calculate_age(row.get('dob', None)),
        }
        
        # WAR Projection
        war = 0
        source = 'None'
        
        # Try exact name match first
        if player in war_projections:
            war = war_projections[player]['war']
            source = war_projections[player]['source']
        else:
            # Try fuzzy match
            matched, score = fuzzy_match(player, war_projections.keys(), threshold=0.85)
            if matched:
                war = war_projections[matched]['war']
                source = war_projections[matched]['source']
        
        result['war_2026'] = round(war, 3) if war else 0
        result['war_source'] = source
        
        # Price Predictions
        is_overseas = 0 if str(row.get('country_display', row.get('country', ''))).lower() in ['india', 'ind'] else 1
        role_category = standardize_role(row.get('role', ''))
        
        vomam_price, vope_price = predict_prices(war, is_overseas, role_category, vomam_model, vope_params, role_columns)
        result['vomam_price_cr'] = vomam_price
        result['vope_price_cr'] = vope_price
        
        # XGBoost price (improved predictions)
        xgb_price = None
        if player in xgb_prices:
            xgb_price = round(xgb_prices[player], 2)
        else:
            # Try fuzzy match for XGB prices
            matched, score = fuzzy_match(player, xgb_prices.keys(), threshold=0.85)
            if matched:
                xgb_price = round(xgb_prices[matched], 2)
        
        result['xgb_price_cr'] = xgb_price
        
        # Average price - use XGB if available, otherwise VOMAM/VOPE average
        if xgb_price:
            result['avg_price_cr'] = xgb_price  # Use XGB as primary prediction
        else:
            result['avg_price_cr'] = round((vomam_price + vope_price) / 2, 2)
        
        # Batter stats - IPL
        batter_stats_ipl = calculate_batter_phase_stats(
            cricsheet_id, player, ipl_data, phase_avgs_ipl, 'ipl'
        )
        result.update(batter_stats_ipl)
        
        # Batter stats - All Cricket
        batter_stats_all = calculate_batter_phase_stats(
            cricsheet_id, player, all_cricket_data, phase_avgs_global, 'all'
        )
        result.update(batter_stats_all)
        
        # Bowler stats - IPL
        bowler_stats_ipl = calculate_bowler_phase_stats(
            cricsheet_id, player, ipl_data, phase_avgs_ipl, 'ipl'
        )
        result.update(bowler_stats_ipl)
        
        # Bowler stats - All Cricket
        bowler_stats_all = calculate_bowler_phase_stats(
            cricsheet_id, player, all_cricket_data, phase_avgs_global, 'all'
        )
        result.update(bowler_stats_all)
        
        results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by WAR projection (descending)
    df = df.sort_values('war_2026', ascending=False)
    
    # Save outputs
    output_dir = PROJECT_ROOT / 'results' / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Full table
    output_path = output_dir / 'auction_2026_comprehensive.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved comprehensive table: {output_path}")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total players: {len(df)}")
    print(f"Players with WAR projections: {len(df[df['war_2026'] > 0])}")
    print(f"Players with IPL batting data: {len(df[df['bat_balls_pp_ipl'].notna()])}")
    print(f"Players with IPL bowling data: {len(df[df['bowl_balls_pp_ipl'].notna()])}")
    
    print("\nWAR Source Distribution:")
    print(df['war_source'].value_counts())
    
    print("\nTop 10 by Projected WAR:")
    print(df[['player', 'role', 'war_2026', 'war_source', 'avg_price_cr']].head(10).to_string(index=False))
    
    print("\nTop 10 by Predicted Price:")
    top_price = df.nlargest(10, 'avg_price_cr')
    print(top_price[['player', 'role', 'war_2026', 'avg_price_cr']].to_string(index=False))


if __name__ == "__main__":
    main()
