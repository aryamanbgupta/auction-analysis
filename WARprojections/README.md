# Full History ML Projections (WARprojections)

## Overview
This module implements a production-grade machine learning system for projecting player performance (WAR) in the IPL. Unlike previous iterations that used limited data, this system leverages the **full IPL history (2008-2025)** to learn robust patterns of player aging, consistency, and career trajectories.

## Methodology

### 1. Data Extraction (`01_extract_full_history.py`)
*   **Source**: Cricsheet ball-by-ball JSON data.
*   **Scope**: All IPL matches from 2008 to 2025 (approx. 1,169 matches, 278,000+ balls).
*   **Output**: `data/ipl_matches_all.parquet`

### 2. Core Metrics (`02_calculate_metrics.py`)
We re-calculated all advanced metrics from scratch for the entire history to ensure consistency across eras.

*   **Expected Runs (XR)**: Trained a global XGBoost model on the full dataset to estimate the expected runs from any given match state (wickets, overs, innings).
*   **Run Values**: Calculated the run value of every ball as `Runs Scored + Change in XR`.
*   **Leverage Index (LI)**: Weighted important moments (high leverage) more heavily.
*   **Context Adjustments (RAA)**: Used linear regression to isolate player skill from:
    *   Venue effects (e.g., Chinnaswamy vs. Chepauk)
    *   Innings effect (1st vs. 2nd innings)
    *   Era effects (Season dummies)
*   **Wins Above Replacement (WAR)**:
    *   Aggregated RAA by player-season.
    *   Calculated a dynamic "Replacement Level" for each season.
    *   Converted VORP (Value Over Replacement Player) to Wins using a standard runs-per-win constant (~13.5).
*   **Consistency Score**: Calculated the standard deviation of a player's match-level RAA within a season.

### 3. Feature Engineering (`03_feature_engineering.py`)
We generated a rich feature set for every player-season:

*   **Lagged Performance**: WAR, RAA, and Consistency from the previous 1, 2, and 3 seasons.
*   **Weighted Averages**: Marcel-style weighted average of past performance (5/4/3 weighting).
*   **Career Trajectory**:
    *   `career_war`: Cumulative WAR entering the season.
    *   `years_played`: Number of IPL seasons played.
    *   `career_balls`: Total experience volume.
*   **Demographics**:
    *   `age`: Player age at the start of the season (derived from metadata).

### 4. Model Architecture (`04_train_model.py`)
*   **Algorithm**: XGBoost Regressor.
*   **Objective**: Minimize Squared Error (RMSE).
*   **Target**: `WAR_next_season` (Predicting Year N+1 using data up to Year N).
*   **Validation Strategy**: Time-Series Cross-Validation (Walk-Forward).
    *   **Train**: 2008-2023
    *   **Backtest**: Predict 2025 (using 2024 data) and compare with actuals.
    *   **Forecast**: Predict 2026 (using 2025 data).

## Performance Results (2025 Backtest)

The Full History ML model significantly outperforms the baseline Marcel projection system.

| Metric | Batting R² | Batting RMSE | Bowling R² | Bowling RMSE |
| :--- | :--- | :--- | :--- | :--- |
| **Full History ML** | **0.36** | **2.76** | **0.19** | **3.46** |
| Marcel Baseline | 0.28 | 0.21* | 0.17 | 0.31* |
| Previous ML (2-yr) | -1.00 | 0.36* | -0.47 | 0.41* |

*Note: RMSE values are on different scales due to WAR unit differences between the legacy pipeline and this new full-history pipeline. R² is the reliable comparative metric.*

## Usage
To reproduce the projections:

```bash
# 1. Extract Data
uv run python WARprojections/01_extract_full_history.py

# 2. Calculate Metrics
uv run python WARprojections/02_calculate_metrics.py

# 3. Generate Features
uv run python WARprojections/03_feature_engineering.py

# 4. Train Model & Forecast
uv run python WARprojections/04_train_model.py
```

## Outputs
*   **Projections**: `results/WARprojections/batter_projections_2026.csv` & `bowler_projections_2026.csv`
*   **Backtest Results**: `results/WARprojections/batter_backtest_2025.csv` & `bowler_backtest_2025.csv`

## Phase 5: Global Data Integration

To further enhance projections, we integrated data from major T20 leagues (BBL, PSL, CPL, etc.) and T20Is.

### Methodology
1.  **Extraction**: Extracted 1.28M+ balls from global T20s involving IPL players (`05_extract_global.py`).
    - Now includes `team1`/`team2` columns for T20I stratification.
2.  **League Strength** (`06_league_strength.py`): 
    - **T20I Stratification**: Split T20Is into Elite/Mixed/Associate tiers based on opponent strength.
    - **Separate Batter/Bowler Factors**: Each role gets its own difficulty factors.
    - **Regression with Intercept**: `IPL_RAA = α + β × LeagueX_RAA`.
    - **Weighted Regression**: Using WLS weighted by balls faced.
    - **Conservative Prior**: Unknown/sparse leagues shrink toward 0.3 (not 1.0).

### League Difficulty Factors

| League | Batter Factor | Bowler Factor | Notes |
|--------|---------------|---------------|-------|
| **IPL** | 1.00 | 1.00 | Reference |
| CPL | 0.34 | 0.46 | |
| BBL | 0.32 | 0.38 | |
| BPL | 0.32 | 0.39 | |
| T20Blast | 0.27 | 0.45 | |
| T20I_Elite | 0.19 | 0.30 | Both teams from Top 10 |
| T20I_Mixed | 0.15 | 0.16 | One strong team |
| SMAT | 0.14 | 0.18 | India domestic |

3.  **Features** (`07_global_features.py`): Uses role-specific factors when computing `global_raa_per_ball` and `global_balls`.
4.  **Modeling** (`08_train_global_model.py`): 
    - **Reduced feature set**: 6-7 features to prevent overfitting.
    - **Bowling-specific config**: Higher `global_balls` threshold (100 vs 30), more regularization.

## Phase 6: Model Improvements (Dec 2024)

### V2: Overfitting Fix
The original XGBoost models were severely overfitting (Train R² 0.73, Test R² 0.02). We fixed this by:

1. **Reduced Features**: From 18 to 5-7 most predictive, least correlated features
2. **Increased Regularization**: `reg_alpha=1.0`, `reg_lambda=2.0`, `max_depth=3`

---

## Phase 7: Advanced Model Development (Dec 2024)

We conducted extensive experimentation to improve model accuracy beyond the baseline fixes.

### Model Evolution

| Version | Batters R² | Bowlers R² | Key Changes |
|---------|------------|------------|-------------|
| **V2** | 0.08 | 0.23 | Overfitting fix, reduced features |
| **V3** | 0.16 | 0.27 | Phase RAA + Regression-to-mean target |
| **V4** | 0.17 | 0.29 | Situational features (chasing/setting) |
| **V5** | 0.24 | 0.32 | XGB + Marcel ensemble |
| **V6** | **0.25** | **0.35** | Optimized + Multi-model (XGB+RF+Marcel) |

### V3 Improvements (`04_train_model_v3.py`)

**New Features** (`03b_phase_features.py`):
- `phase_raa_per_ball_powerplay` - Performance in overs 0-5
- `phase_raa_per_ball_middle` - Performance in overs 6-15
- `phase_raa_per_ball_death` - Performance in overs 16-20
- `last_5_matches_raa` - Recent form from last 5 IPL matches

**Key Innovation**: Predict deviation from regression instead of raw WAR:
```python
expected_war = α + β × WAR_weighted  # Simple regression line
target = actual_war - expected_war    # Predict the residual
```

### V4 Improvements (`04_train_model_v4.py`)

**New Features** (`03c_situational_features.py`):
- `sit_raa_per_ball_chasing` - Performance when chasing
- `sit_raa_per_ball_setting` - Performance when setting
- `bat_position` - Average batting position (opener=1-2, middle=3-5, finisher=6+)
- `win_rate` - Team strength proxy (team's win percentage)

### V5 Improvements (`04_train_model_v5.py`)

**Ensemble Approach**: Combine XGBoost with Marcel using learned weights
- Uses Ridge regression to find optimal weighting
- **Learned Weights**:
  - Batters: 80% XGB + 20% Marcel
  - Bowlers: 49% XGB + 51% Marcel

### V6 Improvements (`04_train_model_v6.py`) - BEST MODEL

**Role-Specific Optimization**:
- **Batters**: `max_depth=5` (deeper trees help)
- **Bowlers**: `max_depth=2`, `n_estimators=50` (shallower, more regularized)

**Multi-Model Ensemble**: XGBoost + RandomForest + Marcel
- **Learned Weights (Batters)**: XGB 1.18, RF -0.43, Marcel 0.25
- **Learned Weights (Bowlers)**: XGB -0.69, RF 1.16, Marcel 0.53

**Key Insight**: For bowlers, RandomForest + Marcel outperforms XGBoost significantly.

---

## Usage

```bash
# Full Pipeline (recommended)
uv run python WARprojections/01_extract_full_history.py   # Extract IPL data
uv run python WARprojections/02_calculate_metrics.py       # Calculate WAR
uv run python WARprojections/03_feature_engineering.py     # Base features
uv run python WARprojections/05_extract_global.py          # Global T20 data
uv run python WARprojections/06_league_strength.py         # League factors
uv run python WARprojections/07_global_features.py         # Global features
uv run python WARprojections/03b_phase_features.py         # Phase + form features
uv run python WARprojections/03c_situational_features.py   # Situational features
uv run python WARprojections/09_marcel_baseline.py         # Marcel baseline
uv run python WARprojections/04_train_model_v6_production.py  # V6 Production model
uv run python WARprojections/11_combine_projections.py     # Combine all sources
uv run python WARprojections/12_global_to_ipl_model.py     # Global-only predictions
uv run python WARprojections/13_score_auction_pool.py      # Score auction players
```

---

## Phase 8: Production & Auction Scoring (Dec 2024)

### V6 Production Model (`04_train_model_v6_production.py`)

The V6 Production model trains on **ALL available data** (2008-2024), including the 2024→2025 transition:

| Model | Training Data | Use Case |
|-------|---------------|----------|
| V6 (backtest) | 2008-2023 | Validated on 2025 actuals |
| **V6 Production** | 2008-2024 | Best for 2026 forecasting |

### Global-to-IPL Translation Model (`12_global_to_ipl_model.py`)

Predicts IPL performance for players with **NO IPL history** using global T20 data:

| Metric | Batters | Bowlers |
|--------|---------|---------|
| CV R² | 0.21 | 0.13 |
| Players Scored | 3,567 | 2,647 |

**Key Features**: T20I balls (49%), franchise balls (14%), global RAA per ball (15%)

### Combined Projections (`11_combine_projections.py`)

Merges predictions with priority: V6 > Marcel > Global-only

### Auction Pool Scoring (`13_score_auction_pool.py`)

Scores all players in `data/ipl_2026_auction_enriched.csv`:

| Source | Coverage |
|--------|----------|
| V6_Production | 52 players |
| Marcel | 63 players |
| Global-Only | 119 players |
| No Data (0 WAR) | 116 players |

---

## Outputs

### Best Model (V6 Production)
- `results/WARprojections/v6_production/batter_projections_2026_prod.csv`
- `results/WARprojections/v6_production/bowler_projections_2026_prod.csv`

### Combined (All Sources)
- `results/WARprojections/combined_2026/batter_projections_2026_combined.csv`
- `results/WARprojections/combined_2026/bowler_projections_2026_combined.csv`

### Global-Only (No IPL History)
- `results/WARprojections/global_only/batter_global_only_predictions.csv`
- `results/WARprojections/global_only/bowler_global_only_predictions.csv`

### Auction Pool
- `results/WARprojections/auction_2026/auction_pool_war_projections.csv`

### All Models Comparison

| Model | Folder | Batters R² | Bowlers R² |
|-------|--------|------------|------------|
| Marcel | `marcel/` | 0.16 | 0.23 |
| V2 (IPL ML) | root | 0.08 | 0.23 |
| V3 | `v3/` | 0.16 | 0.27 |
| V4 | `v4/` | 0.17 | 0.29 |
| V5 | `v5/` | 0.24 | 0.32 |
| **V6** | **`v6/`** | **0.25** | **0.35** |
| **V6 Prod** | **`v6_production/`** | N/A | N/A |
| Global-Only | `global_only/` | 0.21 (CV) | 0.13 (CV) |

---

## Script Reference

| Script | Purpose |
|--------|---------|
| `01_extract_full_history.py` | Extract IPL ball-by-ball data |
| `02_calculate_metrics.py` | Calculate WAR, RAA, LI |
| `03_feature_engineering.py` | Base ML features |
| `03b_phase_features.py` | Phase-specific RAA (powerplay/middle/death) |
| `03c_situational_features.py` | Chasing/setting, batting position |
| `04_train_model.py` | V2 IPL-only model |
| `04_train_model_v3.py` - `v6.py` | Improved model versions |
| `04_train_model_v6_production.py` | **Production model (best)** |
| `05_extract_global.py` | Extract global T20 data |
| `06_league_strength.py` | Calculate league difficulty factors |
| `07_global_features.py` | Global T20 features |
| `08_train_global_model.py` | Global-enhanced model |
| `09_marcel_baseline.py` | Marcel projection system |
| `10_comprehensive_analysis.py` | Model comparison |
| `11_combine_projections.py` | Merge all prediction sources |
| `12_global_to_ipl_model.py` | Predict IPL WAR from global-only data |
| `13_score_auction_pool.py` | Score auction players |

---

## Recommendations

- **Use V6 Production for 2026 forecasting**: Trained on most recent data
- **Use Combined projections for full coverage**: Includes all players
- **Use Global-Only for new players**: Players without IPL history
- **Batting projections**: V6 (R² 0.25) is ~210% better than V2 baseline
- **Bowling projections**: V6 (R² 0.35) is ~52% better than V2 baseline

---

## Phase 9: V9 Enhanced Model (Dec 2024) - CURRENT BEST

The V9 model introduces **opponent quality** and **rolling form** features, achieving the best performance yet.

### New Features (`19_feature_engineering_v9.py`)

| Feature | Description | Importance |
|---------|-------------|------------|
| `opponent_adj_raa_per_ball` | RAA weighted by opponent team strength | **#2 for both roles** |
| `form_trend` | Recent 5 vs older 5 matches (improving/declining) | #5-6 |
| `form_volatility` | Standard deviation of recent match performances | #7-8 |
| `last_10_form` | Average RAA/ball in last 10 IPL matches | #7 |
| `decay_weighted_form` | Exponentially weighted recent form | #9 |
| `last_5_form`, `last_15_form` | Additional rolling windows | Lower |

### Model Performance - 2025 Backtest

| Model | Batters R² | Bowlers R² | Improvement |
|-------|------------|------------|-------------|
| V6 (Previous Best) | 0.246 | **0.351** | Baseline |
| V7 (Unified) | 0.245 | 0.321 | +Global integration |
| **V9 (Enhanced)** | **0.270** | 0.350 | **+9.7% batters** |

**Key Finding**: V9 improves batter predictions by 9.7% while matching V6 for bowlers.

### V9 Production Model (`22_train_v9_production.py`)

Trained on ALL data through 2025 for 2026 forecasting:

```bash
uv run python WARprojections/22_train_v9_production.py
```

**Outputs**:
- `results/WARprojections/v9_production/batter_projections_2026_v9prod.csv`
- `results/WARprojections/v9_production/bowler_projections_2026_v9prod.csv`

---

## Phase 10: Improved Auction Scoring (Dec 2024)

### Coverage Improvement

| Version | Coverage | Key Change |
|---------|----------|------------|
| V1 (Legacy) | 66.9% (234/350) | Name matching only |
| V2 | 66.9% (234/350) | Added cricsheet ID matching |
| V3 | 72.6% (254/350) | Combined V7+V8 models |
| **V4/V9 Prod** | **96.9% (339/350)** | **Aggressive matching + role-aware** |

### Key Improvements

1. **ID-First Matching** (`14_score_auction_v2.py`):
   - 190 players matched by cricsheet ID
   - Falls back to name → fuzzy matching

2. **V8 Domestic Model** (`16_domestic_model_v8.py`):
   - Predicts IPL potential for players WITHOUT prior IPL history
   - Uses SMAT/domestic T20 data (156K balls)
   - CV R²: Batters 0.247, Bowlers 0.170
   - **Adds 173 players** previously at replacement level

3. **Role-Aware Matching** (`23_score_auction_v9prod.py`):
   - Bowlers check bowler lookups first
   - Batters check batter lookups first
   - Prevents cross-role name collisions (e.g., Pathirana fix)

### Final Auction File

**Location**: `results/WARprojections/auction_2026_v9prod/auction_war_projections_v9prod.csv`

**Columns**:
- `player` - Player name
- `country` - Nationality
- `role` - Playing role (Batter/Bowler/Allrounder/WK)
- `base_price` - Base auction price (₹ Lakh)
- `capped` - C (Capped), U (Uncapped), A (Associate)
- `projected_war_2026` - V9 Production WAR projection
- `prediction_source` - Model used (V9_Production/Marcel/V8_Domestic/Global_Only)
- `match_method` - How player was matched (ID/Name/Fuzzy)

### Prediction Source Priority

1. **V9_Production** - Best model for players with 2025 IPL data
2. **Marcel** - Simple weighted average for lower-volume players  
3. **V8_Domestic** - For uncapped players with SMAT/franchise data
4. **Global_Only** - For players with only international T20 data
5. **Replacement_Level** - 0 WAR (only 11 players)

---

## Visualization (`24_visualize_auction.py`)

Creates an interactive Plotly chart:

```bash
uv run python WARprojections/24_visualize_auction.py
open results/WARprojections/auction_2026_v9prod/auction_war_visualization.html
```

**Features**:
- X-axis: Base Price (₹ Lakh)
- Y-axis: Projected WAR
- Color coding by prediction source
- Hover for player details
- Annotations for top 5 players

---

## Quick Start - 2026 Auction Analysis

For new analysts, run this minimal pipeline:

```bash
cd /Users/aryamangupta/CricML/Auction_analysis

# Generate V9 features (if not exists)
uv run python WARprojections/19_feature_engineering_v9.py

# Train V9 production model
uv run python WARprojections/22_train_v9_production.py

# Score auction players
uv run python WARprojections/23_score_auction_v9prod.py

# Generate visualization
uv run python WARprojections/24_visualize_auction.py
```

**Output**: `results/WARprojections/auction_2026_v9prod/auction_war_projections_v9prod.csv`

---

## Complete Script Reference

| Script | Purpose |
|--------|---------|
| **Data Extraction** | |
| `01_extract_full_history.py` | Extract IPL ball-by-ball data |
| `05_extract_global.py` | Extract global T20 data |
| **Metrics & Features** | |
| `02_calculate_metrics.py` | Calculate WAR, RAA, LI |
| `03_feature_engineering.py` | Base ML features |
| `03b_phase_features.py` | Phase-specific RAA |
| `03c_situational_features.py` | Chasing/setting features |
| `06_league_strength.py` | League difficulty factors |
| `07_global_features.py` | Global T20 features |
| `09_marcel_baseline.py` | Marcel projection system |
| **`19_feature_engineering_v9.py`** | **V9: Opponent quality + form features** |
| **Models** | |
| `04_train_model.py` - `v6.py` | Legacy model versions |
| `04_train_model_v6_production.py` | V6 Production model |
| `08_train_global_model.py` | Global-enhanced model |
| `15_unified_model_v7.py` | V7 Unified (IPL+Global+Marcel) |
| `16_domestic_model_v8.py` | V8 Domestic (no IPL history) |
| **`20_train_model_v9.py`** | **V9 backtest model** |
| **`22_train_v9_production.py`** | **V9 Production (BEST)** |
| **Auction Scoring** | |
| `11_combine_projections.py` | Merge prediction sources (legacy) |
| `12_global_to_ipl_model.py` | Predict from global-only data |
| `13_score_auction_pool.py` | Legacy auction scoring |
| `14_score_auction_v2.py` | ID-first matching |
| `17_compare_models.py` | Model comparison report |
| `18_score_auction_v3.py` | Combined V7+V8 scoring |
| **`23_score_auction_v9prod.py`** | **V9 Production scoring (BEST)** |
| **`24_visualize_auction.py`** | **Interactive WAR visualization** |

---

## Final Model Comparison

| Model | Folder | Batters R² | Bowlers R² | Use Case |
|-------|--------|------------|------------|----------|
| Marcel | `marcel/` | 0.08 | 0.21 | Baseline |
| V6 | `v6_production/` | 0.246 | 0.351 | Previous best |
| V7 Unified | `v7_unified/` | 0.245 | 0.321 | IPL+Global |
| V8 Domestic | `v8_domestic/` | 0.247 (CV) | 0.170 (CV) | No IPL history |
| **V9 Production** | **`v9_production/`** | **0.270** | **0.350** | **CURRENT BEST** |

---

## Recommendations

- **Use V9 Production for 2026 forecasting**: Best accuracy, full 2025 data
- **Use `23_score_auction_v9prod.py` for auction analysis**: 96.9% coverage
- **Batting projections**: V9 (R² 0.27) is +9.7% vs V6, +237% vs V2
- **Bowling projections**: V9 (R² 0.35) matches V6 (theoretical ceiling)
- **Opponent quality matters**: #2 most important feature for both roles

## Theoretical Ceiling

For reference, even baseball WAR projections (with far more data and a more stable game) typically achieve R² of 0.3-0.4. Our V9 model at R² 0.35 for bowlers is approaching this theoretical limit for cricket.
