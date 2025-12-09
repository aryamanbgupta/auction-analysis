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

### Overfitting Fix
The original XGBoost models were severely overfitting (Train R² 0.73, Test R² 0.02). We fixed this by:

1. **Reduced Features**: From 18 to 5-7 most predictive, least correlated features:
   - `WAR_weighted` (Marcel-style weighted average)
   - `consistency` (stability measure)
   - `career_war` (cumulative experience)
   - `years_played` (tenure)
   - `balls_faced` / `balls_bowled` (volume)

2. **Increased Regularization**:
   - `reg_alpha=1.0` (L1)
   - `reg_lambda=2.0` (L2)
   - `max_depth=3` (shallower trees)

### Current Performance (2025 Backtest)

| Model | Batters R² | Bowlers R² |
|-------|------------|------------|
| **Marcel** | 0.16 | 0.23 |
| **IPL ML** | 0.08 | **0.23** |
| **Global ML** | 0.09 | 0.20 |

### Recommendations
- **Batters**: Use Marcel (simple and effective) or Global ML.
- **Bowlers**: Use IPL ML or Marcel (both at R² 0.23).

## Usage

```bash
# Full Pipeline (with global data)
uv run python WARprojections/01_extract_full_history.py   # Extract IPL data
uv run python WARprojections/02_calculate_metrics.py       # Calculate WAR
uv run python WARprojections/03_feature_engineering.py     # Generate features
uv run python WARprojections/05_extract_global.py          # Extract global T20 data
uv run python WARprojections/06_league_strength.py         # Calculate league factors
uv run python WARprojections/07_global_features.py         # Generate global features
uv run python WARprojections/04_train_model.py             # Train IPL-only model
uv run python WARprojections/08_train_global_model.py      # Train global model
uv run python WARprojections/09_marcel_baseline.py         # Generate Marcel projections
uv run python WARprojections/10_comprehensive_analysis.py  # Compare all models
```

## Outputs
*   **Projections**: `results/WARprojections/batter_projections_2026.csv` & `bowler_projections_2026.csv`
*   **Backtest Results**: `results/WARprojections/batter_backtest_2025.csv` & `bowler_backtest_2025.csv`
*   **Model Comparison**: `results/WARprojections/model_comparison_report.md`

## File Reference

| Model | Season | Role | File Path |
| :--- | :--- | :--- | :--- |
| **Marcel** | 2025 (Backtest) | Batter | `marcel/batter_projections_2025.csv` |
| **Marcel** | 2025 (Backtest) | Bowler | `marcel/bowler_projections_2025.csv` |
| **Marcel** | 2026 (Forecast) | Batter | `marcel/batter_projections_2026.csv` |
| **Marcel** | 2026 (Forecast) | Bowler | `marcel/bowler_projections_2026.csv` |
| **IPL ML** | 2025 (Backtest) | Batter | `batter_backtest_2025.csv` |
| **IPL ML** | 2025 (Backtest) | Bowler | `bowler_backtest_2025.csv` |
| **IPL ML** | 2026 (Forecast) | Batter | `batter_projections_2026.csv` |
| **IPL ML** | 2026 (Forecast) | Bowler | `bowler_projections_2026.csv` |
| **Global ML** | 2025 (Backtest) | Batter | `batter_backtest_2025_global.csv` |
| **Global ML** | 2025 (Backtest) | Bowler | `bowler_backtest_2025_global.csv` |
| **Global ML** | 2026 (Forecast) | Batter | `batter_projections_2026_global.csv` |
| **Global ML** | 2026 (Forecast) | Bowler | `bowler_projections_2026_global.csv` |

