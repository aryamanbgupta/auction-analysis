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
1.  **Extraction**: Extracted 1.2M+ balls from global T20s involving IPL players (`05_extract_global.py`).
2.  **League Strength**: Calculated "Difficulty Factors" (MLE) for each league relative to the IPL (`06_league_strength.py`).
    *   *Example Factors*: T20I (0.16), BBL (0.27), SMAT (0.01).
3.  **Features**: Generated `global_raa_per_ball` (Adjusted RAA) and `global_balls` (Volume) for the year prior to each IPL season (`07_global_features.py`).
4.  **Modeling**: Retrained XGBoost with these additional features (`08_train_global_model.py`).

### Results
The global model maintains high accuracy for batters (R² 0.36) but introduces some noise for bowlers (R² 0.16 vs 0.19). This suggests that for established IPL bowlers, IPL history remains the gold standard, while global form is a secondary factor. However, this model is likely more robust for players with gaps in their IPL career.

### Outputs
*   **Global Projections**: `results/WARprojections/batter_projections_2026_global.csv` & `bowler_projections_2026_global.csv`

## Phase 6: Comprehensive Model Comparison

We conducted a rigorous comparison of three approaches:
1.  **Marcel Baseline**: Weighted average of past 3 years + regression to mean + aging curve.
2.  **IPL-Only ML**: XGBoost trained on full IPL history (2008-2024).
3.  **Global ML**: XGBoost trained on IPL history + Global T20 form.

### Key Findings (2025 Backtest)
*   **Batting**: Machine Learning approaches (**R² 0.36**) significantly outperform the Marcel baseline (**R² 0.28**). The Global model performs on par with the IPL-only model.
*   **Bowling**: The **IPL-Only ML model** is the most accurate (**R² 0.19**), followed by Marcel (**R² 0.17**). Global data appears to introduce noise for established IPL bowlers.

### File Reference
All results are stored in `results/WARprojections/`.

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
