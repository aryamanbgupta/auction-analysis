# Fantasy Points Projections (FantasyProjections)

## Overview

Predicts **Dream11 fantasy points per match** for all 250 IPL 2026 squad players using an ensemble ML model (XGBoost + RandomForest + Marcel) trained on full IPL history (2008-2025).

**Production model**: R²=0.313 (XGBoost standalone), backtest on 2025 actuals (n=134).

## Dream11 Scoring System

| Category | Rule | Points |
|----------|------|--------|
| **Batting** | Per run | +1 |
| | Boundary bonus (4) | +1 |
| | Six bonus (6) | +2 |
| | Milestones (30/50/100) | +4/+8/+16 |
| | Duck (BAT/WK/AR only) | -2 |
| | SR bonus/penalty (min 10 balls) | ±2/±4/±6 |
| **Bowling** | Per wicket | +25 |
| | Bowled/LBW bonus | +8 |
| | Maiden over | +12 |
| | Haul bonus (3/4/5 wkts) | +4/+8/+16 |
| | Economy bonus/penalty (min 2 overs) | ±2/±4/±6 |
| **Fielding** | Catch | +8 |
| | Stumping | +12 |
| | Direct run out | +12 |
| | 3+ catches bonus | +4 |
| **Other** | Playing XI appearance | +4 |

## Pipeline

### Core Pipeline (IPL-only)

```bash
# 1. Extract IPL ball-by-ball with fielder data
uv run python FantasyProjections/01_extract_with_fielders.py

# 2. Calculate Dream11 fantasy points
uv run python FantasyProjections/02_calculate_fantasy_points.py

# 3. Engineer features (~95 per player-season)
uv run python FantasyProjections/03_feature_engineering.py

# 4. Backtest: compare model architectures
uv run python FantasyProjections/04_train_model.py

# 5. Train production model on all data
uv run python FantasyProjections/05_train_production.py

# 6. Score auction pool (pre-auction, 350 players)
uv run python FantasyProjections/06_score_auction.py

# 7. Score 2026 squads (post-auction, 250 players, 10 teams)
uv run python FantasyProjections/07_score_squads.py

# 8. Generate interactive visualization
uv run python FantasyProjections/08_visualize_squads.py
```

### T20I Integration Pipeline (experimental)

Tested whether adding international T20I performance between IPL seasons improves predictions. **Result: no improvement** — kept as experimental code.

```bash
# Extract T20I ball-by-ball for IPL players (1,452 matches, 332K balls)
uv run python FantasyProjections/02b_extract_t20i_fantasy.py

# Calculate Dream11 fantasy points for T20I matches
uv run python FantasyProjections/02c_calculate_t20i_fantasy_points.py

# Engineer inter-season T20I features (16 features)
uv run python FantasyProjections/03b_t20i_features.py

# Re-run feature engineering (merges T20I if available)
uv run python FantasyProjections/03_feature_engineering.py

# Re-run backtest (includes A/B comparison with T20I)
uv run python FantasyProjections/04_train_model.py
```

## Model Architecture

### Feature Set (~95 IPL features)

| Category | Features | Count |
|----------|----------|-------|
| Core | avg_fantasy_pts, career_avg, matches, years_played | 7 |
| Lag | avg_fantasy_pts_lag{1,2,3}, total_fantasy_pts_lag1 | 6 |
| Phase batting | SR, boundary_rate, six_rate per phase (PP/mid/death) | 12 |
| Phase bowling | econ, dot_rate, wicket_rate per phase | 12 |
| Fantasy rates | boundary_rate, sr_bonus_rate, econ_bonus_rate, maiden_rate | 9 |
| Form (rolling) | fp_last_{5,10,15}, fp_decay_form, fp_volatility, fp_trend | 6 |
| Opponent quality | opp_adj_fp_avg | 1 |
| Role one-hot | role_BAT, role_BOWL, role_AR, role_WK | 4 |

**Target**: avg_fantasy_pts in next IPL season

### Ensemble

- **XGBoost** (primary): n_estimators=100, max_depth=4, lr=0.05, reg_alpha=1.0, reg_lambda=2.0
- **RandomForest**: n_estimators=100, max_depth=4
- **Marcel baseline**: fantasy_pts_weighted (5/4/3 Marcel weighting)
- **Ridge stacking**: learns ensemble weights from training predictions

## Backtest Results (2024 features -> 2025 actuals)

| Model | R² | RMSE | MAE | Spearman |
|-------|-----|------|-----|----------|
| **XGBoost (standalone)** | **0.313** | **12.50** | **10.17** | **0.487** |
| RandomForest | 0.299 | 12.62 | 10.26 | 0.523 |
| Marcel | 0.055 | 14.65 | 11.80 | 0.335 |
| Ensemble (Ridge) | 0.198 | 13.50 | 10.77 | 0.430 |
| Role-Based Combined | 0.145 | 13.93 | 10.98 | 0.420 |

**Winner**: Unified model (all roles together with role one-hot encoding)

## T20I Feature Experiment

### Hypothesis
International T20I matches between IPL seasons capture player form and could improve predictions, especially for capped international players.

### Data
- 4,976 T20I JSON files in `data/other_t20_data/t20s_json/`
- 1,452 matches with IPL players (437 of 766 IPL players, 57%)
- 332K balls extracted, 31K player-match records scored
- 16 inter-season features (form, batting/bowling stats, recency, opponent quality, momentum)

### T20I Features Tested

| Feature | Description |
|---------|-------------|
| t20i_avg_fantasy_pts | Mean FP per T20I match in inter-season window |
| t20i_decay_form | Exponential recency-weighted average |
| t20i_batting_sr | T20I strike rate |
| t20i_boundary_rate | (4s + 6s) / balls faced |
| t20i_bowling_econ | T20I economy rate |
| t20i_vs_ipl_ratio | T20I avg FP / previous IPL avg FP |
| t20i_form_momentum | T20I avg FP - previous IPL avg FP |
| t20i_recency_days | Days since last T20I before IPL |
| t20i_elite_share | Fraction vs elite opponents |
| has_t20i_data | Binary indicator |

### Results

| Model | R² (XGB) | R² (Ensemble) | MAE |
|-------|----------|---------------|-----|
| **Baseline (no T20I)** | **0.313** | **0.198** | 10.17 |
| + T20I features | 0.303 | 0.169 | 10.07 |

**Conclusion: T20I features do not improve predictions.**

- R² drops by 0.01 (XGB) and 0.03 (ensemble)
- MAE improves marginally (0.1 FP) — below the 1.0 significance threshold
- T20I features account for 31% of XGBoost feature importance (signal exists but adds noise)
- Even for capped players only (n=62): baseline R²=0.299 vs +T20I R²=0.112
- **Case study (Phil Salt)**: Baseline predicted 52.8 vs actual 54.4 (error: 1.6). T20I model predicted 38.4 (error: 16.0) — extreme T20I form didn't translate to IPL

**Why T20I didn't help:**
1. T20I conditions differ from IPL (pitches, boundaries, bowling quality)
2. Only 34.8% of training data has T20I features — sparse signal
3. Small test set (n=134) makes 16 extra features prone to overfitting
4. Explosive T20I performers (Salt, Finch) show high variance that doesn't linearly map to IPL

## Squad Projections (IPL 2026)

### Coverage

| Source | Players | % |
|--------|---------|---|
| Production model | 163 | 65.2% |
| Marcel fallback | 30 | 12.0% |
| Replacement level (25 FP) | 57 | 22.8% |
| **Total** | **250** | **100%** |

Replacement-level players are genuinely uncapped/new-to-IPL (Pathum Nissanka, Ben Duckett, Brydon Carse, etc.).

### Team Rankings (Avg Projected FP/Match)

| Rank | Team | Avg FP | Best Player |
|------|------|--------|-------------|
| 1 | DC | 32.1 | KL Rahul (54.6) |
| 2 | PBKS | 32.1 | Shreyas Iyer (56.1) |
| 3 | MI | 32.0 | Suryakumar Yadav (57.7) |
| 4 | SRH | 31.9 | Heinrich Klaasen (60.7) |
| 5 | RR | 31.8 | Yashasvi Jaiswal (56.7) |
| 6 | CSK | 30.8 | Ruturaj Gaikwad (52.8) |
| 7 | LSG | 30.8 | Mitchell Marsh (64.3) |
| 8 | RCB | 30.6 | Virat Kohli (47.7) |
| 9 | KKR | 29.0 | Cameron Green (53.8) |
| 10 | GT | 28.8 | Sai Sudharsan (65.0) |

### ID Matching Fixes

The enriched squad file uses different cricsheet IDs than ball-by-ball data for some players. Manual overrides applied:

| Player | Issue | Fix |
|--------|-------|-----|
| Rohit Sharma | Squad ID ≠ Cricsheet ID (RG Sharma) | Manual override → 740742ef |
| Rasikh Dar | Listed as "Rasikh Salam" in Cricsheet | Manual override → b8527c3d |

Fuzzy matching threshold set to 0.90 with last-name validation to prevent false positives.

## Outputs

| File | Description |
|------|-------------|
| `results/FantasyProjections/fantasy_projections_2026.csv` | Production projections (204 players) |
| `results/FantasyProjections/fantasy_projections_2026_lookup.csv` | Lookup table for scoring |
| `results/FantasyProjections/squad_2026/squad_fantasy_projections_2026.csv` | All 250 squad players scored |
| `results/FantasyProjections/squad_2026/squad_fantasy_visualization.html` | Interactive Plotly dashboard |
| `results/FantasyProjections/backtest_2025_unified.csv` | Backtest predictions vs actuals |
| `results/FantasyProjections/model_comparison.csv` | A/B test results |
| `data/t20i_matches_fantasy.parquet` | T20I ball-by-ball (experimental) |
| `data/t20i_fantasy_points_per_match.csv` | T20I fantasy points (experimental) |
| `data/t20i_fantasy_features.csv` | T20I inter-season features (experimental) |

## Script Reference

| Script | Purpose |
|--------|---------|
| **Core Pipeline** | |
| `01_extract_with_fielders.py` | Extract IPL ball-by-ball + fielder data |
| `02_calculate_fantasy_points.py` | Dream11 scoring (bat/bowl/field) |
| `03_feature_engineering.py` | ~95 ML features per player-season |
| `04_train_model.py` | Backtest comparison (unified vs role-based, ±T20I) |
| `05_train_production.py` | Production model trained on all data |
| `06_score_auction.py` | Score auction pool (pre-auction) |
| `07_score_squads.py` | Score 2026 squads with robust matching |
| `08_visualize_squads.py` | Interactive Plotly HTML dashboard |
| **T20I Experiment** | |
| `02b_extract_t20i_fantasy.py` | Extract T20I ball-by-ball for IPL players |
| `02c_calculate_t20i_fantasy_points.py` | Dream11 scoring for T20I matches |
| `03b_t20i_features.py` | Inter-season T20I feature engineering |
