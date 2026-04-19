# Custom Fantasy Auction — IPL 2026 (Boston League)

Mid-season analysis for a custom-scoring fantasy auction league ("Boston IPL 26"),
separate from the Dream11 projection pipeline documented in `README.md`.

## Goal

Given a custom fantasy scoring system (different from Dream11) and a draft snapshot
from a 7-manager auction league, compute each player's season-to-date points,
reconcile against the app's ground truth, and build auction-value analytics.

## Scoring System (as published by the app)

**Batting**

| Event | Points |
|---|---|
| Run scored | +1 |
| Four | +1 (bonus) |
| Six | +2 (bonus) |
| Duck | −2 |
| 30-run bonus | +4 |
| 50-run bonus | +4 |
| 100-run bonus | +8 |
| SR < 50 | −6 |
| SR 50–59.99 | −4 |
| SR 60–70 | −2 |
| SR 70–130 | 0 |
| SR 130–150 | +2 |
| SR 150.01–170 | +4 |
| SR > 170 | +6 |

Milestones are **cumulative** (a 100 earns +4+4+8 = +16). SR bands apply regardless
of balls faced (no min-ball qualification).

**Bowling**

| Event | Points |
|---|---|
| Dot ball | +1 |
| Maiden over | +12 |
| Wicket | +20 |
| Wide | −1 |
| No ball | −2 |
| 3-wicket bonus | +4 |
| 4-wicket bonus | +8 |
| 5+-wicket bonus | +16 |
| Econ < 5 | +6 |
| Econ 5–5.99 | +4 |
| Econ 6–7 | +2 |
| Econ 7–10 | 0 |
| Econ 10–11 | −2 |
| Econ 11.01–12 | −4 |
| Econ > 12 | −6 |

Wicket bonuses are cumulative. Econ bands apply regardless of overs bowled
(no min-overs qualification).

**Fielding**

| Event | Points |
|---|---|
| Catch | +8 |
| 3+ catches bonus | +4 |
| Run out | +6 |
| Stumping | +6 |

## Data Inputs

| File | Purpose |
|---|---|
| `data/ipl_json (4)/` | Cricsheet ball-by-ball JSONs for IPL 2026 (24 matches played as of Apr 16) |
| `data/Boston IPL 26_Results (1).xlsx` | Draft results per manager + unsold players + app points column |

Data cutoff varies by IPL team. Only three teams have ball-by-ball data that
matches the app's cutoff: **Lucknow, Mumbai, Punjab** (last matches Apr 15–16).
Other teams played additional games not yet in the cricsheet snapshot.

## Pipeline

```bash
# Compute per-player, per-match points under 4 rule variants
uv run python FantasyProjections/10_ipl2026_custom_fantasy.py

# Reconcile computed totals against Boston app points
uv run python FantasyProjections/11_reconcile_boston.py

# Build interactive HTML rankings (filter by team/role/owner)
uv run python FantasyProjections/12_player_rankings_html.py
```

### `10_ipl2026_custom_fantasy.py`

Extracts 2026 ball-by-ball, aggregates per-player-per-match batting/bowling/fielding
stats, and scores under 4 variants to isolate two rule ambiguities:

- `V_BASE` — cumulative milestones + 10-ball SR min + 2-over econ min
- `V_HIGHEST` — highest-only milestone bonus
- `V_NO_SR` — no SR min-balls qualification
- `V_NO_ECON` — no econ min-overs qualification

Outputs:
- `season_to_date_totals.csv` — 150 players × 4 variant totals + raw stats
- `per_match_points.csv` — 555 player-match rows (V_BASE breakdown)
- `variant_diffs.csv` — players whose total changes across variants

### `11_reconcile_boston.py`

Matches Boston app rows against cricsheet names (manual map of ~70 players for
abbreviations like `V Suryavanshi`, `P Simran Singh`, `SV Samson`; fuzzy fallback
for the rest). If a manual target isn't in the variants table, the player is
marked unmatched rather than being fuzzy-remapped — this prevents silent errors
like the **Harshit Rana → N Rana** bug where a player who hasn't played all
season was matched to a different Rana.

Outputs `boston_reconciliation.csv` with per-player app vs variant comparison.

### `12_player_rankings_html.py`

Produces `player_rankings.html` — interactive table of all 310 players
(drafted + unsold + cricsheet-only depth) sorted by points, with filters for
IPL team, IPL role, and Boston owner. Uses app points where available;
computed `V_NO_SR` totals otherwise.

## Key Findings

### Rules confirmed via reconciliation (18 players from complete-data teams)

Under **strictly-published rules** (no SR min, no econ min, cumulative milestones):

- **MAE = 4.4 pts, 8/18 exact match, 12/18 within ±2 pts**
- **All pure batters/WKs match exactly** — Rohit 188, Pant 165, Marsh 154, QdK 164,
  SKY 151, Tilak 95, Pooran 45, Priyansh 178, Wadhera 43, Iyer ±1, Rohit ±1.

### Unexplained residuals

All non-zero residuals are **negative** (I overcount vs app):

| Player | App | Mine | Δ | Notes |
|---|---|---|---|---|
| Prabhsimran Singh (WK) | 303 | 325 | −22 | Bat-side overcounting |
| Arshdeep Singh (BOWL) | 126 | 145 | −19 | 5 wkts, 17 wides |
| Trent Boult (BOWL) | 1 | 11 | −10 | Full XI all 3 matches |
| Yuzvendra Chahal (BOWL) | 72 | 81 | −9 | |
| Marco Jansen (AR) | 109 | 116 | −7 | |
| Hardik Pandya (AR) | 163 | 169 | −6 | |
| Jasprit Bumrah (BOWL) | 44 | 46 | −2 | |
| Aiden Markram (BAT) | 186 | 188 | −2 | |

**Bowlers consistently overcount.** Candidate hidden rules tested:
- Wicket = 16 (not 20): Arshdeep −19→+1, Pandya −6→+2, Chahal −9→+3 — strongest
  single adjustment, but **contradicts the published rules as stated**.
- Wide = −2/ball (not −1): also improves fit, smaller effect.
- Per-over econ instead of per-match: **does not** outperform per-match overall.

**Prabhsimran's −22** is unexplained by any single tweak (tested: six=0, catch=6,
highest-only milestones, SR cap at +4 >150). No clean rule accounts for the
exact 22-pt gap.

### Data-coverage caveat

Reconciliation must be filtered to LSG/MI/PBKS. Running it globally produces
large spurious residuals for teams (CSK, DC, GT, KKR, RR, SRH, RCB) whose app
totals include matches not yet in `data/ipl_json (4)`.

## Full-Season Projection Pipeline (scripts 13a-13g)

Scripts 10-12 produce season-to-date totals. Scripts 13a-13g produce **full
group-stage projections** for every IPL 2026 player and a per-league HTML
dashboard.

```bash
uv run python FantasyProjections/13a_score_historical_custom.py   # Rescore 2007-25 under published rules
uv run python FantasyProjections/13b_marcel_projection.py         # Marcel baseline (5-4-3 weighted, 100-match prior)
uv run python FantasyProjections/13c_build_training_table.py      # Build per-season XGBoost training table
uv run python FantasyProjections/13d_train_xgboost.py             # Train with LOSO (2020-25) backtest
uv run python FantasyProjections/13e_project_2026.py              # Generate 2026 full-season projections
uv run python FantasyProjections/13f_consolidate_and_visualize.py # Boston dashboard
uv run python FantasyProjections/13g_deshdrohi_projections.py     # Deshdrohi Babes dashboard
```

### 13a — Historical rescoring

Re-scores `data/ipl_matches_fantasy.parquet` (278K balls, 2007-2025) under the
**strictly-published** custom rules (no SR min balls, no econ min overs,
cumulative milestones). Outputs `data/historical_custom_points_per_match.parquet`
(25,633 player-match rows) and `data/historical_custom_season_totals.csv` (3,065
player-seasons). Sanity check: V Kohli 1328 in 2016, SP Narine 1228 in 2024,
Shubman Gill 933 in 2025.

### 13b — Marcel baseline

For each IPL 2026 squad player, computes a rate-based Marcel projection:
`ppm_marcel = (5·ppm_2025·m_2025 + 4·ppm_2024·m_2024 + 3·ppm_2023·m_2023 + 100·prior) / (5·m_2025 + 4·m_2024 + 3·m_2023 + 100)`,
blended with 2026 YTD (20-match prior), times remaining group-stage games.
Rookies fall through to role prior × 14 × 0.5. Output: 265 players.

### 13c — Training table

Builds `data/training_table_custom.parquet` (7,544 rows). For each target
season S ∈ {2012…2025}:

- **Group-stage filter**: a match is group stage if neither team has already
  played 14 games before it (strips 3-4 playoff matches per season).
- **Early cutoff**: for each season S, the "early window" covers matches
  where both teams going in have played fewer than **6 games**. This matches
  the user's training design — "include all matches where every team has
  played less than six matches".
- **Features**: career totals, last-3-season lag (ppm + matches), Marcel
  blend, role prior, early-season rates (ppm, bat/bowl/field splits, runs,
  wickets, overs).
- **Target**: full group-stage total points in season S.

### 13d — XGBoost training + LOSO backtest

`xgboost.XGBRegressor(max_depth=5, lr=0.05, n_estimators=800,
early_stopping=30)`, LOSO over 2020-2025 with S-1 held out for early
stopping.

**Feature importance (gain):**
- `early_matches` — 53%
- `early_ppm` — 13%
- `early_runs` — 5%
- `early_overs` — 3%
- `role_BOWL` — 2.6%
- `early_bat_ppm` — 2.4%
- `early_wickets` — 2.3%
- `lag1_matches` / `lag1_ppm` — 1.5% each

Early-season signals dominate because they carry both rate info *and*
playing-time info (a benched player shows 0 early matches).

### 13e — 2026 projection

Extracts 2026 balls, scores under published rules, tags early-2026 matches
(every team has played 4-5 so far), builds the same feature set, and runs
`xgb.predict`. Blends: `final_projection = ytd_total + max(0, xgb_full_season − ytd_total)`.
Rookies (no history, no 2026 appearance) fall back to `role_prior_ppm × 14 × 0.5`.

### 13f — Boston dashboard

Produces `projections_2026.csv` (259 players) and `projections_2026.html` —
ranked table with search, owner/team/role filters, sortable columns, and
XGB/Marcel/Ensemble projection columns side by side.

### 13g — Deshdrohi Babes dashboard

Re-owners the XGB projections using Deshdrohi Babes' auction xlsx and emits
`projections_deshdrohi.html` with 4 tabs:

- **Rankings** — 271 filterable players (60 drafted + 200 unsold + 11 extras)
- **Team Mapping** — per-manager roster cards with prices, YTD, XGB projection
- **Unsold Players** — 200 listed, sorted by projection (top targets: Patidar
  719, Raghuvanshi 570, Rickelton 520)
- **Manager Standings** — ranked by total projected points

**Projected Deshdrohi standings:**
Delhi Daredevils 5107 · Sarda Smashers 4271 · Bijoy's Blasters 3913 ·
Nivi's Angels 3666 · Shaibe de Paape 2871 · Super Kings 2464.

**Projected Boston standings (XGB totals):**
Daredevils 4733 · Mavericks 4347 · Boston Super Kings 3883 · Guppy Challengers 3831 ·
Bandi Blasters 3394 · KarKaRe 3179 · RCBBC 2864.

## Model comparison — XGBoost wins decisively

LOSO backtest (2020-2025, 1,138 player-seasons predicted) on `target_total`
(full group-stage points):

| Model | MAE | RMSE | R² | Bias |
|---|---|---|---|---|
| **XGBoost** | **93.4** | **117.8** | **+0.736** | +0.8 |
| Ensemble (mean of XGB + Marcel) | 159.4 | 184.1 | +0.354 | +106.5 |
| Marcel (ppm × 14) | 260.4 | 303.0 | −0.748 | +212.3 |

Per-season MAE is tight — XGB stays within 90-98 across all six backtest
years, Marcel within 234-275.

**Why Marcel fails so hard:** Marcel predicts `ppm × 14`, assuming every player
plays all 14 group-stage games. In reality, benched/rotational players play
0-6. Marcel has no mechanism to down-weight them, so it systematically
overpredicts (+212 bias). XGB learns games-played directly from
`early_matches` (53% importance) — if a player has 0 early matches, the model
predicts a small total even if their career ppm is high.

**Why the ensemble is worse than XGB alone:** Averaging a good model (bias ≈ 0)
with a badly-biased one (bias +212) drags the combined prediction up by ~106
pts per player. The ensemble inherits Marcel's systematic overprediction.

**Bottom line — XGBoost is the model to use.** The dashboards sort on XGB's
`final_projection`; `marcel_projection` and `ensemble_projection` are shown
for reference only.

## Outputs

| File | Description |
|---|---|
| `results/FantasyProjections/ipl2026_custom/season_to_date_totals.csv` | Per-player YTD totals, all 4 variants |
| `results/FantasyProjections/ipl2026_custom/per_match_points.csv` | Per-player-per-match YTD breakdown |
| `results/FantasyProjections/ipl2026_custom/variant_diffs.csv` | Players whose YTD differs across variants |
| `results/FantasyProjections/ipl2026_custom/boston_reconciliation.csv` | Boston player ↔ cricsheet comparison |
| `results/FantasyProjections/ipl2026_custom/published_vs_app.csv` | Published-rules vs app residuals (LSG/MI/PBKS) |
| `results/FantasyProjections/ipl2026_custom/player_rankings.html` | YTD-only rankings page (script 12) |
| `data/historical_custom_points_per_match.parquet` | 2007-2025 per-match under published rules |
| `data/historical_custom_season_totals.csv` | 2007-2025 per-player-season totals |
| `data/training_table_custom.parquet` | XGBoost training table (7,544 rows, 2012-2025) |
| `results/FantasyProjections/ipl2026_custom/marcel_projections_2026.csv` | Marcel baseline (265 players) |
| `results/FantasyProjections/ipl2026_custom/xgb_loso_backtest.csv` | Per-player per-season backtest predictions |
| `results/FantasyProjections/ipl2026_custom/xgb_feature_importance.csv` | XGBoost gain per feature |
| `results/FantasyProjections/ipl2026_custom/xgb_model.json` | Trained XGBoost model (final, 2012-2025) |
| `results/FantasyProjections/ipl2026_custom/xgb_projections_2026.csv` | Raw XGB projections (Boston owners) |
| `results/FantasyProjections/ipl2026_custom/projections_2026.csv` | XGB + Marcel + ensemble, Boston owners |
| `results/FantasyProjections/ipl2026_custom/projections_2026.html` | Boston IPL 26 dashboard |
| `results/FantasyProjections/ipl2026_custom/projections_deshdrohi.csv` | XGB + Marcel + ensemble, Deshdrohi owners |
| `results/FantasyProjections/ipl2026_custom/projections_deshdrohi.html` | Deshdrohi Babes dashboard (4 tabs) |
