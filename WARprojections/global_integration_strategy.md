# Global T20 Data Integration Strategy

## Objective
To enhance IPL projections by incorporating player performance data from other major T20 leagues (BBL, PSL, CPL, etc.) and International T20s (T20Is). This allows us to:
1.  Capture "Recent Form" leading up to the IPL.
2.  Evaluate "Foreign Player Quality" more accurately.
3.  Assess "Domestic Form" for Indian uncapped/capped players (via Syed Mushtaq Ali Trophy).

## Data Sources
We have access to the following datasets in `data/other_t20_data/`:
*   **International**: `t20s_json` (T20Is)
*   **Major Leagues**: `bbl_json` (Big Bash), `psl_json` (PSL), `cpl_json` (CPL), `sat_json` (SA20), `ilt_json` (ILT20), `mlc_json` (MLC).
*   **Domestic**: `sma_json` (Syed Mushtaq Ali Trophy - India), `ntb_json` (T20 Blast - England).

## Strategy

### 1. Player Filtering (The "IPL Universe")
As requested, we will **only predict for players who have already played in the IPL**.
*   **Why**: To avoid predicting for thousands of irrelevant players.
*   **How**:
    1.  Load the master list of unique `player_id`s from our full IPL history (`data/ipl_matches_all.parquet`).
    2.  When processing global data, filter for matches involving these IDs.
    3.  *Note*: We will need a robust ID mapping. Cricsheet IDs are consistent across leagues, so matching by `registry` ID should work out of the box.

### 2. League Strength Adjustment (MLE)
A run in the IPL is not equal to a run in the SMAT. We need to normalize performance.

**Approach: Relative League Difficulty Factors**
We will calculate a "Difficulty Factor" for each league relative to the IPL.
*   **Method**: Compare the RAA/ball of players who played in *both* the IPL and League X in the same calendar year.
*   **Formula**: $Factor_L = \frac{Average(RAA_{IPL})}{Average(RAA_{L})}$
*   **Application**: $Adjusted\_RAA = Raw\_RAA \times Factor_L$
    *   If IPL is harder, RAA in League X will be discounted (Factor < 1).

### 3. Feature Engineering
We will create new features to feed into the XGBoost model.

#### A. Recent Global Form
*   **Concept**: How has the player performed in the last 12 months globally?
*   **Features**:
    *   `global_raa_per_ball_1yr`: Volume-weighted average of Adjusted RAA in all T20s in the year prior to the IPL season.
    *   `global_consistency_1yr`: Stability of performance across leagues.

#### B. International Experience
*   **Concept**: Is the player an established international star?
*   **Features**:
    *   `is_capped`: Boolean (Has played T20Is).
    *   `t20i_balls_faced_career`: Total experience at the highest level.
    *   `t20i_raa_career`: Career performance in Internationals.

#### C. Domestic Dominance (for Indian Players)
*   **Concept**: How did they do in SMAT?
*   **Features**:
    *   `smat_raa_per_ball_1yr`: Performance in India's domestic T20.

### 4. Implementation Plan

#### Step 1: Global Data Extraction (`WARprojections/05_extract_global.py`)
*   Iterate through all folders in `data/other_t20_data`.
*   Filter for matches containing at least one "IPL Player".
*   Extract ball-by-ball data, tagging each row with `league_name`.

#### Step 2: League Strength Calculation (`WARprojections/06_league_strength.py`)
*   Calculate raw RAA for all leagues (using the same context adjustment logic as IPL, or simplified).
*   Compute League Factors based on overlapping players.
*   Save `league_factors.csv`.

#### Step 3: Global Feature Generation (`WARprojections/07_global_features.py`)
*   Calculate `Adjusted RAA` using League Factors.
*   Aggregate into `Recent Form` and `Career` features.
*   Merge with existing IPL features.

#### Step 4: Model Retraining (`WARprojections/08_train_global_model.py`)
*   Train XGBoost with the new expanded feature set.
*   Expectation: Higher accuracy for overseas players and domestic Indian players.

## Timeline
1.  **Draft Strategy** (Done)
2.  **Extract Data** (Next)
3.  **Calculate Factors**
4.  **Feature Engineering**
5.  **Train & Validate**
