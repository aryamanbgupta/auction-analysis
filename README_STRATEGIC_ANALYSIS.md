# Delhi Capitals 2026 Auction Strategy Analysis

**Objective**: To formulate a data-driven auction strategy for Delhi Capitals (DC) for the 2026 IPL season, using the cricWAR framework to diagnose 2025 failures and identify high-impact targets.

## 📊 Key Outputs

### 1. Interactive Strategy Blog
The primary output is a comprehensive, interactive HTML blog post that combines narrative analysis with dynamic Plotly graphs.
*   **File**: [`results/analysis/strategic/dc_strategy_blog.html`](results/analysis/strategic/dc_strategy_blog.html)
*   **Features**:
    *   **Diagnosis**: Interactive charts showing DC's performance vs League Average.
    *   **Solution**: Scatter plots of potential targets (Auction vs Retained).
    *   **Comparison**: Hover over points to see player stats (RAA, SR, Economy).

### 2. Strategic Report
A detailed markdown report summarizing the findings, methodology, and proposed lineup.
*   **File**: [`scripts/analysis/strategic_report.md`](scripts/analysis/strategic_report.md)

### 3. Target Lists
CSV files containing the top identified targets for specific roles:
*   `results/analysis/strategic/target_openers.csv` (Overseas Openers)
*   `results/analysis/strategic/target_openers_all.csv` (All Openers)
*   `results/analysis/strategic/target_middle_order.csv` (Middle Order Batters)
*   `results/analysis/strategic/target_death_bowlers.csv` (Death Bowlers)
*   `results/analysis/strategic/target_pp_pacers.csv` (Powerplay Pacers)

---

## 🛠️ Methodology

The analysis follows a 3-phase approach:

### Phase 1: Diagnosis (Why did we fail in 2025?)
We analyzed DC's 2025 performance using ball-by-ball data to identify critical weaknesses:
*   **Middle Over Slump**: Compared DC's Run Rate in overs 7-15 against the league average.
*   **Opening Instability**: Analyzed the number of unique opening pairs vs average Powerplay score.
*   **Death Bowling Crisis**: Ranked teams by Economy Rate in overs 16-20.
*   **Powerplay Bowling Void**: Ranked teams by total wickets in overs 0-6.

### Phase 2: Solution (Who should we buy?)
We filtered the auction pool to find players who solve these specific problems.
*   **Metrics**: Used **RAA** (Runs Above Average) and **Strike Rate/Economy** in specific phases (Powerplay, Middle, Death).
*   **Comparison**: Plots include both **Auction Targets (Blue)** and **Retained Players (Red)** to provide context and benchmarks.
*   **Exclusions**: Automatically excluded retained players (e.g., Shami, Boult, Chameera) from the "Target" lists while keeping them on graphs for comparison.

### Phase 3: Vision (The New Lineup)
Proposed a balanced playing XI combining retained core (KL Rahul, Axar, Kuldeep, Stubbs) with top auction targets.

---

## 📂 Project Structure

### Scripts
*   **`scripts/analysis/strategic_analysis.py`**: The core engine. Performs the diagnosis, filters targets, and generates all Plotly graphs.
*   **`scripts/analysis/generate_auction_pool.py`**: Creates the `auction_pool_*.csv` files by filtering out retained players from the main dataset.
*   **`scripts/analysis/generate_blog.py`**: Converts the markdown report into a styled HTML blog with embedded iframes.
*   **`scripts/analysis/hard_exclude.py`**: Utility to manually remove retained players who leaked into the pool (e.g., KK Ahmed, Arshad Khan).

### Data
*   **Input**: `results/06_context_adjustments/ipl_with_raa.parquet` (Full ball-by-ball data with RAA).
*   **Pool**: `results/analysis/auction_pool/` (Batters and Bowlers CSVs).
*   **Retentions**: `data/ipl_2026_retentions.csv` (List of retained players).

---

## 🚀 Usage

To re-run the complete analysis:

```bash
# 1. Generate/Clean the Auction Pool
uv run python scripts/analysis/generate_auction_pool.py
uv run python scripts/analysis/hard_exclude.py

# 2. Run Strategic Analysis (Generates Graphs & CSVs)
uv run python scripts/analysis/strategic_analysis.py

# 3. Generate the HTML Blog
uv run python scripts/analysis/generate_blog.py
```

## 📝 Notes
*   **Retained Players**: Players like Mohammed Shami, Trent Boult, and Dushmantha Chameera are marked as "Retained" based on the provided retention list and are shown in Red on the graphs.
*   **Data Source**: Analysis uses projected/simulated data for 2025 and historical data from 2022-2024.
