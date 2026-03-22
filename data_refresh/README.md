# Data Refresh — Cricsheet Downloader

Automated tool to download and update T20 cricket match data from [Cricsheet](https://cricsheet.org/downloads/) for the WAR projection pipeline.

## Quick Start

```bash
# Download all competitions to staging
uv run python data_refresh/refresh_cricsheet.py --full

# Verify the staged data
uv run python data_refresh/refresh_cricsheet.py --verify

# Deploy verified data to data/
uv run python data_refresh/refresh_cricsheet.py --deploy
```

## How It Works

The refresh process has three phases to protect existing data:

1. **Download → staging**: Data is downloaded into `data_refresh/staging/`, never directly into `data/`.
2. **Verify**: JSON integrity is validated and file counts are compared against `data/`.
3. **Deploy**: Only after verification passes, new files are copied to `data/`. Existing files are never overwritten or deleted.

## Commands

| Command | Description |
|---------|-------------|
| `--full` | Download complete competition zips to staging |
| `--incremental` | Download only matches added in the last 30 days |
| `--verify` | Validate staged data (JSON integrity + comparison) |
| `--deploy` | Copy verified staging to `data/` |
| `--status` | Show side-by-side comparison of staging vs `data/` |

### Options

| Option | Description |
|--------|-------------|
| `--competitions ipl bbl ...` | Limit to specific competitions (default: all) |
| `--dry-run` | Preview what would be downloaded without downloading |

## Competitions

12 men's T20 competitions are supported, matching the leagues used by `WARprojections/05_extract_global.py`:

| Code | Competition | Destination |
|------|-------------|-------------|
| `ipl` | Indian Premier League | `data/ipl_json/` |
| `bbl` | Big Bash League | `data/other_t20_data/bbl_json/` |
| `psl` | Pakistan Super League | `data/other_t20_data/psl_json/` |
| `cpl` | Caribbean Premier League | `data/other_t20_data/cpl_json/` |
| `ntb` | T20 Blast | `data/other_t20_data/ntb_json/` |
| `bpl` | Bangladesh Premier League | `data/other_t20_data/bpl_json/` |
| `sma` | Syed Mushtaq Ali Trophy | `data/other_t20_data/sma_json/` |
| `t20s` | T20 Internationals | `data/other_t20_data/t20s_json/` |
| `sat` | SA20 | `data/other_t20_data/sat_json/` |
| `ilt` | International League T20 | `data/other_t20_data/ilt_json/` |
| `mlc` | Major League Cricket | `data/other_t20_data/mlc_json/` |
| `msl` | Mzansi Super League | `data/other_t20_data/msl_json/` |

## Common Workflows

### Full refresh before retraining

When you want to ensure all data is up to date before retraining models:

```bash
uv run python data_refresh/refresh_cricsheet.py --full
# Review the verification table, then:
uv run python data_refresh/refresh_cricsheet.py --deploy
```

### Quick update during a tournament

When a tournament is ongoing and you want the latest matches:

```bash
uv run python data_refresh/refresh_cricsheet.py --incremental
uv run python data_refresh/refresh_cricsheet.py --deploy
```

### Update a single competition

```bash
uv run python data_refresh/refresh_cricsheet.py --full --competitions ipl
uv run python data_refresh/refresh_cricsheet.py --deploy --competitions ipl
```

### Check what's pending

```bash
uv run python data_refresh/refresh_cricsheet.py --status
```

## Notes

- **Dependencies**: Uses only Python stdlib (no extra packages needed).
- **Idempotent**: Safe to re-run — files already in staging or `data/` are skipped.
- **T20I count**: The `data/` folder may contain more T20I files than staging because the original manual download included women's matches. These are harmless (filtered out by the pipeline) and are not deleted by deploy.
- **Data source**: All data comes from [Cricsheet](https://cricsheet.org/) (JSON format, men's matches only).
