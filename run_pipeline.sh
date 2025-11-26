#!/bin/bash
set -e

echo "Starting Analysis Pipeline Update..."

echo "Running 04_calculate_run_values.py..."
uv run python scripts/04_calculate_run_values.py

echo "Running 05_calculate_leverage_index.py..."
uv run python scripts/05_calculate_leverage_index.py

echo "Running 06_context_adjustments.py..."
uv run python scripts/06_context_adjustments.py

echo "Running 08_replacement_level.py..."
uv run python scripts/08_replacement_level.py

echo "Running 09_vorp_war.py..."
uv run python scripts/09_vorp_war.py

echo "Running 10_uncertainty_estimation.py..."
uv run python scripts/10_uncertainty_estimation.py

echo "Running 11_war_vs_price.py..."
uv run python scripts/11_war_vs_price.py

echo "Running 12_financial_valuation.py..."
uv run python scripts/12_financial_valuation.py

echo "Pipeline Update Complete!"
