#!/bin/bash
set -e

echo "Running Analysis Pipeline (Skipping Uncertainty)..."

# We already ran up to 09 in the previous attempt (or 10 was running).
# Let's ensure 09 completed successfully or just rerun it to be safe/quick.
# 09 is fast.

echo "Running 09_vorp_war.py..."
uv run python scripts/09_vorp_war.py

echo "Running 2025data.py..."
uv run python scripts/2025data.py

echo "Running 11_war_vs_price.py..."
uv run python scripts/11_war_vs_price.py

echo "Running 12_financial_valuation.py..."
uv run python scripts/12_financial_valuation.py

echo "Pipeline Complete!"
