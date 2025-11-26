#!/bin/bash
set -e

echo "Resuming Analysis Pipeline..."

echo "Running 10_uncertainty_estimation.py..."
uv run python scripts/10_uncertainty_estimation.py

echo "Running 11_war_vs_price.py..."
uv run python scripts/11_war_vs_price.py

echo "Running 12_financial_valuation.py..."
uv run python scripts/12_financial_valuation.py

echo "Pipeline Update Complete!"
