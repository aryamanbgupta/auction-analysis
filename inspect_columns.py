import pandas as pd
from pathlib import Path

try:
    df = pd.read_parquet('results/06_context_adjustments/ipl_with_raa.parquet')
    print("Columns:", df.columns.tolist())
except Exception as e:
    print(f"Error: {e}")
