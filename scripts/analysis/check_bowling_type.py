import pandas as pd
from pathlib import Path

def main():
    root = Path('.')
    try:
        ipl_df = pd.read_parquet(root / 'results' / '06_context_adjustments' / 'ipl_with_raa.parquet')
        
        for name in ['GJ Maxwell', 'M Theekshana', 'JP Behrendorff']:
            subset = ipl_df[ipl_df['bowler_name'] == name]
            if not subset.empty:
                btype = subset['bowling_type'].mode()[0]
                print(f"{name}: {btype}")
            else:
                print(f"{name}: Not found")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
