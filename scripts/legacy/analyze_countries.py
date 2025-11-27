
import pandas as pd
from pathlib import Path

def main():
    project_root = Path('.').resolve()
    metadata_file = project_root / 'data' / 'player_metadata.csv'
    
    if not metadata_file.exists():
        print(f"Error: {metadata_file} not found.")
        return

    df = pd.read_csv(metadata_file)
    
    # Logic from scripts/12_financial_valuation.py
    # df['is_overseas'] = df['country'].apply(lambda x: 0 if str(x).lower() in ['india', 'ind'] else 1)
    
    def get_overseas_status(country):
        if pd.isna(country):
            return 'Unknown'
        if str(country).lower() in ['india', 'ind']:
            return 'Domestic (India)'
        return 'Overseas'

    df['status'] = df['country'].apply(get_overseas_status)
    
    print("Country Distribution:")
    print(df['country'].value_counts())
    
    print("\nOverseas vs Domestic Status:")
    print(df['status'].value_counts())
    
    print("\nCrosstab: Role Category vs Status:")
    print(pd.crosstab(df['role_category'], df['status']))
    
    print("\nMissing Country:")
    print(df[df['country'].isna()][['player_name', 'player_id']])

if __name__ == "__main__":
    main()
