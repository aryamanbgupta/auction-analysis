
import pandas as pd
from pathlib import Path

def main():
    project_root = Path('.').resolve()
    metadata_file = project_root / 'data' / 'player_metadata.csv'
    
    df = pd.read_csv(metadata_file)
    
    print("Bowling Type counts:")
    print(df['bowling_type'].value_counts())
    
    print("\nRole Category counts:")
    print(df['role_category'].value_counts())
    
    print("\nSpinners not in 'Spinner' category:")
    spinners = df[df['bowling_type'] == 'spin']
    non_cat_spinners = spinners[spinners['role_category'] != 'Spinner']
    print(non_cat_spinners[['player_name', 'playing_role', 'role_category']])

if __name__ == "__main__":
    main()
