
import pandas as pd
from pathlib import Path

def main():
    project_root = Path('.').resolve()
    csv_file = project_root / 'data' / 'updated_players_export.csv'
    
    df = pd.read_csv(csv_file)
    
    # Fix MJ Owen
    mask = df['player_id'] == 'f2c936d7'
    df.loc[mask, 'new_role_category'] = 'Batter'
    
    print("Updated row:")
    print(df[mask])
    
    df.to_csv(csv_file, index=False)
    print("Saved file.")

if __name__ == "__main__":
    main()
