
import pandas as pd
from pathlib import Path

def main():
    project_root = Path('.').resolve()
    metadata_file = project_root / 'data' / 'player_metadata.csv'
    
    df = pd.read_csv(metadata_file)
    
    print("Unique Playing Roles:")
    print(df['playing_role'].value_counts())
    
    print("\nUnique Bowling Types:")
    print(df['bowling_type'].value_counts())
    
    print("\nCrosstab: Role Category vs Bowling Type:")
    print(pd.crosstab(df['role_category'], df['bowling_type']))

    # Check "Batter" specific roles
    batters = df[df['role_category'] == 'batter']
    print("\nBreakdown of 'batter' category by playing_role:")
    print(batters['playing_role'].value_counts())

if __name__ == "__main__":
    main()
