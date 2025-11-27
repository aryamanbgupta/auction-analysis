
import pandas as pd
from pathlib import Path

def main():
    project_root = Path('.').resolve()
    metadata_file = project_root / 'data' / 'player_metadata.csv'
    
    df = pd.read_csv(metadata_file)
    unknowns = df[df['role_category'] == 'unknown']
    
    print(f"Found {len(unknowns)} unknown players:")
    for _, row in unknowns.iterrows():
        print(f"ID: {row['player_id']}, Name: {row['player_name']}, Role: {row['playing_role']}")

if __name__ == "__main__":
    main()
