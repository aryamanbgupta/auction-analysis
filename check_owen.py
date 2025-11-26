
import pandas as pd
from pathlib import Path

def main():
    project_root = Path('.').resolve()
    ipl_file = project_root / 'data' / 'ipl_matches.parquet'
    
    df = pd.read_parquet(ipl_file)
    
    player_id = 'f2c936d7'
    
    balls_faced = len(df[df['batter_id'] == player_id])
    balls_bowled = len(df[df['bowler_id'] == player_id])
    
    print(f"Stats for {player_id}:")
    print(f"Balls faced: {balls_faced}")
    print(f"Balls bowled: {balls_bowled}")

if __name__ == "__main__":
    main()
