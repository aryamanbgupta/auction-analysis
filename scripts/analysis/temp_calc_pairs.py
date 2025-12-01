import pandas as pd
from pathlib import Path

def main():
    root = Path('.')
    try:
        ipl_df = pd.read_parquet(root / 'results' / '06_context_adjustments' / 'ipl_with_raa.parquet')
        df_2025 = ipl_df[ipl_df['season'] == '2025']
        openers = df_2025[(df_2025['over'] == 0) & (df_2025['ball_in_over'] == 1)].copy()
        openers['pair'] = openers.apply(lambda x: tuple(sorted([x['batter_name'], x['non_striker_name']])), axis=1)
        stability = openers.groupby('batting_team')['pair'].nunique().reset_index(name='Unique_Pairs')
        
        dc_pairs = stability[stability['batting_team'] == 'Delhi Capitals']['Unique_Pairs'].values[0]
        league_avg = stability[stability['batting_team'] != 'Delhi Capitals']['Unique_Pairs'].mean()
        
        print(f"DC_PAIRS:{dc_pairs}")
        print(f"LEAGUE_AVG:{league_avg:.1f}")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
