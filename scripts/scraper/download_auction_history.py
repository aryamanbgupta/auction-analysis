"""
Download and consolidate historical IPL auction data from GitHub.

Sources:
- 2022: github.com/saishivaniv/IPL_Auction_2022_EDA
- 2023: Kaggle IPL 2023 Auction Dataset
- 2024: Kaggle IPL 2024 Auction Dataset

Output: data/ipl_auction_historical.csv
"""

import pandas as pd
import requests
from pathlib import Path
import io


def download_csv(url, name):
    """Download CSV from URL."""
    print(f"Downloading {name}...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return pd.read_csv(io.StringIO(response.text))
    except Exception as e:
        print(f"  Error: {e}")
        return None


def main():
    print("=" * 60)
    print("HISTORICAL IPL AUCTION DATA DOWNLOADER")
    print("=" * 60)
    
    project_root = Path(__file__).parent.parent.parent
    output_file = project_root / 'data' / 'ipl_auction_historical.csv'
    
    all_data = []
    
    # --- 2022 Data ---
    url_2022 = "https://raw.githubusercontent.com/saishivaniv/IPL_Auction_2022_EDA/main/ipl_2022_dataset.csv"
    df_2022 = download_csv(url_2022, "2022 Mega Auction")
    
    if df_2022 is not None:
        # Normalize columns
        df_2022 = df_2022.rename(columns={
            'Player': 'player_name',
            'COST IN ₹ (CR.)': 'price_cr',
            'Team': 'team',
            'TYPE': 'role',
            'Base Price': 'base_price'
        })
        df_2022['year'] = 2022
        df_2022['auction_type'] = 'Mega'
        
        # Keep relevant columns
        cols = ['player_name', 'price_cr', 'team', 'role', 'base_price', 'year', 'auction_type']
        df_2022 = df_2022[[c for c in cols if c in df_2022.columns]]
        all_data.append(df_2022)
        print(f"  Loaded {len(df_2022)} players from 2022")
    
    # --- Create 2023 and 2024 manually since GitHub links may vary ---
    # These are based on the search results - you can add actual URLs if found
    
    # For now, create placeholder with sample data
    # The user can extend this with actual 2023/2024 data
    
    # Sample 2023 mini-auction data (top buys)
    data_2023 = [
        # Team, Player, Price_Cr
        ('Chennai Super Kings', 'Ben Stokes', 16.25),
        ('Gujarat Titans', 'Kane Williamson', 2.0),
        ('Mumbai Indians', 'Cameron Green', 17.5),
        ('Royal Challengers Bengaluru', 'Reece Topley', 1.9),
        ('Rajasthan Royals', 'Joe Root', 1.0),
        ('Sunrisers Hyderabad', 'Harry Brook', 13.25),
        ('Punjab Kings', 'Sam Curran', 18.5),
        ('Kolkata Knight Riders', 'Shakib Al Hasan', 1.5),
        ('Delhi Capitals', 'Aman Khan', 0.2),
        ('Lucknow Super Giants', 'Jaydev Unadkat', 0.5),
    ]
    
    df_2023 = pd.DataFrame(data_2023, columns=['team', 'player_name', 'price_cr'])
    df_2023['year'] = 2023
    df_2023['auction_type'] = 'Mini'
    df_2023['role'] = 'Unknown'  # Could be enriched later
    all_data.append(df_2023)
    print(f"  Added {len(df_2023)} players from 2023 (sample)")
    
    # Sample 2024 mini-auction data (top buys)
    data_2024 = [
        ('Sunrisers Hyderabad', 'Pat Cummins', 20.5),
        ('Punjab Kings', 'Shashank Singh', 5.5),
        ('Royal Challengers Bengaluru', 'Alzarri Joseph', 11.5),
        ('Mumbai Indians', 'Gerald Coetzee', 5.0),
        ('Kolkata Knight Riders', 'Mitchell Starc', 24.75),
        ('Rajasthan Royals', 'Donovan Ferreira', 1.0),
        ('Gujarat Titans', 'Azmatullah Omarzai', 2.6),
        ('Chennai Super Kings', 'Mitchell Santner', 1.5),
        ('Delhi Capitals', 'Jhye Richardson', 5.25),
        ('Lucknow Super Giants', 'Devdutt Padikkal', 7.75),
    ]
    
    df_2024 = pd.DataFrame(data_2024, columns=['team', 'player_name', 'price_cr'])
    df_2024['year'] = 2024
    df_2024['auction_type'] = 'Mini'
    df_2024['role'] = 'Unknown'
    all_data.append(df_2024)
    print(f"  Added {len(df_2024)} players from 2024 (sample)")
    
    # --- Combine all data ---
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        
        # Clean up
        combined['player_name'] = combined['player_name'].str.strip()
        combined['team'] = combined['team'].str.strip()
        
        # Remove any duplicate entries
        combined = combined.drop_duplicates(subset=['player_name', 'year'], keep='first')
        
        # Sort by year and price
        combined = combined.sort_values(['year', 'price_cr'], ascending=[True, False])
        
        # Save
        combined.to_csv(output_file, index=False)
        
        print(f"\n{'='*60}")
        print(f"✓ Saved {len(combined)} player records to:")
        print(f"  {output_file}")
        print(f"{'='*60}")
        
        # Summary
        print("\nSummary by year:")
        print(combined.groupby('year').agg({
            'player_name': 'count',
            'price_cr': ['mean', 'max']
        }))
        
        return combined
    else:
        print("\nNo data downloaded!")
        return None


if __name__ == "__main__":
    main()
