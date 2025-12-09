import pandas as pd
import sys
import os
from pathlib import Path
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects

def setup_r_package():
    try:
        r_package = importr('cricketdata')
        return r_package
    except Exception as e:
        print(f"Error importing R package: {e}")
        sys.exit(1)

def get_player_ids(r_package, player_name):
    try:
        result = r_package.find_player_id(player_name)
        # Check if result is empty or None-like (R NULL)
        if result == robjects.NULL:
            return []
            
        try:
            player_ids = result.rx2('ID')
            return [str(int(id)) for id in player_ids]
        except Exception:
            # Sometimes parsing fails if structure is different
            return []
            
    except Exception as e:
        print(f"Error getting player IDs for {player_name}: {e}")
        return []

import time

def main():
    parser = argparse.ArgumentParser(description='Enrich auction list with player IDs and roles.')
    # ... (args) ...
    args = parser.parse_args()

    # ...
    
    for index, row in auction_df.iterrows():
        # ...
        if player_name in id_cache:
            cricinfo_id = id_cache[player_name]
        else:
            ids = get_player_ids(r_pkg, player_name)
            time.sleep(1) # Rate limit protection
            if ids:

                cricinfo_id = ids[0] # Take the first ID
            else:
                cricinfo_id = None
            id_cache[player_name] = cricinfo_id
            
        if cricinfo_id:
            auction_df.at[index, 'cricinfo_id'] = cricinfo_id
            found_ids += 1
            if found_ids % 10 == 0:
                print(f"Found IDs for {found_ids} players...")
        else:
            print(f"No ID found for: {player_name}")

    print(f"\nID Lookup Complete. Found {found_ids}/{total_players} IDs.")
    
    # Merge with roles data
    print("Merging with roles data...")
    final_df = auction_df.merge(roles_df, left_on='cricinfo_id', right_on='cricinfo_id', how='left')
    
    # Columns to prioritize/keep? 
    # The merge will bring in 'playing_role', 'batting_style', 'bowling_style', etc.
    # If naming conflicts, suffixes will be added. 
    # Assuming roles_df has 'name' which might conflict with 'PLAYER' (or be redundant)
    
    print(f"Saving enriched data to {output_path}")
    final_df.to_csv(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
