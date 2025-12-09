import pandas as pd
from espncricinfo.player import Player
import time
from pathlib import Path
import sys
import json

# Safe subclass to bypass broken HTML parsing as provided by user
class SafePlayer(Player):
    def _parse_player_information(self):
        return None

def get_playing_role(cricinfo_id):
    try:
        # Ensure ID is a string and valid
        if pd.isna(cricinfo_id) or cricinfo_id == '0' or cricinfo_id == 0:
            return "Unknown"
            
        p = SafePlayer(str(int(cricinfo_id)))
        
        # Check the 'position' key
        position_data = p.json.get('position')
        
        if isinstance(position_data, dict):
             return position_data.get('name', "Unknown")
        elif position_data:
            return str(position_data)
            
        return "Unknown"
    except Exception as e:
        # print(f"Error fetching for ID {cricinfo_id}: {e}")
        return "Error"

def main():
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / 'data'
    input_path = data_dir / 'all_players_enriched.csv'
    output_path = data_dir / 'all_players_enriched_roles_api.csv'
    
    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path)
    
    # Filter for players with valid cricinfo_id for processing, but we want to keep all rows
    # We will iterate and update.
    
    print(f"Total players: {len(df)}")
    
    print(f"Total players: {len(df)}")
    
    # Use ThreadPoolExecutor for parallel fetching
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    roles_map = {}
    
    start_time = time.time()
    
    # Create a list of (index, cricinfo_id) tuples to process
    tasks = []
    for idx, row in df.iterrows():
        tasks.append((idx, row.get('cricinfo_id')))
        
    print("Starting parallel fetch with 10 workers...")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all tasks
        future_to_idx = {executor.submit(get_playing_role, cid): idx for idx, cid in tasks}
        
        completed = 0
        total = len(tasks)
        
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                role = future.result()
                roles_map[idx] = role
            except Exception as e:
                roles_map[idx] = "Error"
            
            completed += 1
            if completed % 100 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                remaining = (total - completed) / rate
                print(f"Processed {completed}/{total} ({rate:.1f} players/sec). Est. remaining: {remaining/60:.1f} min")

    # Reconstruct the roles list in order
    roles = [roles_map[i] for i in range(len(df))]
    
    df['playing_role'] = roles
    
    print(f"Saving to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
