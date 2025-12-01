import json
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def extract_players_from_json(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        if 'info' in data and 'registry' in data['info'] and 'people' in data['info']['registry']:
            return data['info']['registry']['people']
        else:
            return {}
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}

def main():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    
    # Directories to search
    search_dirs = [
        data_dir / 'ipl_json',
        data_dir / 'other_t20_data'
    ]
    
    all_players = {} # ID -> Name
    
    print("Starting player extraction...")
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            print(f"Directory not found: {search_dir}")
            continue
            
        print(f"Scanning {search_dir}...")
        # Walk through directory
        json_files = list(search_dir.rglob('*.json'))
        
        for json_file in tqdm(json_files, desc=f"Processing {search_dir.name}"):
            players = extract_players_from_json(json_file)
            # Update master dict. 
            # Note: This will overwrite name if ID exists. 
            # Usually desirable to get latest name usage, but order is random here.
            # We assume ID is unique and Name is consistent enough.
            for name, pid in players.items():
                all_players[pid] = name
                
    print(f"Total unique players found: {len(all_players)}")
    
    # Convert to DataFrame
    df = pd.DataFrame(list(all_players.items()), columns=['cricsheet_id', 'name'])
    
    # Save
    output_path = data_dir / 'all_players_combined.csv'
    df.to_csv(output_path, index=False)
    print(f"Saved player list to {output_path}")

if __name__ == "__main__":
    main()
