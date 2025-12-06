import pandas as pd
import argparse
import os
import json
import requests
import time
from pathlib import Path
import sys

# Add the current directory to path to allow imports if needed, 
# but better to use relative imports or assume running from root.
# We will assume running from project root.
sys.path.append(str(Path(__file__).parent))

try:
    from parse_firecrawl_output import extract_playing_role
except ImportError:
    # Fallback if running from a different directory context
    sys.path.append(str(Path(__file__).parent))
    from parse_firecrawl_output import extract_playing_role

def get_firecrawl_data(query, api_key):
    url = "https://api.firecrawl.dev/v2/search"
    
    payload = {
      "query": query,
      "sources": ["web"],
      "categories": [],
      "limit": 1,
      "scrapeOptions": {
        "onlyMainContent": False,
        "maxAge": 172800000, # 2 days
        "parsers": ["pdf"],
        "formats": []
      }
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching data for query '{query}': {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Enrich player data with Playing Role from Firecrawl.')
    parser.add_argument('--dry-run', action='store_true', help='Print queries without making API calls')
    parser.add_argument('--limit', type=int, default=5, help='Number of players to process in dry run')
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / 'data'
    input_path = data_dir / 'all_players_enriched.csv'
    output_path = data_dir / 'all_players_enriched_roles.csv'
    
    # Load data
    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path)
    
    # Filter for players with cricinfo_id
    # Ensure cricinfo_id is treated as string (it might be float with NaNs)
    df['cricinfo_id'] = df['cricinfo_id'].fillna(0).astype(int).astype(str)
    
    # Filter out '0' or empty IDs if any
    mask = df['cricinfo_id'] != '0'
    players_to_process = df[mask].copy()
    
    print(f"Found {len(players_to_process)} players with Cricinfo IDs.")
    
    if args.dry_run:
        print(f"\n--- DRY RUN MODE (Showing first {args.limit} queries) ---")
        count = 0
        for idx, row in players_to_process.iterrows():
            if count >= args.limit:
                break
                
            name = row['name']
            cricinfo_id = row['cricinfo_id']
            
            # Format: "v-kohli-253802 playing role"
            # We need to slugify the name slightly to match the user's example "v-kohli"
            # But the user's example "v-kohli-253802" suggests using the name as is or simplified.
            # The prompt said: 'creates a search query like "v-kohli-253802 playing role"'
            # Let's try to match that format: lowercase, spaces to hyphens.
            
            slug_name = str(name).lower().strip().replace(' ', '-')
            query = f"{slug_name}-{cricinfo_id} playing role"
            
            print(f"Player: {name} (ID: {cricinfo_id}) -> Query: '{query}'")
            count += 1
            
        print("\nDry run complete. No API calls made.")
        return

    # LIVE MODE
    # Load API Key
    from dotenv import load_dotenv
    load_dotenv(project_root / '.env')
    api_key = os.getenv("FIRECRAWL_API_KEY") or os.getenv("API_KEY")
    
    if not api_key:
        print("Error: API_KEY not found in environment variables.")
        return
        
    print("Starting live enrichment...")
    
    results = []
    count = 0
    
    for idx, row in players_to_process.iterrows():
        if count >= args.limit:
            break
            
        name = row['name']
        cricinfo_id = row['cricinfo_id']
        slug_name = str(name).lower().strip().replace(' ', '-')
        query = f"{slug_name}-{cricinfo_id} playing role"
        
        print(f"\n[{count+1}/{args.limit}] Processing: {name} (ID: {cricinfo_id})")
        print(f"  Query: {query}")
        
        # API Call
        data = get_firecrawl_data(query, api_key)
        
        if data:
            # Extraction
            role = extract_playing_role(data)
            print(f"  Extracted Role: {role}")
            
            # Store result
            results.append({
                'cricsheet_id': row['cricsheet_id'],
                'name': name,
                'cricinfo_id': cricinfo_id,
                'playing_role': role
            })
        else:
            print("  Failed to get data.")
            
        count += 1
        # Be nice to the API
        time.sleep(1)
        
    # Save results if any
    if results:
        results_df = pd.DataFrame(results)
        print(f"\nProcessed {len(results)} players.")
        print("Results:")
        print(results_df[['name', 'playing_role']])
        
        # For this test run, we won't overwrite the main file yet, just show output.
        # But if we wanted to save:
        # output_df = df.merge(results_df[['cricsheet_id', 'playing_role']], on='cricsheet_id', how='left')
        # output_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
