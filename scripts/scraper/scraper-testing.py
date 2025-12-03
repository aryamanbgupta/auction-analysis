import csv
import requests
from bs4 import BeautifulSoup
from rpy2 import robjects
from rpy2.robjects.packages import importr
from collections import Counter
import os
import json
import time
import shutil


# Import the R package that contains find_player_id
r_package = importr('cricketdata')



def get_player_ids(player_name):
    result = r_package.find_player_id(player_name)
    player_ids = result.rx2('ID')
    return [str(int(id)) for id in player_ids]

def get_player_info(player_name):
    player_ids = get_player_ids(player_name)
    all_info = []
    
    for player_id in player_ids:
        url = f"https://www.espncricinfo.com/cricketers/{player_name.lower().replace(' ', '-')}-{player_id}"
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Cache-Control': 'max-age=0',
            }
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                info = {
                    "player_id": player_id,
                    "bowling_style": "Not found",
                    "batting_style": "Not found",
                    "playing_role": "Not found",
                    "full_name": "Not found",
                    "teams": "Not found"
                }
                
                    # Extract bowling style
                bowling_style_div = soup.find('p', string='Bowling Style')
                if bowling_style_div:
                    info["bowling_style"] = bowling_style_div.find_next_sibling('span').text.strip()
                
                # Extract batting style
                batting_style_div = soup.find('p', string='Batting Style')
                if batting_style_div:
                    info["batting_style"] = batting_style_div.find_next_sibling('span').text.strip()
                
                # Extract playing role
                playing_role_div = soup.find('p', string='Playing Role')
                if playing_role_div:
                    info["playing_role"] = playing_role_div.find_next_sibling('span').text.strip()
                
                # Extract full name
                full_name_div = soup.find('p', string='Full Name')
                if full_name_div:
                    full_name_span = full_name_div.find_next_sibling('span', class_='ds-text-title-s')
                    if full_name_span:
                        info["full_name"] = full_name_span.text.strip()
                
                # Extract teams
                teams_div = soup.find('p', string='TEAMS')
                if teams_div:
                    teams_grid = teams_div.find_next_sibling('div', class_='ds-grid')
                    if teams_grid:
                        team_links = teams_grid.find_all('a')
                        teams = [link['href'].split('/')[-1] for link in team_links]
                        info["teams"] = '; '.join(teams)
                
                all_info.append(info)
            else:
                print(f"Failed to retrieve the page for {player_name} (ID: {player_id}). Status code: {response.status_code}")
        
        except requests.RequestException as e:
            print(f"Request failed for {player_name} (ID: {player_id}): {str(e)}")
    
    # Cache the results - REMOVED for testing
    # with open(cache_file, 'w') as f:
    #     json.dump(all_info, f)
    
    return all_info

# def process_names_csv(input_file, output_file, tally_file):
#     # ... (CSV processing logic commented out) ...
#     pass

# Main execution
# input_file = 'names_v2.csv'
# output_file = 'players_info.csv'
# tally_file = 'bowling_style_tally.csv'

# start_time = time.time()
# process_names_csv(input_file, output_file, tally_file)
# end_time = time.time()

if __name__ == "__main__":
    player_name = "V Kohli"
    print(f"Testing scraper for: {player_name}")
    
    start_time = time.time()
    info = get_player_info(player_name)
    end_time = time.time()
    
    print("\nResults:")
    print(json.dumps(info, indent=2))
    print(f"\nExecution time: {end_time - start_time:.2f} seconds")