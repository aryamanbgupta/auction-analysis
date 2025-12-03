import json
import time
import re
from bs4 import BeautifulSoup
from rpy2.robjects.packages import importr
from playwright.sync_api import sync_playwright

# Import the R package that contains find_player_id
try:
    r_package = importr('cricketdata')
except Exception as e:
    print(f"Error importing R package: {e}")
    exit(1)

def get_player_ids(player_name):
    try:
        result = r_package.find_player_id(player_name)
        player_ids = result.rx2('ID')
        return [str(int(id)) for id in player_ids]
    except Exception as e:
        print(f"Error getting player IDs for {player_name}: {e}")
        return []

def get_player_info(player_name):
    player_ids = get_player_ids(player_name)
    all_info = []
    
    if not player_ids:
        print(f"No IDs found for {player_name}")
        return []

    with sync_playwright() as p:
        # Launch browser with stealth args in HEADED mode
        browser = p.chromium.launch(
            headless=False,  # Visible browser
            args=[
                '--disable-blink-features=AutomationControlled',
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-infobars',
                '--window-position=0,0',
                '--ignore-certifcate-errors',
                '--ignore-certifcate-errors-spki-list',
                '--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            ]
        )
        context = browser.new_context(
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080}
        )
        
        # Inject script to hide webdriver property
        context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """)
        
        for player_id in player_ids:
            url = f"https://www.espncricinfo.com/cricketers/{player_name.lower().replace(' ', '-')}-{player_id}"
            print(f"Querying URL: {url}")
            
            try:
                page = context.new_page()
                page.goto(url, timeout=30000)
                
                # Wait for challenge to solve/page to load
                page.wait_for_timeout(5000)
                
                content = page.content()
                
                # Debug: Save content to file
                with open('debug_page_headed.html', 'w') as f:
                    f.write(content)
                print("Saved page content to debug_page_headed.html")
                
                soup = BeautifulSoup(content, 'html.parser')
                
                info = {
                    "player_id": player_id,
                    "bowling_style": "Not found",
                    "batting_style": "Not found",
                    "playing_role": "Not found",
                    "full_name": "Not found",
                    "teams": "Not found"
                }
                
                # Helper to find text safely
                def find_text(soup, pattern):
                    return soup.find(lambda tag: tag.name in ['p', 'div', 'span'] and tag.text and re.search(pattern, tag.text, re.IGNORECASE))

                # Extract bowling style
                bowling_style_tag = find_text(soup, r'Bowling\s+Style')
                if bowling_style_tag:
                    sibling = bowling_style_tag.find_next_sibling('span')
                    if sibling:
                        info["bowling_style"] = sibling.text.strip()
                
                # Extract batting style
                batting_style_tag = find_text(soup, r'Batting\s+Style')
                if batting_style_tag:
                    sibling = batting_style_tag.find_next_sibling('span')
                    if sibling:
                        info["batting_style"] = sibling.text.strip()
                
                # Extract playing role
                playing_role_tag = find_text(soup, r'Playing\s+Role')
                if playing_role_tag:
                    sibling = playing_role_tag.find_next_sibling('span')
                    if sibling:
                        info["playing_role"] = sibling.text.strip()
                
                # Extract full name
                full_name_tag = find_text(soup, r'Full\s+Name')
                if full_name_tag:
                    sibling = full_name_tag.find_next_sibling('span')
                    if sibling:
                        info["full_name"] = sibling.text.strip()
                
                # Extract teams
                teams_tag = find_text(soup, r'TEAMS')
                if teams_tag:
                    teams_grid = teams_tag.find_next_sibling('div')
                    if teams_grid:
                        team_links = teams_grid.find_all('a')
                        teams = [link.get_text(strip=True) for link in team_links]
                        info["teams"] = '; '.join(teams)
                
                all_info.append(info)
                page.close()
                
            except Exception as e:
                print(f"Error processing {url}: {e}")
        
        browser.close()
    
    return all_info

if __name__ == "__main__":
    player_name = "V Kohli"
    print(f"Testing Playwright scraper for: {player_name}")
    
    start_time = time.time()
    info = get_player_info(player_name)
    end_time = time.time()
    
    print("\nResults:")
    print(json.dumps(info, indent=2))
    print(f"\nExecution time: {end_time - start_time:.2f} seconds")
