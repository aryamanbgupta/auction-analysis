
import requests
from bs4 import BeautifulSoup
import re

def get_player_role(player_id):
    url = f"https://www.espncricinfo.com/ci/content/player/{player_id}.html"
    print(f"Fetching {url}...")
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for playing role in the profile
            # Usually in a grid or list
            # Example text: "Playing role Bowler"
            
            # Method 1: Search for "Playing role" text
            role_tag = soup.find(string=re.compile("Playing role"))
            if role_tag:
                parent = role_tag.parent
                # The value might be in the next sibling or parent's sibling
                print(f"Found 'Playing role' tag: {role_tag}")
                # traversing to find the value
                # often it's like <p><span>Playing role</span> <span>Bowler</span></p>
                # or in a grid
                
                # Let's print some context
                print(f"Parent text: {parent.text}")
                print(f"Parent next sibling: {parent.next_sibling}")
                
            # Method 2: Look for specific class names (might change)
            # Method 3: Just dump text and search
            text = soup.get_text()
            match = re.search(r"Playing role\s+([A-Za-z\s]+)", text)
            if match:
                print(f"Regex match: {match.group(1).strip()}")
                return match.group(1).strip()
                
        else:
            print(f"Failed to fetch page: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # B Kumar
    get_player_role("326016")
