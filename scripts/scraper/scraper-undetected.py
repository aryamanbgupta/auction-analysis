import json
import time
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from rpy2.robjects.packages import importr

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

    # Launch undetected chrome
    options = uc.ChromeOptions()
    # options.add_argument('--headless') # Start headed first to verify
    driver = uc.Chrome(options=options)

    try:
        for player_id in player_ids:
            url = f"https://www.espncricinfo.com/cricketers/{player_name.lower().replace(' ', '-')}-{player_id}"
            print(f"Querying URL: {url}")
            
            try:
                driver.get(url)
                
                # Wait for content to load
                wait = WebDriverWait(driver, 20)
                # Wait for the "Full Name" text or similar to appear
                wait.until(EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Full Name')]")))
                
                info = {
                    "player_id": player_id,
                    "bowling_style": "Not found",
                    "batting_style": "Not found",
                    "playing_role": "Not found",
                    "full_name": "Not found",
                    "teams": "Not found"
                }
                
                # Helper to find text safely
                def get_text_by_xpath(xpath):
                    try:
                        element = driver.find_element(By.XPATH, xpath)
                        return element.text.strip()
                    except:
                        return "Not found"

                # Extract details using XPath based on the structure we saw earlier
                # Structure: <p>Label</p><span>Value</span>
                
                info["full_name"] = get_text_by_xpath("//p[contains(text(), 'Full Name')]/following-sibling::span")
                if info["full_name"] == "Not found":
                     # Try h1
                     info["full_name"] = get_text_by_xpath("//h1[contains(@class, 'ds-text-title-l')]")

                info["bowling_style"] = get_text_by_xpath("//p[contains(text(), 'Bowling Style')]/following-sibling::span")
                info["batting_style"] = get_text_by_xpath("//p[contains(text(), 'Batting Style')]/following-sibling::span")
                info["playing_role"] = get_text_by_xpath("//p[contains(text(), 'Playing Role')]/following-sibling::span")
                
                # Teams
                # Structure: TEAMS -> div -> a -> span
                try:
                    teams_elements = driver.find_elements(By.XPATH, "//*[contains(text(), 'TEAMS')]/following-sibling::div//a")
                    teams = [t.text.strip() for t in teams_elements if t.text.strip()]
                    info["teams"] = '; '.join(teams)
                except:
                    pass
                
                all_info.append(info)
                
            except Exception as e:
                print(f"Error processing {url}: {e}")
                # Save screenshot for debug
                driver.save_screenshot(f"debug_{player_id}.png")
                
    finally:
        driver.quit()
    
    return all_info

if __name__ == "__main__":
    player_name = "V Kohli"
    print(f"Testing Undetected Chromedriver scraper for: {player_name}")
    
    start_time = time.time()
    info = get_player_info(player_name)
    end_time = time.time()
    
    print("\nResults:")
    print(json.dumps(info, indent=2))
    print(f"\nExecution time: {end_time - start_time:.2f} seconds")
