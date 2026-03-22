"""
IPL Auction History Scraper.

Scrapes historical auction prices from IPLT20.com for players across multiple seasons.
Uses Playwright to handle JavaScript-rendered content.

Outputs: data/ipl_auction_historical.csv
"""

import asyncio
import pandas as pd
from pathlib import Path
from playwright.async_api import async_playwright
import re
import json
import time

# IPL Teams and their URLs
TEAMS = {
    'Chennai Super Kings': 'chennai-super-kings',
    'Mumbai Indians': 'mumbai-indians',
    'Royal Challengers Bengaluru': 'royal-challengers-bengaluru',
    'Kolkata Knight Riders': 'kolkata-knight-riders',
    'Rajasthan Royals': 'rajasthan-royals',
    'Delhi Capitals': 'delhi-capitals',
    'Punjab Kings': 'punjab-kings',
    'Sunrisers Hyderabad': 'sunrisers-hyderabad',
    'Gujarat Titans': 'gujarat-titans',
    'Lucknow Super Giants': 'lucknow-super-giants',
}

# Seasons to scrape (mini-auctions)
SEASONS = [2022, 2023, 2024, 2025]


async def scrape_auction_year(page, year):
    """Scrape sold players from a specific auction year."""
    url = f"https://www.iplt20.com/auction/{year}"
    print(f"Scraping auction {year}: {url}")
    
    try:
        await page.goto(url, timeout=30000)
        await page.wait_for_timeout(3000)  # Wait for JS to render
        
        # Click on "Sold Players" tab
        try:
            sold_tab = page.locator('text=Sold Players').first
            await sold_tab.click()
            await page.wait_for_timeout(2000)
        except Exception as e:
            print(f"  Could not find Sold Players tab: {e}")
        
        # Get all player cards
        players = []
        
        # Try different selectors for player data
        player_cards = await page.locator('.ap-plcard, .player-card, .auction-player').all()
        
        if not player_cards:
            print(f"  No player cards found with primary selector, trying alternatives...")
            # Try to get any elements with price data
            content = await page.content()
            
            # Parse with regex as fallback
            # Look for patterns like "Player Name" followed by price
            price_pattern = re.compile(r'₹\s*([\d.]+)\s*(Cr|Lakh|L)', re.IGNORECASE)
            
        for card in player_cards:
            try:
                name_el = await card.locator('.player-name, .ap-plname, h3, h4').first
                name = await name_el.text_content() if name_el else "Unknown"
                
                price_el = await card.locator('.ap-plprice, .price, .sold-price').first
                price_text = await price_el.text_content() if price_el else "0"
                
                team_el = await card.locator('.team-name, .ap-plteam').first
                team = await team_el.text_content() if team_el else "Unknown"
                
                players.append({
                    'player_name': name.strip() if name else "",
                    'price_text': price_text.strip() if price_text else "",
                    'team': team.strip() if team else "",
                    'year': year
                })
            except Exception as e:
                continue
        
        print(f"  Found {len(players)} players for {year}")
        return players
        
    except Exception as e:
        print(f"  Error scraping {year}: {e}")
        return []


async def scrape_team_squad(page, team_slug, team_name, year=2025):
    """Scrape team squad with prices."""
    url = f"https://www.iplt20.com/teams/{team_slug}/squad"
    print(f"Scraping {team_name}: {url}")
    
    try:
        await page.goto(url, timeout=30000)
        await page.wait_for_timeout(3000)
        
        content = await page.content()
        
        # Extract player links and names
        players = []
        player_links = await page.locator('a[href*="/players/"]').all()
        
        for link in player_links:
            try:
                href = await link.get_attribute('href')
                text = await link.text_content()
                
                if href and '/players/' in href and text:
                    # Extract player ID from href
                    match = re.search(r'/players/[\w-]+/(\d+)', href)
                    player_id = match.group(1) if match else None
                    
                    name = text.strip()
                    if name and len(name) > 2:  # Filter out empty/short strings
                        players.append({
                            'player_name': name,
                            'team': team_name,
                            'iplt20_id': player_id,
                            'year': year
                        })
            except:
                continue
        
        # Deduplicate
        seen = set()
        unique_players = []
        for p in players:
            if p['player_name'] not in seen:
                seen.add(p['player_name'])
                unique_players.append(p)
        
        print(f"  Found {len(unique_players)} players for {team_name}")
        return unique_players
        
    except Exception as e:
        print(f"  Error scraping {team_name}: {e}")
        return []


def parse_price(price_text):
    """Parse price text like '18 Cr' or '30 Lakh' to crores."""
    if not price_text:
        return 0.0
    
    price_text = str(price_text).lower().strip()
    
    # Remove currency symbols
    price_text = re.sub(r'[₹$,]', '', price_text)
    
    if 'cr' in price_text:
        match = re.search(r'([\d.]+)', price_text)
        if match:
            return float(match.group(1))
    elif 'lakh' in price_text or 'l' in price_text:
        match = re.search(r'([\d.]+)', price_text)
        if match:
            return float(match.group(1)) / 100
    
    return 0.0


async def main():
    print("=" * 60)
    print("IPL AUCTION HISTORY SCRAPER")
    print("=" * 60)
    
    project_root = Path(__file__).parent.parent.parent
    output_file = project_root / 'data' / 'ipl_auction_historical.csv'
    
    all_players = []
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--no-sandbox',
            ]
        )
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080}
        )
        page = await context.new_page()
        
        # Method 1: Try auction pages
        print("\n--- Scraping Auction Pages ---")
        for year in SEASONS:
            players = await scrape_auction_year(page, year)
            all_players.extend(players)
            await asyncio.sleep(1)
        
        # Method 2: Scrape team squads (current)
        print("\n--- Scraping Team Squads ---")
        for team_name, team_slug in TEAMS.items():
            players = await scrape_team_squad(page, team_slug, team_name)
            all_players.extend(players)
            await asyncio.sleep(1)
        
        await browser.close()
    
    # Create DataFrame
    if all_players:
        df = pd.DataFrame(all_players)
        
        # Parse prices
        if 'price_text' in df.columns:
            df['price_cr'] = df['price_text'].apply(parse_price)
        
        # Clean up
        df = df.drop_duplicates(subset=['player_name', 'year'], keep='first')
        
        # Save
        df.to_csv(output_file, index=False)
        print(f"\n✓ Saved {len(df)} player records to {output_file}")
        
        # Summary
        print("\nSummary by year:")
        print(df.groupby('year').size())
    else:
        print("\nNo data scraped. The website structure may have changed.")
        print("Consider using manual data entry or Kaggle download instead.")
    
    return all_players


if __name__ == "__main__":
    asyncio.run(main())
