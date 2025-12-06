import json
import re

def extract_playing_role(firecrawl_response):
    """
    Extracts the Playing Role from the Firecrawl API response.
    """
    try:
        # Navigate to the description field
        # Structure: {'success': True, 'data': {'web': [{'description': '...'}]}}
        if not firecrawl_response.get('success', False):
            return "Error: Request was not successful"
            
        web_results = firecrawl_response.get('data', {}).get('web', [])
        if not web_results:
            return "Error: No web results found"
            
        # Get the first result's description
        description = web_results[0].get('description', '')
        print(f"Description found: {description}")
        
        # Regex to find "Playing Role. <Role>."
        # We look for "Playing Role" followed by a dot/space, then capture until the next dot.
        match = re.search(r"Playing Role\.?\s*([^.]+)", description, re.IGNORECASE)
        
        if match:
            role = match.group(1).strip()
            return role
        else:
            return "Error: Playing Role not found in description"
            
    except Exception as e:
        return f"Error parsing response: {str(e)}"

def main():
    # The sample output provided by the user
    sample_output = {
        'success': True, 
        'data': {
            'web': [
                {
                    'url': 'https://www.espncricinfo.com/cricketers/virat-kohli-253802', 
                    'title': 'Virat Kohli Profile - Cricket Player India | Stats, Records, Video', 
                    'description': 'Virat Kohli. Born. November 05, 1988, Delhi. Age. 37y 30d. Batting Style. Right hand Bat. Bowling Style. Right arm Medium. Playing Role. Top order Batter. TEAMS.', 
                    'position': 1
                }
            ]
        }, 
        'creditsUsed': 2
    }
    
    print("Testing extraction on sample output...")
    role = extract_playing_role(sample_output)
    print(f"\nExtracted Role: '{role}'")

if __name__ == "__main__":
    main()
