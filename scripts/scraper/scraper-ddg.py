from ddgs import DDGS
import re

def get_playing_role(player_identifier):
    # 1. Search DuckDuckGo for the player's ESPN Cricinfo profile
    # We add "espncricinfo playing role" to ensure the snippet contains the data we want
    query = f"{player_identifier} espncricinfo profile"
    
    print(f"Searching for: {query}...")
    
    # Use the Context Manager to avoid resource leaks
    with DDGS() as ddgs:
        # Get 1 result, text search
        results = list(ddgs.text(query, max_results=1))

    if not results:
        return "No results found."

    # 2. Extract the snippet (body) from the first result
    first_result = results[0]
    snippet = first_result.get('body', '')
    
    print(f"\n--- Raw Snippet Found ---\n{snippet}\n-------------------------")

    # 3. Use Regex to find "Playing Role"
    # Looking for: "Playing Role. <Something>. "
    # The snippet usually looks like: "... Batting Style. Right hand Bat. Playing Role. Top order Batter. TEAMS..."
    match = re.search(r"Playing Role\.?\s*(.*?)\.", snippet, re.IGNORECASE)

    if match:
        return match.group(1).strip()
    else:
        return "Playing role not found in snippet."

# --- Usage ---
role = get_playing_role("v-kohli-253802")
print(f"\nExtracted Playing Role: {role}")