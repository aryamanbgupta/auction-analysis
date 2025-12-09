import requests
import pandas as pd
import re
import json
from io import StringIO

def get_datawrapper_data(chart_id, version=1):
    """
    Fetches data from a Datawrapper chart.
    Tries the direct CSV endpoint first, then falls back to parsing the HTML.
    """
    # Method 1: Try direct CSV download
    csv_url = f"https://datawrapper.dwcdn.net/{chart_id}/{version}/dataset.csv"
    print(f"Attempting to fetch CSV from: {csv_url}")
    
    try:
        response = requests.get(csv_url)
        if response.status_code == 200:
            print("Success! CSV found.")
            # Read CSV content into a pandas DataFrame
            data = pd.read_csv(StringIO(response.text))
            return data
    except Exception as e:
        print(f"CSV method failed: {e}")

    # Method 2: Fallback to scraping the script tag
    print("CSV not accessible. Attempting to parse HTML script tags...")
    embed_url = f"https://datawrapper.dwcdn.net/{chart_id}/{version}/"
    response = requests.get(embed_url)
    
    if response.status_code != 200:
        print(f"Failed to load chart page: {response.status_code}")
        return None

    # Use regex to find the data object inside the script
    # Looking for: "data": {...} inside window.__dw.init
    pattern = r'"data":(\{.*?\})\,'
    matches = re.search(pattern, response.text)
    
    if matches:
        json_str = matches.group(1)
        try:
            data_dict = json.loads(json_str)
            # Datawrapper often stores the actual table data in a simplified format here
            # or implies the CSV structure. If this part is complex, usually Method 1 works.
            print("Data object found in script!")
            return data_dict
        except json.JSONDecodeError:
            print("Could not decode JSON data.")
    
    return None

# --- Main Execution ---
CHART_ID = "5cWjL"  # Extracted from your HTML id="datawrapper-chart-5cWjL"
VERSION = "8"       # Extracted from the src URL

df = get_datawrapper_data(CHART_ID, VERSION)

if df is not None:
    print("\n--- Extracted Data (First 5 Rows) ---")
    print(df.head())
    
    # Save to file
    filename = "ipl_2026_auction_list.csv"
    df.to_csv(filename, index=False)
    print(f"\nFull dataset saved to {filename}")
else:
    print("Could not extract data.")