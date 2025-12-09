from rpy2.robjects.packages import importr
import rpy2.robjects as robjects

try:
    print("Attempting to import cricketdata...")
    r_package = importr('cricketdata')
    print("Successfully imported cricketdata")
    
    # Test lookup
    player_name = "Devon Conway"
    print(f"Looking up {player_name}...")
    
    # Using the user-provided logic structure
    result = r_package.find_player_id(player_name)
    print(f"Raw Result Type: {type(result)}")
    # print("Raw Result:", result)
    
    try:
        player_ids = result.rx2('ID')
        ids = [str(int(id)) for id in player_ids]
        print(f"Extracted IDs: {ids}")
    except Exception as e:
        print(f"Extraction failed: {e}")
        # Inspect columns if extraction fails
        try:
            print("Columns:", result.colnames)
        except:
            pass

except Exception as e:
    print(f"Error: {e}")
