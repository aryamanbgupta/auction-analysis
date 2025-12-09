from espncricinfo.player import Player
import json

# 1. Safe subclass to bypass broken HTML parsing
class SafePlayer(Player):
    def _parse_player_information(self):
        return None

p = SafePlayer('277916') 

# Check the 'position' key
position_data = p.json.get('position')
print(f"Position Data: {position_data}")

# If position is a dictionary, the name is likely inside 'name' or 'description'
if isinstance(position_data, dict):
     print(f"Role Name: {position_data.get('name')}")

# 4. Correctly handling the 'style' list
# We iterate through the list to find batting/bowling styles
styles = p.json.get('style', [])
for item in styles:
    # Each 'item' is a dictionary inside the list
    style_type = item.get('type')
    description = item.get('description')
    print(f"{style_type.capitalize()} Style: {description}")