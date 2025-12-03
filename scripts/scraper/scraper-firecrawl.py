import requests
import os
import dotenv

dotenv.load_dotenv()

url = "https://api.firecrawl.dev/v2/search"

payload = {
  "query": "v-kohli-253802 playing role",
  "sources": [
    "web"
  ],
  "categories": [],
  "limit": 1,
  "scrapeOptions": {
    "onlyMainContent": False,
    "maxAge": 172800000,
    "parsers": [
      "pdf"
    ],
    "formats": []
  }
}

headers = {
    "Authorization": "Bearer " + os.getenv("API_KEY"),
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

print(response.json())