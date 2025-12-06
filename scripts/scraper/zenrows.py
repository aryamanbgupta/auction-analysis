# pip install requests
import requests

url = 'https://www.espncricinfo.com/cricketers/virat-kohli-253802'
apikey = ''
params = {
    'url': url,
    'apikey': apikey,
	'js_render': 'true',
	'json_response': 'true',
	'premium_proxy': 'true',
	'response_type': 'markdown',
}
response = requests.get('https://api.zenrows.com/v1/', params=params)
print(response.text)