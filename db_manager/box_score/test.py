import os
import requests
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("API_KEY")

# Define the API endpoint and headers
API_ENDPOINT = "https://api.balldontlie.io/v1/games"
headers = {
    'Authorization': API_KEY  # or use 'Authorization': f'Bearer {API_KEY}' if required
}

# Parameters for the test request (optional)
params = {"per_page": 1}  # Requesting one item to keep it simple

try:
    # Make the request
    response = requests.get(API_ENDPOINT, headers=headers, params=params)
    
    # Check the response status code
    if response.status_code == 200:
        print("API key is working correctly. Response data:")
        print(response.json())  # Print a sample of the response
    elif response.status_code == 401:
        print("API key is not authorized (401). Check if the key is valid and correctly formatted.")
    elif response.status_code == 403:
        print("API key does not have permission to access this resource (403).")
    else:
        print(f"Received unexpected status code: {response.status_code}")
        print("Response content:", response.text)
except requests.exceptions.RequestException as e:
    print("An error occurred:", e)
