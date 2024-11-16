#!/bin/bash

# API Call and Save to Response File
curl 'https://api.prizepicks.com/projections?league_id=7&per_page=250&single_stat=true&game_mode=pickem' \
  -H 'sec-ch-ua: "Not/A)Brand";v="8", "Chromium";v="126", "Google Chrome";v="126"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36' \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json' \
  -H 'Referer: https://app.prizepicks.com/' \
  -H 'X-Device-ID: 1a9d6304-65f3-4304-8523-ccf458d3c0c4' \
  -H 'sec-ch-ua-platform: "macOS"' \
  -o data/response.json

# Pretty-print the JSON using jq
jq '.' data/response.json > data/formatted_response.json

echo "Formatted JSON saved to 'data/formatted_response.json'"
