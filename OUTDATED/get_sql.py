import json
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv

load_dotenv()

# Fetch the values from environment variables
api_key = os.getenv("API_KEY")
db_user = os.getenv("DB_USER")
db_pass = os.getenv("DB_PASS")
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")

conn = psycopg2.connect(
    dbname="box_scores",
    user=db_user,
    password=db_pass,
    host=db_host,
    port=db_port
)

print("Connected to the database successfully.")

# Query to fetch player box scores
query = """
SELECT
    p.player_id,
    p.first_name,
    p.last_name,
    g.game_id,
    g.date AS game_date,
    pg.min,
    pg.fgm,
    pg.fga,
    pg.fg_pct,
    pg.fg3m,
    pg.fg3a,
    pg.fg3_pct,
    pg.ftm,
    pg.fta,
    pg.ft_pct,
    pg.oreb,
    pg.dreb,
    pg.reb,
    pg.ast,
    pg.stl,
    pg.blk,
    pg.turnover,
    pg.pf,
    pg.pts
FROM
    player_game pg
JOIN
    player p ON pg.player_id = p.player_id
JOIN
    game g ON pg.game_id = g.game_id
ORDER BY
    p.player_id, g.date;
"""

# Execute the query and fetch results
with conn.cursor(cursor_factory=RealDictCursor) as cursor:
    cursor.execute(query)
    result = cursor.fetchall()

# Process results into desired JSON structure
players = {}
for row in result:
    player_id = row['player_id']
    if player_id not in players:
        players[player_id] = {
            'player_id': player_id,
            'first_name': row['first_name'],
            'last_name': row['last_name'],
            'games': []
        }
    # Append game stats to player's list of games
    game_data = {
        'game_id': row['game_id'],
        'game_date': row['game_date'],
        'min': row['min'],
        'fgm': row['fgm'],
        'fga': row['fga'],
        'fg_pct': row['fg_pct'],
        'fg3m': row['fg3m'],
        'fg3a': row['fg3a'],
        'fg3_pct': row['fg3_pct'],
        'ftm': row['ftm'],
        'fta': row['fta'],
        'ft_pct': row['ft_pct'],
        'oreb': row['oreb'],
        'dreb': row['dreb'],
        'reb': row['reb'],
        'ast': row['ast'],
        'stl': row['stl'],
        'blk': row['blk'],
        'turnover': row['turnover'],
        'pf': row['pf'],
        'pts': row['pts']
    }
    players[player_id]['games'].append(game_data)

# Convert the dictionary to JSON format
json_output = json.dumps(list(players.values()), indent=4, default=str)

# Save JSON output to a file
with open("data/all_box.json", "w") as json_file:
    json_file.write(json_output)

# Close the database connection
conn.close()

print("Box scores for all players have been saved to 'data/all_box.json'")
