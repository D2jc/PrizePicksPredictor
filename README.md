# PrizePicksPredictor

## Preprocessing Steps:

1. Drop Irrelevant Columns: There are many columns such as game_status, game_period, player_jersey_number, player_college, player_country, player_draft_year, player_draft_round, player_draft_number, team_conference, team_division, team_city, team_abbreviation, game_postseason, game_status, game_time and much more are not very important for predictions on a player's particular stats such as for how many points they make / rebounds they receive / assists they make.
2. We would try to account for any missing values across columns. For specific game statistics such as points, rebounds, assists, etc. we would impute the mean or median of the position that the player is and fill it in. For categorical columns like player_position or team_name, we believe filling in "Unknown" would make the most sense.
3. We would prepare a time-series data for lagged versions of key stats (points, assists, rebounds) to capture the player’s performance trends over recent games. For example, features like points_last_game, points_last_3_games_avg, or assists_last_5_games_avg can help the model capture recent performance trends and provide a more accurate way of predicting how a player would perform.
4. We would also create new features like points per minute played, assists per minute played, and rebounds per minute played to truly capture the player's efficiency and impact instead of pure counts.
5. We would normalize features like player_height, player_weight, and the ratings to a standard scale (mean = 0, std = 1). This ensures that all features contribute equally to the model.
6. We also believe that deriving opponent team statistics such as defense rating can be useful in reflecting the influence of team dynamics on a player's individual performance.

## Prizepicks
Since PrizePicks lines change daily we will select a specific week to lock in the lines for our predictions:

1. We’ll pick a single week and specific players to make a prediction within the NBA season to focus on our predictions. By fixing this time period, we’ll ensure that our data and model align with the lines available during that week, providing a consistent basis for evaluating player performance.
3. Lock in Lines for Each Player: For this chosen week, we’ll record PrizePicks lines for points, rebounds, and assists for each player. These lines will serve as the target thresholds in our model, determining whether a player will go “over” or “under” in each category.
