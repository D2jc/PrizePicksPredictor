## Preprocessing Steps:

1. Drop Irrelevant Columns: There are many columns such as turnover, pf, and player_id that we would drop as they are not very important for predictions on a player's particular stats such as for how many points they make / rebounds they receive / assists they make. After combining the first and last name of players, we would drop the individual columns as well as the player_id.
2. We would prepare a time-series data for lagged versions of key stats (points, assists, rebounds) to capture the player’s performance trends over recent games. For example, features like on_hotstreak_pts, on_hotstreak_rebounds, or on_hotstreak_assists can help the model capture whether or not the player is playing better than their average performance from past games.
3. We would also create new features like points per minute played, assists per minute played, and rebounds per minute played to truly capture the player's efficiency and impact instead of pure counts.
4. We would normalize features like 'fgm', 'fga', 'oreb', 'dreb', 'ast', 'stl', 'blk' and 'pts' to a standard scale (mean = 0, std = 1). This ensures that all features contribute equally to the model.

## Prizepicks
Since PrizePicks lines change daily we will select a specific week to lock in the lines for our predictions:

1. We’ll pick a single week and specific players to make a prediction within the NBA season to focus on our predictions. By fixing this time period, we’ll ensure that our data and model align with the lines available during that week, providing a consistent basis for evaluating player performance.
3. Lock in Lines for Each Player: For this chosen week, we’ll record PrizePicks lines for points, rebounds, and assists for each player. These lines will serve as the target thresholds in our model, determining whether a player will go “over” or “under” in each category.

## Data Preprocessing
1. We converted the 'Date' column to accurately ensure a proper day, month, and year date setting in the column.
2. We combined the 'first_name' and 'last_name' to create a new column of 'player_name' so that when we merge the Prizepicks data, it can match up to the individual player name.
3. We ended up dropping the 'first_name', 'last_name', 'turnover', and 'pf' as for the first two, we have already combined those columns, and 'turnover' and 'pf' was discussed to be unhelpful in our model.
4. We checked for any missing data and found that there was none missing.
5. We merged the projections dataframe from Prizepicks with the data we have preprocessed so we had a dataframe that had 'player_name ', 'date', 'fgm', 'fga', 'fg_pct', 'fg3m', 'fg3a', 'fg3_pct', 'ftm', 'fta', 'ft_pct', 'oreb', 'dreb', 'reb', 'ast', 'stl', 'blk', 'pts', and 'line_score'.
6. We then created features 'on_hotstreak_pts, 'on_hotstreak_asts', and 'on_hotstreak_reb' where we computed the last 5 averages in points, assists, and rebounds for each player. Then, gave a binary value of either '1' to indicate that they are doing better than the average of their past 5 games, and a '0' if they are not.
7. We also created a feature called 'above_threshold' which is a binary label where if the player exceeded the PrizePicks line, they got a '1', and if they did not, the player had a '0'.
8. Lastly, we scaled our features 'fgm', 'fga', 'pts','fg3m','fg3a','fta','ftm', as we believe that these features are directly tied to player performance and ensures that no single feature dominates the model simply due to its scale.
9. We implemented a function to create sequences of player data for time-series modeling, where each sequence represents a sliding window of consecutive games for a given player.
10. Then, we split the data into training and testing sets while ensuring that sequences from the same player are kept intact within either set. By grouping sequences based on player names, 80% of each player’s data is allocated for training, and the remaining 20% is reserved for testing. We also converted the data into Pytorch tensors.

## Second Model

1. We created an initial random forest classifier with 75% train and 25% test split.
2. We tuned the hyperparameters using GridSearchCV, adjusting n_estimators, max_features, max_depth, and min_samples_split to achieve the maximum accuracy of 88% on the test data.
3. We plotted train and test accuracy for different settings for hyperparameters using matplotlib, which shows the hyperparameters that we chose as the best possible ones.

### Conclusion:

Following using a random forest classifier for our second model, we realized from this model that in order to get the minimum test error, we needed to overfit our training set a little bit. After optimizing our hyperparameters, we got the maximum accuracy at about 80%, which is what we previosuly capped out at. We increased the hidden layers in our random forest model and learned that it did a better job than the original model of determing hidden features and other intracies in our data. For our next model, it may be worth it to try approaches that are closer to the random forest approach in comparison to the RNN approach, as we found this works significantly better.
