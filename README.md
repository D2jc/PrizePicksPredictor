> TODOs right now: clean up repo and structure, add figures around, review methods section, do results section, modify and improve discussion section, do conclusion section

# PrizePicksPredictor

Minchan Kim, David Moon, Mihir Joshi, Rohil Kadekar, Jason Kim

---------

## Introduction

The realm of sports analytics has witnessed a significant transformation with the advent of advanced machine learning techniques. These methods, when applied to vast datasets, have the potential to uncover hidden patterns and insights that can revolutionize the way we perceive and predict athletic performance. In this project, we delved into the realm of daily fantasy sports, specifically focusing on PrizePicks, a platform that allows users to bet and make predictions on individual player performances. Our objective was to develop a robust machine learning model capable of accurately predicting whether a player will outperform or underperform their projected performance, thereby providing a valuable edge to fantasy sports enthusiasts.

To achieve this goal, we leveraged a comprehensive dataset sourced from PrizePicks, encompassing a wide range of player statistics and historical performance data. By employing a combination of feature engineering techniques and state-of-the-art machine learning algorithms, we aimed to construct a model that can effectively capture the intricate nuances of player performance and factors that influence their outcomes.

More specifically, in this project, we aimed to develop models to predict whether NBA players will score over or under the point, rebound, and assist lines set by PrizePicks. By leveraging a dataset of historical NBA player statistics from the official NBA website, we analyzed a range of variables such as points per game, shooting percentages, minutes played, and much more. Our focus in this project was to identify patterns and trends that influence the player in a specific match-up, allowing us to make a more accurate prediction than a human can. Our approach was optimized in predicting player outcomes by experimenting with various variables to increase its accuracy. This study provides a data-driven strategy for enhancing player prop betting predictions.

For the binary classification task at hand of predicting underperformance or overperformance of projections , we developed and tuned two different models: a Long Short-Term Memory (LSTM) recurrent neural network (RNN) and a random forest classifier. The LSTM model is well-suited for capturing temporal dependencies in time series data, such as player performance trends over time. The random forest classifier, on the other hand, is a powerful ensemble method that can handle complex decision-making processes.

The implications of this project extend beyond the realm of fantasy sports. A successful model could potentially aid sports bettors, coaches, and even front-office personnel in making informed decisions. Additionally, this project contributes to the broader field of sports analytics by demonstrating the power of machine learning in predicting individual player performance, a challenge that has long intrigued a variety of stakeholders alike.

---------

## Methods:

> We should remove anything related to accuracy --> move to Results section. Also fix the links to notebooks to be in the main branch.

### Data Exploration:
Run the Jupyter Notebook called AnalysisVis.ipynb which can be found in MS2. This notebook has our data exploration and many different visualizations of the data set that we are using. Most of the visualizations that we have created revolve around the player's performances.

### Preprocessing Steps:

1. Drop Irrelevant Columns: There are many columns such as turnover, pf, and player_id that we would drop as they are not very important for predictions on a player's particular stats such as for how many points they make / rebounds they receive / assists they make. After combining the first and last name of players, we would drop the individual columns as well as the player_id.
2. We would prepare a time-series data for lagged versions of key stats (points, assists, rebounds) to capture the player’s performance trends over recent games. For example, features like on_hotstreak_pts, on_hotstreak_rebounds, or on_hotstreak_assists can help the model capture whether or not the player is playing better than their average performance from past games.
3. We would also create new features like points per minute played, assists per minute played, and rebounds per minute played to truly capture the player's efficiency and impact instead of pure counts.
4. We would normalize features like 'fgm', 'fga', 'oreb', 'dreb', 'ast', 'stl', 'blk' and 'pts' to a standard scale (mean = 0, std = 1). This ensures that all features contribute equally to the model.

### Prizepicks Projections
Since PrizePicks lines change daily we selected a specific week to lock in the lines for our predictions:

1. We picked a single week and specific players to make a prediction within the NBA season to focus on our predictions. By fixing this time period, we ensured that our data and model align with the lines available during that week, providing a consistent basis for evaluating player performance.
3. Lock in Lines for Each Player: For this chosen week, we recorded PrizePicks lines for points, rebounds, and assists for each player. These lines will serve as the target thresholds in our model, determining whether a player will go “over” or “under” in each category.

### [Data Preprocessing and LSTM RNN (First Model)](https://github.com/D2jc/PrizePicksPredictor/tree/main/MS3/Preprocessing/Data%20Processing%20and%20First%20Model.ipynb)

> The notebook for Data Preprocessing and First Model can be found at []

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
11. Following the splitting of our data and conversion, we trained and tested our data on a baseline Long-short Term Model.

### [Random Forest Classifier (Second Model)](https://github.com/D2jc/PrizePicksPredictor/tree/main/MS4/Second_Model.ipynb)

> The notebook for Milestone 4 (Second Model) can be found at []

1. We completed pre-processing of the data in the same way as in Milestone 3, dropping unnecessary columns, standardized some of the features, aggregated features to extract new features like hot streaks. We used PrizePicks targets as the y variable and the cleaned data features as our x variables.
2. We created an initial random forest classifier with 75% train and 25% test split.
3. We tuned the hyperparameters using GridSearchCV, adjusting n_estimators, max_features, max_depth, and min_samples_split to achieve the maximum accuracy of 88% on the test data.
4. We plotted train and test accuracy for different settings for hyperparameters using matplotlib, which shows the hyperparameters that we chose as the best possible ones.

---------

## Results

> FROM INSTRUCTIONS: This will include the results from the methods listed above (C). You will have figures here about your results as well. No exploration of results is done here. This is mainly just a summary of your results. The sub-sections will be the same as the sections in your methods section.

### Data Exploration
- what findings?

### Data Preprocessing
- how much cleaner is the data? before and after states?

### LSTM RNN Model
- accuracy and other metrics?

### Random Forest Classifier
- accuracy, precision, recall, fp and fn rates, and other metrics?

> We are supposed to follow the same structure as the methods section but I don't think it makes sense for us to have PrizePicks Projections as a section in Results. Thoughts?

---------

## Discussion

> FROM INSTRUCTIONS: This is where you will discuss the why, and your interpretation and your though process from beginning to end. This will mimic the sections you have created in your methods section as well as new sections you feel you need to create. You can also discuss how believable your results are at each step. You can discuss any short comings. It's ok to criticize as this shows your intellectual merit, as to how you are thinking about things scientifically and how you are able to correctly scrutinize things and find short comings. In science we never really find the perfect solution, especially since we know something will probably come up int he future (i.e. donkeys) and mess everything up. If you do it's probably a unicorn or the data and model you chose are just perfect for each other!

We believe that the LSTM model performed well given the context of the problem with an accuracy rate of 67%, which is better than blindly guessing (50% chance). We believe that our model can be improved with different feature engineering such as giving the actual averages of our players from different time periods or hyperparameter tuning our model where we experiment with different LTSM options such as hidden layer sizes, learning rates, and sequence lengths. Lastly, we can try different models such as Attention Long-short Term Model and Stacked Long-short Term Model to see how they would be compared to our baseline LSTM.

Following using a random forest classifier for our second model, we realized from this model that in order to get the minimum test error, we needed to overfit our training set a little bit. After optimizing our hyperparameters, we got the maximum accuracy at about 80%, which is what we previosuly capped out at. We increased the hidden layers in our random forest model and learned that it did a better job than the original model of determing hidden features and other intracies in our data. For our next model, it may be worth it to try approaches that are closer to the random forest approach in comparison to the RNN approach, as we found this works significantly better.

---------

## Conclusion

> FROM INSTRUCTIONS: This is where you do a mind dump on your opinions and possible future directions. Basically what you wish you could have done differently. Here you close with final thoughts.

TODO

---------

## Statement of Collaboration:

- Minchan Kim, Group Member: Contributed to code and report
- David Moon, Group Member: Contributed to code and report
- Rohil Kadekar, Group Member: Contributed to code and report
- Jason Kim, Group Member: Contributed to code and report
- Mihir Joshi, Group Member: Contributed to code and report

No members of the team consulted any non-members for any aspect of the project. No members of the team utilized any resources other than publicly available documentation in the development and writing processes of all milestones of the project. As this project is publicly available and completed for educational purposes, other individuals are permitted to responsibly use and build upon our work for non-commercial purposes.
