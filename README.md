> TODOs right now: add any more figures(?) and possibly get rid of some figures in methods->data exploration, add data exploration to methods section, do results -> data exploration/preprocessing and discussion section

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

> Maybe only include one of the figures and get rid of the others

### Points Scored vs. Field Goals Attempted ###

![Points Scored vs Field Goals Attempted](Figures/Points_scored_vs_field_goals_attempted_figure.png)

#### Description: ####
This scatterplot examines the relationship between the number of field goals attempted (FGA) and points scored. Data points are sized and colored based on FG%, providing additional insight into shooting efficiency. This figure explores scoring tendencies and efficiency across players, emphasizing how attempts correlate with outcomes and potentially other features.
#### Key Insights: ####
While more field goal attempts generally correspond to more points scored, the efficiency (FG%) varies, highlighting the importance of this metric in predictive models.

### Rebounds Distribution ###

![Rebounds Distribution](Figures/Rebounds_distribution_figure.png)

#### Description: ####
This boxplot shows the distribution of offensive rebounds, defensive rebounds, and total rebounds among players. The data showcases variability across these categories, providing an overview of rebounding patterns, which are critical for feature selection and understanding player contributions.
#### Key Insights: ####
Defensive rebounds are generally higher than offensive rebounds, but total rebounds show significant variability, reflecting differences in player roles and game dynamics.

### Field Goal Percentage vs Points Scored ###

![Field Goal Percentage vs Points Scored](Figures/Field_goal_percentage_vs_points_scored_figure.png)

#### Description: ####
This scatterplot visualizes the relationship between field goal percentage (FG%) and the total points scored in a game. Each data point is color-coded and sized based on FG%, offering a multidimensional perspective of the relationship. Highlights trends in player efficiency and scoring performance, helping to identify patterns or anomalies that may influence predictive modeling.
#### Key Insights: ####
As expected, players with higher FG% generally score more points. However, when FG% becomes exceptionally high, the limited number of attempts can result in a high FG% but a lower overall point total. Additionally, outliers highlight variability in player contributions, suggesting that factors beyond shooting efficiency influence scoring performance.


### Preprocessing Steps:

1. Drop Irrelevant Columns: There are some columns such as turnover, pf, and player_id that we would drop as they are not very important for predictions on a player's particular stats such as for how many points they make / rebounds they receive / assists they make. After combining the first and last name of players, we would drop the individual columns as well as the player_id.
2. We would prepare a time-series data for lagged versions of key stats (points, assists, rebounds) to capture the player’s performance trends over recent games. For example, features like on_hotstreak_pts, on_hotstreak_rebounds, or on_hotstreak_assists.
3. We would also create new features like points per minute played, assists per minute played, and rebounds per minute played.
4. We would normalize features like 'fgm', 'fga', 'oreb', 'dreb', 'ast', 'stl', 'blk' and 'pts' to a standard scale (mean = 0, std = 1).

#### Prizepicks Projections
Since PrizePicks lines change daily we selected a specific week to lock in the lines for our predictions:

1. We picked a single week and specific players to make a prediction within the NBA season to focus on our predictions. By fixing this time period, we ensured that our data and model align with the lines available during that week, providing a consistent basis for evaluating player performance.
3. Lock in Lines for Each Player: For this chosen week, we recorded PrizePicks lines for points, rebounds, and assists for each player. These lines will serve as the target thresholds in our model, determining whether a player will go “over” or “under” in each category.

### [Data Preprocessing and LSTM RNN (First Model)](https://github.com/D2jc/PrizePicksPredictor/blob/main/Data%20Processing%20and%20First%20Model.ipynb)

The notebook for the First Model can be found at ```Data Processing and First Model.ipynb```

1. We converted the 'Date' column to accurately ensure a proper day, month, and year date setting in the column.
2. We combined the 'first_name' and 'last_name' to create a new column of 'player_name' so that when we merge the Prizepicks data, it can match up to the individual player name.
3. We ended up dropping the 'first_name', 'last_name', 'turnover', and 'pf' as for the first two
4. We checked for any missing data and found that there was none missing.
5. We merged the projections dataframe from Prizepicks with the data we have preprocessed so we had a dataframe that had 'player_name ', 'date', 'fgm', 'fga', 'fg_pct', 'fg3m', 'fg3a', 'fg3_pct', 'ftm', 'fta', 'ft_pct', 'oreb', 'dreb', 'reb', 'ast', 'stl', 'blk', 'pts', and 'line_score'.
6. We then created features 'on_hotstreak_pts, 'on_hotstreak_asts', and 'on_hotstreak_reb' where we computed the last 5 averages in points, assists, and rebounds for each player. Then, gave a binary value of either '1' to indicate that they are doing better than the average of their past 5 games, and a '0' if they are not.
7. We also created a feature called 'above_threshold' which is a binary label where if the player exceeded the PrizePicks line, they got a '1', and if they did not, the player had a '0'.
8. Lastly, we scaled our features 'fgm', 'fga', 'pts','fg3m','fg3a','fta','ftm', as we believe that these features are directly tied to player performance and ensures that no single feature dominates the model simply due to its scale.
9. We implemented a function to create sequences of player data for time-series modeling, where each sequence represents a sliding window of consecutive games for a given player.
10. Then, we split the data into training and testing sets while ensuring that sequences from the same player are kept intact within either set. By grouping sequences based on player names, 80% of each player’s data is allocated for training, and the remaining 20% is reserved for testing. We also converted the data into Pytorch tensors.
11. Following the splitting of our data and conversion, we trained and tested our data on a baseline Long-short Term Model.

### [Random Forest Classifier (Second Model)](https://github.com/D2jc/PrizePicksPredictor/blob/main/Second_Model.ipynb)

The notebook for Milestone 4 (Second Model) can be found at ```Second_Model.ipynb```

1. We completed pre-processing of the data in the same way as in Milestone 3, dropping unnecessary columns, standardized some of the features, aggregated features to extract new features like hot streaks. We used PrizePicks targets as the y variable and the cleaned data features as our x variables.
2. We created an initial random forest classifier with 75% train and 25% test split.
3. We plotted train and test accuracy for different settings for hyperparameters using matplotlib, which shows the hyperparameters that we chose as the best possible ones.

---------

## Results

> FROM INSTRUCTIONS: This will include the results from the methods listed above (C). You will have figures here about your results as well. No exploration of results is done here. This is mainly just a summary of your results. The sub-sections will be the same as the sections in your methods section.

### Data Exploration
- what findings?

### Data Preprocessing
- , we have already combined those columns, and 'turnover' and 'pf' was discussed to be unhelpful in our model.
- 
- how much cleaner is the data? before and after states?
- can help the model capture whether or not the player is playing better than their average performance from past games.
-  to truly capture the player's efficiency and impact instead of pure counts.
-  This ensures that all features contribute equally to the model.

### LSTM RNN Model
Model Performance:
- Training Accuracy: Reached 68.72% by the 10th epoch, showing the model effectively learned from the training data.
- Test Accuracy: Peaked at 67.49%, demonstrating reasonable generalization to unseen data.
Loss Reduction:
- The loss decreased from 17.34 to 14.93 over the course of 10 epochs, indicating consistent improvements in model predictions.

![Training and test accuracy over epochs](Figures/Training_and_test_accuracy_over_epochs.png)
  
Interpretation:
- The model performed better than random guessing (50%) for binary classification.
- The 67.49% test accuracy suggests room for improvement in model design or feature engineering.

### Random Forest Classifier
Model Performance:
- Train Accuracy: Reached 99%, demonstrating the model effectively captured the patterns in the training data.
- Test Accuracy: Achieved 88%, significantly better than the baseline (50%), indicating strong predictive power.
Error Analysis:
- False Positive Rate: 4%, reflecting the model's tendency to minimize false alarms.
- False Negative Rate: 8%, indicating some room for improvement in identifying true positives.

![Accuracy of random forest classifier during training](Figures/Accuracy_of_random_forest_classifier_during_training.png)
  
Interpretation:
- The Random Forest Classifier outperformed the first model (LSTM), with a substantial increase in test accuracy.
- The balance between FPR and FNR suggests the model is more conservative in making positive predictions.

> We are supposed to follow the same structure as the methods section but I don't think it makes sense for us to have PrizePicks Projections as a section in Results. Thoughts?

---------

## Discussion

> FROM INSTRUCTIONS: This is where you will discuss the why, and your interpretation and your though process from beginning to end. This will mimic the sections you have created in your methods section as well as new sections you feel you need to create. You can also discuss how believable your results are at each step. You can discuss any short comings. It's ok to criticize as this shows your intellectual merit, as to how you are thinking about things scientifically and how you are able to correctly scrutinize things and find short comings. In science we never really find the perfect solution, especially since we know something will probably come up int he future (i.e. donkeys) and mess everything up. If you do it's probably a unicorn or the data and model you chose are just perfect for each other!



We believe that the LSTM model performed well given the context of the problem with an accuracy rate of 67%, which is better than blindly guessing (50% chance). We believe that our model can be improved with different feature engineering such as giving the actual averages of our players from different time periods or hyperparameter tuning our model where we experiment with different LTSM options such as hidden layer sizes, learning rates, and sequence lengths. Lastly, we can try different models such as Attention Long-short Term Model and Stacked Long-short Term Model to see how they would be compared to our baseline LSTM.

Following using a random forest classifier for our second model, we realized from this model that in order to get the minimum test error, we needed to overfit our training set a little bit. After optimizing our hyperparameters, we got the maximum accuracy at about 80%, which is what we previosuly capped out at. We increased the hidden layers in our random forest model and learned that it did a better job than the original model of determing hidden features and other intracies in our data. For our next model, it may be worth it to try approaches that are closer to the random forest approach in comparison to the RNN approach, as we found this works significantly better.

---------

## Conclusion

This project has been very hectic with being able to use an API for using our data, preprocessing the data, and creating models that can somewhat have a better guess than random choice. We approached this challenge with two distinct machine learning algorithms, being Long short-term memory (LSTM) Neural Network, and a Random Forest Classifier. Though our basic LSTM NN model had struggle in correctly predicting our player projections, we believed that it was able to perform better than a blind guess, which was our intention for the entire project. In our second model, the Random Forest Classifier, we have seen major improvement in accuracy performance. While the models we implemented were pretty effective in terms of the goal of our project, looking back, we believe that we could have explored more specialized architectures such as Attention-based LSTMs or Gradient Boosting Machines. Additionally, incorporating a wider range of features, creating new features, and employing advanced ensemble techniques could have further enhanced the predictive performance and robustness of our models. Looking ahead, there a ton of more work to be done in regards to this project. First, using more external data sources, such as advanced player stats or play-by-play data, could provide better contextual insights for each prediction, especially when a player plays against a specific team. Additionally, changing from a fixed PrizePicks lines to dynamic predictions that update in real time as lines shift throughout the day could also create a more robust and adaptable framework. Towards the end of our project, being able to create a pipeline that streamlines data preprocessing, feature engineering, and model evaluation, into an automated end-to-end system would significantly improve efficiency and scalability. Finally, the development of a user-friendly interface to display predictions could make this tool accessible and engaging to the public.

---------

## Statement of Collaboration:

- Minchan Kim, Group Member: Contributed to the code and writing the report.
- David Moon, Group Member: Contributed to the code and writing the report.
- Rohil Kadekar, Group Member: Contributed to the code and writing the report.
- Jason Kim, Group Member: Contributed to the code and writing the report.
- Mihir Joshi, Group Member: Contributed to the code and writing the report.

No members of the team consulted any non-members for any aspect of the project. No members of the team utilized any resources other than publicly available documentation in the development and writing processes of all milestones of the project. As this project is publicly available and completed for educational purposes, other individuals are permitted to responsibly use and build upon our work for non-commercial purposes.
