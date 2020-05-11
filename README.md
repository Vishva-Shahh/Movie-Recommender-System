# Movie-Recommender-System
MovieLens is a dataset that is collected by the GroupLens Research Project at the University of
Minnesota and made available rating data sets from the MovieLens web site. Download and
unzip the MovieLens 100K Dataset (ml-100k.zip). http://grouplens.org/datasets/movielens/
u.data is the dataset for this assignment
The full dataset contains 100000 ratings by 943 users on 1682 items. Each user has rated at
least 20 movies. Users and items are numbered consecutively from 1. The data is randomly
ordered. The format is:
user_id<tab>item_id<tab>rating<tab>timestamp.
The time stamps are unix seconds since 1/1/1970 UTC

TODO List: 

1. Import the MovieLens dataset.
2. Build a recommendation model using Alternating Least Squares
3. Report the original performance (Mean Squared Error)
4. Try to improve the performance of the original model using cross validation and solve the
cold-start problem.
5. Report the improved performance after the step 4 and output top 10 movies for all the
users with the following format:
userID<\tab>itemID1,itemID2,itemID3 ...,itemID10
