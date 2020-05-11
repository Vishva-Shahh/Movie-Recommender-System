#!/usr/bin/env python
# coding: utf-8

# # Spark Installation

# In[ ]:


# installation 

from google.colab import drive
drive.mount('/content/drive')
get_ipython().system('apt-get install openjdk-8-jdk-headless -qq > /dev/null')
get_ipython().system('wget -q http://apache.mirrors.hoobly.com/spark/spark-2.4.5/spark-2.4.5-bin-hadoop2.7.tgz')
get_ipython().system('tar -xvf spark-2.4.5-bin-hadoop2.7.tgz')
get_ipython().system('pip install -q findspark')

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-2.4.5-bin-hadoop2.7"

import findspark
findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").getOrCreate()


# # Recommeder System for MovieLens dataset

# In[ ]:


#Import libraries
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.sql import Row


# In[ ]:


get_ipython().system('ls')


# # Step 1. Read data - Import the MovieLens Dataset

# In[23]:


#load ratings data from the MovieLens dataset, each row consisting of a user, a movie, a rating and a timestamp
lines = spark.read.text("/content/drive/Shared drives/Analytics for Big Data/HW4/ml-100k/u.data").rdd  
parts = lines.map(lambda row: row.value.split("\t"))
ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                     rating=float(p[2]), timestamp=float(p[3])))
ratings = spark.createDataFrame(ratingsRDD)
ratings.show()


# In[24]:


#Dropping Timestamp column
ratings = ratings.drop("timestamp")
ratings.show()


# # Split training and testing

# In[ ]:


(training, test) = ratings.randomSplit([0.8, 0.2])


# # Step 2. Build the recommendation model using ALS

# In[ ]:


als_original = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", nonnegative=True)
model = als_original.fit(training)


# # Step 3. Reporting the Original Performance

# In[27]:


# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Original Root-mean-square error = " + str(rmse))


# As we can see rmse is coming out to be nan. This is the cold start problem.
# 
# In this, Spark assigns NaN predictions during ALSModel. To solve this problem, spark allows dropping rows in the DataFrame of predictions that contain NaN values. It is done by setting coldStartStrategy parameter to "drop"

# # Step 4a. Solving the cold start problem

# In[28]:


als_new = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", nonnegative=True, coldStartStrategy= "drop")
model = als_new.fit(training)

# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))


# The RMSE is coming out to be 0.916, now trying to imrpove the performance using Cross Validation

# # Step 4b. Performance improvement using Cross Validation

# In[32]:


from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import ParamGridBuilder

model_new = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", nonnegative = True, coldStartStrategy="drop")

# Parameters for tuning
paramGrid = ParamGridBuilder()     .addGrid(model_new.regParam, [0.1, 0.01, 0.001])     .addGrid(model_new.rank, [5, 10, 15])     .build()

crossvalidation = CrossValidator(estimator = model_new,
                     estimatorParamMaps = paramGrid,
                     evaluator = evaluator,
                     numFolds=10)

#Using the Best Model
model_cv = crossvalidation.fit(training).bestModel

#Evaluate and print the predictions
print("RMSE value after solving cold start problem is: ", evaluator.evaluate(model_cv.transform(test)))


# As we can see, even after CV there isn't much improvement.

# # Step 5. Top 10 movies for all the users 

# In[33]:


recommendations = model_cv.recommendForAllUsers(10)
recommendations.show()


# In[ ]:


import pandas as pd

recommendations = recommendations.toPandas()


# In[36]:


#Initialize lists that will be used for converting to dataframe
list_users = []
list_recs = []

#Iterate over the whole data set
for i in range(len(recommendations)):
  #Add userId to user list
  list_users.append(recommendations.iloc[i,0])
  
  #Initialize a string for storing a given user's recommendations
  user_recs = "" 

  #Iterate over all recommendations and pick the movieIds
  for item in recommendations.iloc[i,1]:
    user_recs = user_recs + ", " + str(item.asDict()["movieId"])

  list_recs.append(user_recs[2:])

recommendations_df = pd.DataFrame(data = zip(list_users, list_recs), columns=["user", "recommendations"])
recommendations_df.head()


# In[ ]:


#Write to a text file
with open("recommendations.txt", "w") as f:
  f.write("userId\trecommendations\n")
  for i in range(len(recommendations_df)):
    f.write(str(recommendations_df.iloc[i,0]) + "\t" + recommendations_df.iloc[i,1] + "\n")

