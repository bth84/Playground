# # Fitting and Evaluating Recommendation Systems

# Copyright © 2010–2018 Cloudera. All rights reserved.
# Not to be reproduced or shared without prior written 
# consent from Cloudera.


# ## Introduction

# * The ALS algorithm takes a DataFrame with user, item, and rating columns as input.


# ## Scenario


# ## Setup

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract


# ## Create a SparkSession

spark = SparkSession.builder.master("local").appName("recommend").getOrCreate()


# ## Read the data

# Read the Apache access logs from Earcloud:
access_logs = spark.read.text("/duocar/earcloud/apache_logs/")
access_logs.printSchema()
access_logs.head(5)
access_logs.count()


# ## Prepare the data for modeling

# Create user, artist, and playcount columns:
pattern = "^.*artist=(.*)&playcount=(.*) HTTP.*USER=(.*)\".*"
playcounts = access_logs.filter(col("value").contains("/listen?")) \
  .withColumn("user", regexp_extract("value", pattern, 3).cast("integer")) \
  .withColumn("artist", regexp_extract("value", pattern, 1).cast("integer")) \
  .withColumn("playcount", regexp_extract("value", pattern, 2).cast("integer")) \
  .persist()
playcounts.head(20)

# Note that the playcount column includes some negative, binary values:
playcounts.filter("playcount < 0").show()

# Fix the playcount column:
from pyspark.sql.functions import when, abs, conv
playcounts_fixed = playcounts.withColumn("playcount_fixed", when(col("playcount") < 0, conv(abs(col("playcount")), 2, 10).cast("integer")).otherwise(col("playcount")))
playcounts_fixed.printSchema()                            
playcounts_fixed.filter("playcount < 0").select("playcount", "playcount_fixed").show()

# **Note:** The `conv()` function returns a string.

# Select modeling data:
recommendation_data = playcounts_fixed \
    .select("user", "artist", "playcount_fixed") \
    .withColumnRenamed("playcount_fixed", "playcount")
recommendation_data.show()

# Save modeling data:
recommendation_data.write.parquet("data/recommendation_data/", mode="overwrite")


# ## Create train and test datasets

(train, test) = recommendation_data.randomSplit(weights=[0.75, 0.25], seed=12345)


# ## Specify and fit an ALS model

from pyspark.ml.recommendation import ALS
als = ALS(userCol="user", itemCol="artist", ratingCol="playcount", implicitPrefs=True, seed=23456)
print(als.explainParams())
als_model = als.fit(train)


# ## Examine the ALS model

als_model.userFactors.head(5)
als_model.itemFactors.head(5)

# **Note:** Some artists are not represented in the training data:
als_model.userFactors.count()
als_model.itemFactors.count()


# ## Apply the model

test_with_predictions = als_model.transform(test)
test_with_predictions.printSchema()
test_with_predictions.sort("user").show()


# ## Evaluate the model


# ## Generate recommendations

# Recommend the top $n$ items for each user:
als_model.recommendForAllUsers(5).sort("user").head(5)

# Recommend the top $n$ users for each item:
als_model.recommendForAllItems(5).sort("artist").head(5)

# Additional functionality:
als_model.recommendForUserSubset?
als_model.recommendForItemSubset?


# ## Cleanup

# Stop the SparkSession:
spark.stop()
