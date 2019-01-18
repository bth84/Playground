# # Exploring DataFrames - Solutions

# Copyright © 2010–2018 Cloudera. All rights reserved.
# Not to be reproduced or shared without prior written 
# consent from Cloudera.


# ## Setup

# Import useful packages, modules, classes, and functions:
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create a SparkSession:
spark = SparkSession.builder.master("local").appName("explore_solutions").getOrCreate()

# Load the enhanced ride data from HDFS:
rides_sdf = spark.read.parquet("/duocar/joined_all")

# Create a random sample and load it into a pandas DataFrame:
rides_pdf = rides_sdf.sample(withReplacement=False, fraction=0.01, seed=12345).toPandas()


# ## Exercises

# (1) Look for variables that might help us predict ride duration.

# Does the ride distance depend on the day of the week?

# Produce a summary table:
result1 = rides_sdf \
  .groupBy(F.dayofweek("date_time").alias("day_of_week")) \
  .agg(F.count("distance"), F.mean("distance")) \
  .orderBy("day_of_week")
result1.show()

# Plot the summarized data:
result1_pdf = result1.toPandas()
sns.barplot(x="day_of_week", y="avg(distance)", data=result1_pdf)

# (2) Look for variables that might help us predict ride rating.

# Do elite vehicles get higher ratings?

# Produce a summary table:
result2 = rides_sdf \
  .groupBy("vehicle_elite") \
  .agg(F.count("star_rating"), F.mean("star_rating")) \
  .orderBy("vehicle_elite")
result2.show()

# Plot the summarized data:
result2_pdf = result2.toPandas()
sns.barplot(x="vehicle_elite", y="avg(star_rating)", data=result2_pdf)


# ## Cleanup

# Stop the SparkSession:
spark.stop()
