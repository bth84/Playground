# # Transforming DataFrames - Supplement

# Copyright © 2010–2018 Cloudera. All rights reserved.
# Not to be reproduced or shared without prior written 
# consent from Cloudera.


# ## Contents

# * Selecting columns using Python list comprehension
# * Replacing valid values with null values


# ## Setup

# Create a SparkSession:
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local").appName("dataframes_supplement").getOrCreate()

# Read the raw data from HDFS:

rides = spark.read.csv("/duocar/raw/rides/", header=True, inferSchema=True)
drivers = spark.read.csv("/duocar/raw/drivers/", header=True, inferSchema=True)
riders = spark.read.csv("/duocar/raw/riders/", header=True, inferSchema=True)


# ## Selecting columns using Python list comprehension

# Select the first three columns:
cols = riders.columns[0:3]
riders.select(cols).show(5)

# Select a subset of columns:
cols = [riders.columns[i] for i in [0, 3, 4]]
riders.select(cols).show(5)

# Select all string columns:
cols = [x[0] for x in riders.dtypes if x[1] == 'string']
riders.select(cols).show(5)


# ## Replacing valid values with null values

# Consider the following DataFrame:
df = spark.createDataFrame([(-9, ), (0, ), (1, ), (-9, ), (1, ), (0, )], ["value"])
df.show()

# Suppose that the value `-9` represents a missing value.  Then use the
# `replace` method to change these values to null values:
df.replace(-9, None).show()


# ## Cleanup

# Stop the SparkSession:
spark.stop()
