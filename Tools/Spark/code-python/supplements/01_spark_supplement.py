# # Running a Spark Application from CDSW - Supplement

# Copyright © 2010–2018 Cloudera. All rights reserved.
# Not to be reproduced or shared without prior written 
# consent from Cloudera.


# ## Contents

# * Accessing core Spark
# * Working with `Row` objects


# ## Setup

# Create a SparkSession:
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local").appName("spark_supplement").getOrCreate()

# Read the raw ride data from HDFS:
rides = spark.read.csv("/duocar/raw/rides/", header=True, inferSchema=True)


# ## Accessing core Spark

# The `SparkSession` class provides access to Spark SQL and the DataFrame API.
# The `SparkContext` class provides access to core Spark and the *resilient
# distributed dataset* (RDD) API.  Creating a `SparkSession` object also
# creates an underlying `SparkContext` object:
spark.sparkContext

# **Note:** The Spark UI link does not work in CDSW.

# Use the `parallelize` method of the `SparkContext` instance to create an RDD:
rdd = spark.sparkContext.parallelize([3.1416, 2.7183, 1.6180, 0.5772])

# Use the `count` method of the `RDD` instance to count the number of elements in
# the RDD:
rdd.count()

# Use the `take` method to return elements of the RDD as a Python list:
rdd.take(5)

# Stopping a `SparkSession` also stops the underlying `SparkContext`.


# ## Working with `Row` objects

# The `head` and `take` methods return a Python list of Spark `Row` objects:
rows = rides.head(5)
rows

# Use standard Python index notation to access elements of the list:
rows[0]
rows[-1]
rows[1:-2]

# Use either index, key, or dot notation to access elements of a `Row` object:
rows[0][3]
rows[0]['date_time']
rows[0].date_time

# Use the `asDict` method to convert a `Row` object to a Python dictionary:
rows[0].asDict()


# ## Cleanup

# Stop the SparkSession:
spark.stop()
