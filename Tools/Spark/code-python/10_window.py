# # Window Functions

# Copyright © 2010–2018 Cloudera. All rights reserved.
# Not to be reproduced or shared without prior written 
# consent from Cloudera.


# ## Overview

# In this module we demonstrate how to create and apply window functions.


# ## Window Functions

# * Spark SQL supports the following window functions:
#   * `cume_dist`
#   * `dense_rank`
#   * `lag`
#   * `lead`
#   * `ntile`
#   * `percent_rank`
#   * `rank`
#   * `row_number`

# * Aggregate and window functions are applied `over` a window specification

# * A window specification consists of at least one of the following:
#   * Partitioning column
#   * Ordering column
#   * Row specification

# * Windows are experimental as of Spark 2.3.


# ## Setup

# Create a SparkSession:
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local").appName("window").getOrCreate()

# Read the enhanced ride data from HDFS:
rides = spark.read.parquet("./Tools/Spark/duocar/joined/")


# ## Example: Cumulative Count and Sum

# Create a simple DataFrame:
df = spark.range(10)
df.show()

# Create a simple window specification:
from pyspark.sql.window import Window
ws = Window.rowsBetween(Window.unboundedPreceding, Window.currentRow)
type(ws)

# Use the window specification to compute cumulative count and sum:
from pyspark.sql.functions import count, sum
df.select("id", count("id").over(ws).alias("cum_cnt"), sum("id").over(ws).alias("cum_sum")).show()

# **Tip:** Examine the default column name to gain additional insight (if you
# are SQL literate):
df.select("id", count("id").over(ws), sum("id").over(ws)).printSchema()


# ## Example: Compute average days between rides for each rider

# Create window specification:
ws = Window.partitionBy("rider_id").orderBy("date_time")

# Use the `lag` function to extract the date and time of the previous ride:
from pyspark.sql.functions import lag
rides2 = rides.withColumn("previous_ride", lag("date_time").over(ws))
rides2.select("rider_id", "date_time", "previous_ride").show(truncate=False)

# **Note:** A rider's first ride does not have a previous ride; therefore, the
# value is set to null.

# Compute the number of days between consecutive rides:
from pyspark.sql.functions import datediff
rides3 = rides2.withColumn("days_between_rides", datediff("date_time", "previous_ride"))
rides3.select("rider_id", "date_time", "previous_ride", "days_between_rides").show(truncate=False)

# Compute the average days between rides for each rider:
from pyspark.sql.functions import count, mean
rides4 = rides3 \
  .groupBy("rider_id") \
  .agg(count("*").alias("num_rides"), mean("days_between_rides").alias("mean_days_between_rides"))

# Compute top and bottom 10 riders:
rides4 \
  .where(rides4.mean_days_between_rides.isNotNull()) \
  .orderBy("mean_days_between_rides") \
  .show(10)

rides4 \
  .orderBy("mean_days_between_rides", ascending=False) \
  .show(10)

# **Question:** How can we make this analysis better?


# ## Exercises

# (1) What is the average time between rides for each driver?


# ## References

# [Spark Python API - pyspark.sql.Window
# class](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html?highlight=catalog#pyspark.sql.Window)

# [Spark Python API - pyspark.sql.WindowSpec
# class](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html?highlight=catalog#pyspark.sql.WindowSpec)


# ## Cleanup

# Stop the SparkSession:
spark.stop()
