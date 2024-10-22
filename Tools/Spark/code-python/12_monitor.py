# # Monitoring, Tuning, and Configuring Spark Applications

# Copyright © 2010–2018 Cloudera. All rights reserved.
# Not to be reproduced or shared without prior written 
# consent from Cloudera.


# ## Monitoring Spark Applications

# We monitor a *Spark application* via the *Spark UI*.  The Spark UI is not
# available until we start a Spark application.  We start a Spark application
# by creating a `SparkSession` instance:
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local").appName("monitor").getOrCreate()

# In CDSW, the Spark UI is available via the grid icon in the upper right.


# ### Example 1: Partitioning DataFrames

# Read the ride data as a text file:
rides = spark.read.text("./Tools/Spark/duocar/raw/rides/")

# View the Spark UI and note that this operation does not generate a *Spark
# job*.

# Get the number of partitions:
rides.rdd.getNumPartitions()

# Note that we are accessing the [Resilient Distributed Dataset
# (RDD)](http://spark.apache.org/docs/latest/rdd-programming-guide.html#resilient-distributed-datasets-rdds)
# underlying the DataFrame.

# Print the schema:
rides.printSchema()

# Note that this operation also does not generate a job.

# Print a few rows:
rides.show(5)

# Note that `show` is an *action* and generates a job with one *stage* and one
# *task*.  Show is actually a *partial action* since Spark does not have to
# read all the data to print out a few rows.

# `count` is also an action:
print(rides.count())

# Note that `count` generates a job with two stages and three tasks.  The first
# stage consists of two parallel tasks that count the number of rows in each
# partition.  The second stage consists of one task that adds these partial sum
# to compute the final count.

# Save the DataFrame to HDFS:
rides.write.mode("overwrite").text("data/monitor/")

# Note that each partition is written to a separate file.

# Let us repartition the DataFrame into six partitions:
rides6 = rides.repartition(6)

# Count the number of rows:
print(rides6.count())

# Note the number of stages and tasks.

# The `coalesce` method is a more efficient way to reduce the number of
# partitions:
rides3 = rides6.coalesce(3)
print(rides3.count())

# Note that in this particular case, Spark ignored the repartition and
# subsequent coalesce as it found a more efficient way to compute the result.

# Remove the temporary file:
!hdfs dfs -rm -r data/monitor/


# ### Example 2: Persisting DataFrames

# Read the ride data as a (comma) delimited text file:
rides = spark.read.csv("./Tools/Spark/duocar/raw/rides", header=True, inferSchema=True)

# Note that Spark has run three exploratory jobs to read the header and infer
# the schema.

# Duplicate the ride data to make it bigger:
big_rides = spark.range(100).crossJoin(rides)

# Print the number of partitions:
big_rides.rdd.getNumPartitions()

# Chain together a more elaborate set of transformations:
from pyspark.sql.functions import count, mean, stddev
result = big_rides \
  .groupby("rider_id") \
  .agg(count("*"), count("distance"), mean("distance"), stddev("distance")) \
  .orderBy("count(distance)", ascending=False)

# Spark determines the appropriate number of partitions:
result.rdd.getNumPartitions()

# Persist the DataFrame in memory:
result.persist()

# Review the **Storage** tab in the Spark UI.  Spark does not persist the DataFrame
# until it is actually computed.

# Run an action to compute the DataFrame:
%time print(result.count())

# Note that the DataFrame is now listed under the **Storage** tab in the Spark UI.

# Run the action again:
%time print(result.count())

# Note that it runs noticeably faster since the result is already in memory:

# Free up memory:
result.unpersist()

# Stop the SparkSession:
spark.stop()

# This also stops the Spark Application and disables the Spark UI.


# ## Configuring the Spark Environment

# We have been creating a SparkSession using the following syntax:
#```python
# spark = SparkSession.builder \
#   .master("local") \
#   .appName("config") \
#   .getOrCreate()
#```

# This is actually a special case of the following more general syntax:
#```python
# spark = SparkSession.builder \
#   .config("spark.master", "local") \
#   .config("spark.app.name", "config") \
#   .getOrCreate()
#```

# We can configure additional environment settings:
spark = SparkSession.builder \
  .config("spark.master", "local") \
  .config("spark.app.name", "config") \
  .config("spark.driver.memory", "2g") \
  .getOrCreate()

# We can query a configuration property using the following syntax:
spark.conf.get("spark.driver.memory")

# We can view other settings under the **Environment** tab of the Spark UI.

# You can set configuration properties in the `spark-defaults.conf` file:
!cat spark-defaults.conf


# ## Configuring Spark Logging

# You can increase or decrease the log level for a Spark Application using
# `setLogLevel` method for the underlying SparkConext:
spark.sparkContext.setLogLevel?

# Increase the logging to the `INFO` level:
spark.sparkContext.setLogLevel("INFO")

# Load some data:
rides = spark.read.csv("/duocar/raw/rides", header=True, inferSchema=True)

# Reset the log level to the default for CDSW:
spark.sparkContext.setLogLevel("ERROR")

# Stop the SparkSession (and the Spark application):
spark.stop()


## References

# [Monitoring Spark
# Applications](https://www.cloudera.com/documentation/enterprise/latest/topics/operation_spark_applications.html#spark_monitoring)

# [Tuning Spark
# Applications](https://www.cloudera.com/documentation/enterprise/latest/topics/admin_spark_tuning1.html)

# [Configuring the Cloudera Distribution of Apache Spark
# 2](https://www.cloudera.com/documentation/data-science-workbench/latest/topics/cdsw_spark_configuration.html)
