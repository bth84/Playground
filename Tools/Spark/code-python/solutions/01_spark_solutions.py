# # Running a Spark Application from CDSW - Solutions

# Copyright © 2010–2018 Cloudera. All rights reserved.
# Not to be reproduced or shared without prior written 
# consent from Cloudera.


# ## Exercises

# (1) Create a new `SparkSession` and configure Spark to run locally with one
# thread.

from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local").appName("spark_solutions").getOrCreate()

# The following syntax is equivalent:
spark = SparkSession.builder.master("local[1]").appName("spark_solutions").getOrCreate()

# (2) Read the raw driver data from HDFS.

drivers = spark.read.csv("/duocar/raw/drivers/", header=True, inferSchema=True)

# (3) Examine the schema of the drivers DataFrame.

drivers.printSchema()

# (4) Count the number of rows of the drivers DataFrame.

drivers.count()

# (5) Examine a few rows of the drivers DataFrame.

drivers.head(5)

# (6) **Bonus:** Repeat exercises (2)-(5) with the raw rider data.

riders = spark.read.csv("/duocar/raw/riders/", header=True, inferSchema=True)
riders.printSchema()
riders.count()
riders.head(5)

# (7) **Bonus:** Repeat exercises (2)-(5) with the raw ride review data.
# **Hint:** Verify the file format before reading the data.

reviews = spark.read.csv("/duocar/raw/ride_reviews/", sep="\t", header=False, inferSchema=True)
reviews.printSchema()
reviews.count()
reviews.show(5)

# (8) Stop the SparkSession.

spark.stop()
