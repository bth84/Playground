# # Reading and Writing Data - Solutions

# Copyright © 2010–2018 Cloudera. All rights reserved.
# Not to be reproduced or shared without prior written 
# consent from Cloudera.


# ## Setup

# Create a SparkSession:
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local").appName("read_solutions").getOrCreate() 

# Read the raw rider data from HDFS:
riders = spark.read.csv("/duocar/raw/riders/", header=True, inferSchema=True)


# ## Exercises

# (1) Use the `json` method of the `DataFrameWriter` class to write the
# `riders` DataFrame to the `data/riders_json/` (HDFS) directory.

riders.write.json("data/riders_json/", mode="overwrite")

# (2) Use the `hdfs dfs -ls` command to list the contents of the
# `data/riders_json/` directory.

!hdfs dfs -ls data/riders_json

# (3) Use the `hdfs dfs -cat` and `head` commands to display a JSON file in
# the `data/riders_json` directory.

!hdfs dfs -cat data/riders_json/part* | head -n 5

# (4) Use Hue to browse the `data/riders_json/` directory.

# Note that the JSON file is noticeably larger than the original CSV file.

# (5) Use the `json` method of the `DataFrameReader` class to read the JSON
# file into a DataFrame.

riders_json = spark.read.json("data/riders_json/")

# (6) Examine the schema of the DataFrame.  Do you notice anything different?

riders_json.printSchema()

# Note that the columns are in alphabetical order.


# ## Cleanup

# Stop the SparkSession:
spark.stop()
