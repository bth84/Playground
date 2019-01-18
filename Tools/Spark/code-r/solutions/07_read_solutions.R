# # Reading and Writing Data

# Copyright © 2010–2018 Cloudera. All rights reserved.
# Not to be reproduced or shared without prior written 
# consent from Cloudera.


# Sample answers.  You can run this standalone, or execute selected statements 
# in a wider session.


# ## Setup

library(sparklyr)

# Connect local session
config <- spark_config()
config$spark.driver.host <- Sys.getenv("CDSW_IP_ADDRESS")
spark <- spark_connect(
  master = "local",
  app_name = "read_solutions",
  config = config
)

# Reset `data` directory in HDFS
system("hdfs dfs -rm -r -skipTrash data")
system("hdfs dfs -mkdir data")


# ## Exercises

# (1) Read the raw drivers file into a Spark DataFrame.

# View a few records to verify the file format (could also be done in Hue)
system("hdfs dfs -cat /duocar/raw/drivers/* | head -3")

# Comma-delimited, header present.  Read the csv file.
drivers <- spark_read_csv(
  sc = spark,
  name = "drivers",
  path = "/duocar/raw/drivers/",
  header = TRUE,
  infer_schema = TRUE,
  delimiter = ",")

# Verify
drivers


# (2) Save the drivers DataFrame as a JSON file in your `data` directory.

# (guessing a syntax similar to the one for parquet)
spark_write_json(
  drivers,
  path = "data/drivers_json")

# Verify (could also be viewed in Hue)
system("hdfs dfs -cat data/drivers_json/* | head -3")


# (3) Read the drivers JSON file into a Spark DataFrame.

# (guessing syntax similar to the one for parquet)
drivers_json <- spark_read_json(
  sc = spark,
  name = "drivers_json",
  path = "data/drivers_json")


# (4) Delete the JSON file.

# Cleanup directory from HDFS
system("hdfs dfs -rm -r -skipTrash data/drivers_json")


# ## Cleanup

# Stop the `SparkSession`
spark_disconnect(spark)
