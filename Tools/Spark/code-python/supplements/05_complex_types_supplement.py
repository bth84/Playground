# # Complex Types - Supplement

# Copyright © 2010–2018 Cloudera. All rights reserved.
# Not to be reproduced or shared without prior written 
# consent from Cloudera.


# ## Contents

# * Referencing array elements
# * Referencing map elements
# * Generating arrays using aggregation functions
# * Obtaining complex types from a JSON file


# ## Setup

# Create a SparkSession:
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local").appName("complex_types_supplement").getOrCreate()

# Read the raw data from HDFS:
rides = spark.read.csv("/duocar/raw/rides/", header=True, inferSchema=True)
drivers = spark.read.csv("/duocar/raw/drivers/", header=True, inferSchema=True)
riders = spark.read.csv("/duocar/raw/riders/", header=True, inferSchema=True)


# ## Referencing array elements

# Create an array:
from pyspark.sql.functions import array
drivers_array = drivers \
  .withColumn("vehicle_array", array("vehicle_make", "vehicle_model"))

# Use index notation to access elements of the array:
drivers_array \
  .select("vehicle_array", drivers_array.vehicle_array[0]) \
  .show(1, False)

drivers_array \
  .select("vehicle_array", drivers_array["vehicle_array"][0]) \
  .show(1, False)

from pyspark.sql.functions import col
drivers_array \
  .select("vehicle_array", col("vehicle_array")[0]) \
  .show(1, False)

from pyspark.sql.functions import column
drivers_array \
  .select("vehicle_array", column("vehicle_array")[0]) \
  .show(1, False)

from pyspark.sql.functions import expr
drivers_array \
  .select("vehicle_array", expr("vehicle_array[0]")) \
  .show(1, False)

drivers_array \
  .selectExpr("vehicle_array", "vehicle_array[0]") \
  .show(1, False)


# ## Referencing map elements

# Create a map:
from pyspark.sql.functions import lit, create_map
drivers_map = drivers \
  .withColumn("vehicle_map", create_map(lit("make"), "vehicle_make", lit("model"), "vehicle_model"))

# Use dot notation to access a value by key:
drivers_map \
  .select("vehicle_map", drivers_map.vehicle_map.make) \
  .show(1, False)

drivers_map \
  .select("vehicle_map", drivers_map["vehicle_map"].make) \
  .show(1, False)

drivers_map \
  .select("vehicle_map", col("vehicle_map").make) \
  .show(1, False)

drivers_map \
  .select("vehicle_map", column("vehicle_map").make) \
  .show(1, False)


# ## Generating arrays using aggregation functions

from pyspark.sql.functions import collect_list, collect_set

rides_filled = rides.fillna("Car", subset=["service"])

rides_filled.groupby("rider_id").agg(collect_list("service")).head(5)

rides_filled.groupby("rider_id").agg(collect_set("service")).head(5)


# ## Obtaining complex types from a JSON file

bleats = spark.read.json("/duocar/earcloud/bleats/")
bleats.printSchema()
bleats.select(col("entities")).head(5)
bleats.select(col("entities").urls).head(5)
bleats.select(col("entities").urls[0]).head(5)
bleats.select(col("entities").urls[0].url).head(5)

from pyspark.sql.functions import col, size
bleats.select(size(col("entities").urls)).show(5)
bleats.select(size(col("entities").urls).alias("size")).groupby("size").count().show()

bleats.select(col("user")).show(5, False)
bleats.select(col("user").follower_count).show(5, False)


# ## Cleanup

# Stop the SparkSession:
spark.stop()
