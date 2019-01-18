# # Inspecting a Spark SQL DataFrame - Solutions

# Copyright © 2010–2018 Cloudera. All rights reserved.
# Not to be reproduced or shared without prior written 
# consent from Cloudera.


# ## Setup

# Create a SparkSession:
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local").appName("inspect_solutions").getOrCreate()


# ## Exercises

# (1) Read the raw driver data from HDFS to a Spark SQL DataFrame called
# `drivers`.

drivers = spark.read.csv("/duocar/raw/drivers/", header=True, inferSchema=True)

# (2) Examine the inferred schema.  Do the data types seem appropriate?

drivers.printSchema()

# (3) Verify the integrity of the putative primary key `drivers.id`.

from pyspark.sql.functions import count, countDistinct
drivers.select(count("*"), count("id"), countDistinct("id")).show()

# (4) Determine the unique values of `drivers.vehicle_make`.

drivers.select("vehicle_make").distinct().count()
drivers.select("vehicle_make").distinct().show(30)

# (5) Compute basic summary statistics for `drivers.rides`.

drivers.describe("rides").show()

# (6) Inspect `drivers.birth_date`.  What data type did Spark SQL infer?

drivers.select("birth_date").show(5)

# Spark SQL inferred the data type to be a date AND time even though the
# column represents a pure date.  This is likely to ensure compatibility with
# Apache Impala, which does not support pure dates at this time.

# (7) **Bonus:** Inspect additional columns of the `drivers` DataFrame.

# (8) **Bonus:** Inspect the raw rider data.


# ## Cleanup

# Stop the SparkSession:
spark.stop()
