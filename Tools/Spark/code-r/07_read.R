# # Reading and Writing Data

# Copyright © 2010–2018 Cloudera. All rights reserved.
# Not to be reproduced or shared without prior written 
# consent from Cloudera.

# ## Setup

library(sparklyr)
config <- spark_config()
config$spark.driver.host <- Sys.getenv("CDSW_IP_ADDRESS")
spark <- spark_connect(
  master = "local",
  app_name = "read",
  config = config
)


# ## Working with delimited text files

# The rider data is a comma-delimited text file.
# Use the `spark_read_csv()` function to read it into 
# a Spark DataFrame.

# ### Infer the schema

# Showing only the required arguments:

riders <- spark_read_csv(
  sc = spark,
  name = "riders",
  path = "/duocar/raw/riders/"
)

# Showing some more arguments, which are set to the
# default values:

riders <- spark_read_csv(
  sc = spark,
  name = "riders",
  path = "/duocar/raw/riders/",
  header = TRUE,
  infer_schema = TRUE,
  delimiter = ","
)

# Show the schema and first few rows of data:
riders

# The returned object (`riders` in this example) is a 
# `tbl_spark`.

class(riders)

# This represents a remote Spark DataFrame. It is not 
# loaded in R's memory.
# Notice that only the first few rows of data are shown.


# ### Manually specify the schema

riders2 <- spark_read_csv(
  sc = spark,
  name = "riders",
  path = "/duocar/raw/riders/",
  infer_schema = FALSE,
  columns = c(
    id = "character",
    birth_date = "character",
    start_date = "character",
    first_name = "character",
    last_name = "character",
    sex = "character",
    ethnicity = "character",
    student = "integer",
    home_block = "character",
    home_lat = "numeric",
    home_lon = "numeric",
    work_lat = "numeric",
    work_lon = "numeric"
  )
)

riders2

# Notice, in this explicit schema we override the default 
# choices that Spark would have made for the data types 
# of the `id` and `home_block` columns.

# Write the file to a tab-delimited file:

system("hdfs dfs -rm -r -skipTrash data")

system("hdfs dfs -mkdir data")

spark_write_csv(
  riders,
  path = "data/riders_tsv",
  delimiter = "\t"
)

system("hdfs dfs -ls data/riders_tsv")

system("hdfs dfs -cat data/riders_tsv/* | head -n 5")

# **Note:** Disregard the `cat: Unable to write to 
# output steam.` message.


# ## Working with Parquet files

# Write the riders data to a Parquet file:

spark_write_parquet(
  riders,
  path = "data/riders_parquet"
)

system("hdfs dfs -ls data/riders_parquet")

# **Warning:** If you try to overwrite an existing file
# in HDFS, an error will result. You can try this by
# running the above `spark_write_parquet()` command
# a second time.

# You can specify `mode = "overwrite"` to overwrite
# any existing data with the new data:

spark_write_parquet(
  riders,
  path = "data/riders_parquet",
  mode = "overwrite"
)

# Note that in a Parquet file, the schema is stored 
# with the data.

# Read in the Parquet file:

riders_parquet <- spark_read_parquet(
  sc = spark,
  name = "riders_parquet",
  path = "data/riders_parquet"
)

riders_parquet


# ## Working with Hive databases and tables

# In addition to using any data you have read into Spark
# in the current session, you can also access tables 
# defined in the metastore.

# To do this, you need to use some functions from the 
# R package dplyr:

library(dplyr)

# To list all tables in the default metastore database—plus
# any data you have read into Spark in the current session—
# use the function `src_tbls()` from the dplyr package:

src_tbls(spark)

# To create a `tbl_spark` object representing a Spark 
# DataFrame containing the data in one of these Hive tables,
# use the function `tbl()` from the dplyr package:

airlines <- tbl(spark, "airlines")

airlines


# Sometimes you need to use a table that's in a non-default
# database in the metastore. To see what databases are 
# available, call the `src_databases()` function:

src_databases(spark)

# Then use the function `in_schema()` in the dbplyr 
# package to qualify which database a table is in.
# To avoid loading the dbplyr package, you can use 
# `dbplyr::` before the function call:

drivers <- tbl(spark, dbplyr::in_schema("duocar", "drivers"))

# Another way to use a table that's in a non-default 
# database is to change the current database.
# You can do this using the `tbl_change_db()` function:

tbl_change_db(spark, "duocar")

# Then you can refer to tables in the duocar database 
# without using `dbplyr::in_schema()`:

drivers <- tbl(spark, "drivers")

# Then switch back to the default database:

tbl_change_db(spark, "default")

# But be careful about switching databases using 
# `tbl_change_db()`: Any `tbl_spark` objects you created 
# based on tables in a different database will no longer work
# once you switch databases. For example, this will now fail:

#```r
#drivers
#```

# It is safer to use `dbplyr::in_schema()`.

rm(drivers)


# ## Executing SQL queries 

# You can use the `dbGetQuery()` function from the DBI
# package to run SQL queries on Spark SQL:

library(DBI)

dbGetQuery(spark, "SHOW DATABASES")

dbGetQuery(spark, "SHOW TABLES")

dbGetQuery(spark, "DESCRIBE airlines")

dbGetQuery(spark, "SELECT * from airlines limit 10")


# `dbGetQuery()` returns the query result to R as a data frame

airlines <- dbGetQuery(spark, "SELECT * FROM airlines")

class(airlines)

airlines

# **Important:** Only use `dbGetQuery()` when the query 
# result will be small enough to fit in memory in your R 
# session.


# You can also return the result of a SQL query as a
# `tbl_spark`. To do this, you need to use the dplyr
# functions `tbl()` and `sql()`:

flights <- tbl(spark, sql("SELECT * FROM flights"))

class(flights)

flights


# But remember that for simple queries like this, 
# you need not use SQL statements to access Hive tables 
# with sparklyr. Instead you can just reference 
# the Hive table name with `tbl()`:

flights <- tbl(spark, "flights")

# This gives exactly the same result:

class(flights)

flights

# There are more details in upcoming modules about how
# the R packages sparklyr and dplyr work together.


# ## Copying data frames from R to Spark

# Use the `sdf_copy_to()` function to copy a 
# local R data frame to Spark:

iris_tbl <- sdf_copy_to(spark, iris)

iris_tbl

# If you remove the variable `iris_tbl` (which 
# represents this remote Spark data frame) you can
# re-create it without copying the data to Spark again.
# Just use `tbl()` and reference the name (`iris`):

rm(iris_tbl)

iris_tbl <- tbl(spark, "iris")

iris_tbl

# **Note:** `sdf_copy_to()` *does not persist* the 
# copied data in HDFS.
# The data is stored in a temporary location in 
# HDFS and may be cached in Spark memory.
# After you end your session by disconnecting 
# from Spark, it will no longer be available.


# ## Exercises

# (1) Read the raw drivers file into a Spark DataFrame.

# (2) Save the drivers DataFrame as a JSON file in your `data` directory.

# (3) Read the drivers JSON file into a Spark DataFrame.

# (4) Delete the JSON file.


# ## Cleanup

# Remove files:

system("hdfs dfs -rm -r -skipTrash data/riders_tsv")
system("hdfs dfs -rm -r -skipTrash data/riders_parquet")

# Stop the `SparkSession`:

spark_disconnect(spark)
