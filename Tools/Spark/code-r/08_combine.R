# # Combining and Splitting Data

# Copyright © 2010–2018 Cloudera. All rights reserved.
# Not to be reproduced or shared without prior written 
# consent from Cloudera.

# In this module we demonstrate how to use sparklyr to combine 
# and split Spark DataFrames.

# * Joining DataFrames
# * Applying set operations to DataFrames
# * Combining DataFrames by columns or rows
# * Splitting a DataFrame


# ## Setup

library(sparklyr)
library(dplyr)

config <- spark_config()
config$spark.driver.host <- Sys.getenv("CDSW_IP_ADDRESS")
spark <- spark_connect(
  master = "local",
  app_name = "combine",
  config = config
)


# ## Joining DataFrames

# dplyr provides several *two-table verbs* that can be used
# to join tables together. sparklyr generates Spark
# SQL-compatible queries from calls to these verbs.

# We will use the following tables to demonstrate joins:

data_scientists <- spark_read_csv(
  sc = spark,
  name = "data_scientists",
  path = "/duocar/raw/data_scientists/"
)
data_scientists

# DuoCar has four data scientists on staff. Sophia and Thomas
# work in the office with office_id 1. Emmanuel works
# remotely (at a home office), indicated by a missing value 
# for office_id. Martina works in the office with office_id
# 5.

offices <- spark_read_csv(
  sc = spark,
  name = "offices",
  path = "/duocar/raw/offices/"
)
offices

# DuoCar has five offices, four in France and one in 
# Fargo, North Dakota.


# ### Inner join

# Returns only matched records

# Example: Return a record for each data scientist who
# works in an office, with information about the office
# they work in.

# Each join function can be called as a function with the 
# left and right tables as the first two arguments:

inner_join(data_scientists, offices, by = "office_id")

# Or you can use the pipe operator `%>%`:

data_scientists %>% inner_join(offices, by = "office_id")

# The examples below all use the pipe. This syntax is
# usually preferred, because this makes it easier to include
# a join in a chain of other operations.


# ### Left outer join

# Returns all records in left table and only matched 
# records in right table

# Example: Return a record for each data scientist, 
# with information about the office they work in if they
# work in an office.

data_scientists %>% left_join(offices, by = "office_id")


# ### Right outer join

# Returns all records in right table and only matched 
# records in left table

# Example: Return a record for each data scientist who works
# in an office, and a record for each office regardless of 
# whether a data scientist works there.

data_scientists %>% right_join(offices, by = "office_id")

# You could use this result to determine which offices 
# don't have any data scientists working in them, by
# looking for the missing values in result.


# ### Full outer join

# Returns all matched and unmatched records

# Example: Return a record for each data scientist 
# regardless of whether they work in an office and
# a record for each office regardless of whether a 
# data scientist works there.

data_scientists %>% full_join(offices, by = "office_id")


# ### Semi join

# A special efficient type of inner join that behaves 
# more like a filter than a join.

# Left semi joins only return records from the table 
# on the left for which there is a match in the table 
# on the right.

# Example: Return a record for each data scientist who 
# who works in an office, but with no details about 
# the offices.

data_scientists %>% semi_join(offices, by = "office_id")


# ### Anti join

# Like a semi join, but returns records from the table
# on the left for which there is *no match* in the table
# on the right

# Example: Return a record for each data scientist who 
# who does *not* work in an office.

data_scientists %>% anti_join(offices, by = "office_id")

# Or: Return a record for each office that does *not*
# have any data scientists working in it.

offices %>% anti_join(data_scientists, by = "office_id")

# Semi joins and anti joins are referred to as *filtering
# joins* because they filter records from the left table
# based on whether or not they match records in the right
# table.


# ### Example: Joining the DuoCar data

# You can chain together multiple join operations in order
# to join more than two tables. In this example, let's join
# the DuoCar rides, drivers, riders, and ride reviews data.

# Read the clean data from HDFS:
rides <- spark_read_parquet(
  sc = spark,
  name = "rides",
  path = "/duocar/clean/rides/"
)
drivers <- spark_read_parquet(
  sc = spark,
  name = "drivers",
  path = "/duocar/clean/drivers/"
)
riders <- spark_read_parquet(
  sc = spark,
  name = "riders",
  path = "/duocar/clean/riders/"
)
ride_reviews <- spark_read_parquet(
  sc = spark,
  name = "ride_reviews",
  path = "/duocar/clean/ride_reviews/"
)

# To join on a column that has different names in the left
# and right tables, specify `by` as a named vector, as in 
# this example below.

# If the names of the columns in the left and right tables
# are non-unique, then the suffixes ".x" and ".y" will be 
# automatically added to the end of the column names from
# the left and right tables respectively. To specify your
# own suffixes instead, pass a character vector of length
# 2 as the `suffix` argument, as shown in this example below.

joined <- rides %>%
  inner_join(
    drivers,
    by = c("driver_id" = "id")
  ) %>%
  inner_join(
    riders,
    by = c("rider_id" = "id"),
    suffix = c("_driver", "_rider")
  ) %>%
  left_join(
    ride_reviews,
    by = c("id" = "ride_id")
  )


# ## Applying set operations to DataFrames

# dplyr provides several verbs that apply basic set
# operations to the records or values in the data.

# We will use the following datasets to demonstrate set
# operations:

riders <- spark_read_parquet(
  sc = spark,
  name = "riders",
  path = "/duocar/clean/riders/"
)

drivers <- spark_read_parquet(
  sc = spark,
  name = "drivers",
  path = "/duocar/clean/drivers/"
)

rider_names <- riders %>% select(first_name)
rider_names

driver_names <- drivers %>% select(first_name)
driver_names


# ### Union

# `union()` returns all unique records that exist in *either
# or both* of the datasets.

names_distinct <- union(rider_names, driver_names)
names_distinct %>% tally()
names_distinct


# ### Intersection

# `intersect()` returns all unique records that exist in 
# *both* of the datasets. 

names_intersect <- intersect(rider_names, driver_names)
names_intersect %>% tally()
names_intersect


# ### Difference

# `setdiff()` returns the unique records that are in 
# *only one* of the two datasets, but *not both*.
# This is also called the *exclusive or* or the 
# *disjunction*.

names_exclusive <- setdiff(rider_names, driver_names)
names_exclusive %>% tally()
names_exclusive


# All three of these set operations verbs (`union()`, 
# `intersect()`, and `setdiff()`) remove duplicate records,
# returning distinct records.

# ### Mini Exercise:

# Apply the three set operation functions to the *full*
# names (first and last) of the riders and drivers.
# Count the numbers of returned results and describe what
# these numbers represent.


# ##  Combining DataFrames by columns or rows

# ### Combining two DataFrames by row

# `rbind()` combines together the rows of two datasets
# and returns the full result. `rbind()` does not remove
# duplicate records.

names_all <- rbind(rider_names, driver_names)
names_all %>% tally()
names_all


# `rbind()` calls the DataFrame API directly. Alternatively,
# you can use the dplyr two-table verb `union_all()` which
# returns the same result as `rbind()` using SQL.

names_all <- union_all(rider_names, driver_names)
names_all %>% tally()
names_all


# ### Combining two or more DataFrames by column

# `cbind()` combines together the columns of two datasets
# and returns the full result. `cbind()` does not perform
# any matching. Because the order in which rows are returned
# in a Spark SQL query is arbitrary, you should generally
# not use `cbind()`; you should instead perform a join.


# ### Combining more than two DataFrames by row or column


# `sdf_bind_rows()` and `sdf_bind_cols()` work like 
# `rbind()` and `cbind()` but you can pass more than two
# DataFrames as parameters.


# ## Splitting a DataFrame

# `sdf_partition()` splits a dataset into random subsets.
# The function takes named parameters that map table names
# to proportions. If the proportions do not add up to 1,
# then they will be normalized.

# This example creates three random subsets of the riders
# table: a 60% training set, a 20% validation set, and
# a 20% test set.

samples <- riders %>% sdf_partition(
    train = 0.6,
    validate = 0.2,
    test = 0.2
  )

# `sdf_partition()` returns a named list containing the 
# randomly sampled DataFrames. 

class(samples)
names(samples)

samples$train
samples$validate
samples$test

samples$train %>% tally()
samples$validate %>% tally()
samples$test %>% tally()

riders %>% tally()


# ## Exercises

# Join together the rides, drivers, riders, and ride reviews
# tables to produce a table that is the same as the joined
# table already stored in HDFS. It's OK if the column names
# are not the same, but the number of records should be the
# same as in the existing joined table.


# ## Cleanup

spark_disconnect(spark)
