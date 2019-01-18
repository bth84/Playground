# # Window functions

# Copyright © 2010–2018 Cloudera. All rights reserved.
# Not to be reproduced or shared without prior written 
# consent from Cloudera.

# This module demonstrates how to use *window functions* with
# sparklyr. Window functions are a variation on aggregation 
# functions. Whereas an aggregation function, like `sum()`,
# groups sets of rows together into single rows, a window 
# function is applied over sets of rows *without* combining 
# them. Window functions are also called *analytic functions*.


# ## Setup

library(sparklyr)
library(dplyr)

config <- spark_config()
config$spark.driver.host <- Sys.getenv("CDSW_IP_ADDRESS")
spark <- spark_connect(
  master = "local",
  app_name = "window",
  config = config
)


# ## Load the rides data from HDFS into a Spark DataFrame

rides <- spark_read_parquet(
  sc = spark,
  name = "rides",
  path = "/duocar/clean/rides/"
)


# ## Window functions with dplyr

# Window functions in dplyr are implemented using the
# familiar dplyr verbs, but used in different
# combinations and in different order.

# Recall that to perform *standard* aggregation with
# dplyr, you use `summarise()`, preceded optionally by
# `group_by()` if you want to aggregate within groups.
# For example, to aggregate the total distance in miles
# of all DuoCar rides:

rides %>% 
  summarise(sum_distance_mi = sum(distance / 1609.34, na.rm = TRUE))

# The result shows that DuoCar drivers have provided 
# more than 250,000 miles of rides.

# We would like to identify which exact ride was the
# one during which the 250,000th mile was driven, so
# that we can give the driver and passenger a special
# award. To do this, we need to use window functions.

# Arrange the data by date and time, and calculate 
# the cumulative sum of distance (in miles):

cum_sum_miles <- rides %>% 
  arrange(date_time) %>%
  mutate(cum_sum_distance_mi = cumsum(distance / 1609.34))

# Let's limit the columns and look at the first few
# rows of the result DataFrame:

cum_sum_miles %>%
	arrange(date_time) %>%
	select(date_time, cum_sum_distance_mi) %>%
	head()

# and the last few rows of the result DataFrame:

cum_sum_miles %>%
	arrange(desc(date_time)) %>%
	select(date_time, cum_sum_distance_mi) %>%
	head()

# That looks right. To see the SQL that generates
# these results, use `show_query()`:

cum_sum_miles %>% show_query()

# Notice the `OVER` clause in the SQL. In SQL, the
# `OVER` clause defines a *window specification* which
# controls which rows the preceding function operates
# on.

# To find the ride during which the 250,000th mile
# was driven, filter to the rides with cumulative
# sum 250,000 or greater, arrange in ascending order
# by cumulative sum, and take the top row:

award_winner <- cum_sum_miles %>%
	filter(cum_sum_distance_mi >= 250000) %>%
	arrange(cum_sum_distance_mi) %>%
	head(1)

# Let's see which ride it is:

award_winner %>%
	select(id, rider_id, driver_id, distance, cum_sum_distance_mi)

# You could look up this ride ID in the joined data
# to find the driver's and rider's names and other
# details.

# The above example demonstrates how to compute a
# cumulative sum using `cumsum()`. You can also
# compute cumulative means using `cummean()`.

# Other applications of window functions include:
# * Calculating differences between group aggregates
#   and individual row values
# * Computing ranks (see `?ranking` for details)
# * Comparing values ahead of or behind the current
#   value (see `?lead-lag` for details)


# ## Exercise

# The joined data (in `/duocar/joined`) contains the columns
# `driver_rides` and `driver_stars` from the drivers table.
# These columns represent the total number of rides and stars
# that the driver has accumulated as of the time when the 
# data was generated. But these values are potentially 
# misleading, because these values do not represent the 
# values at the time when each ride occurred.
# Beginning with the joined data, compute a new dataset 
# with the `driver_rides` and `driver_stars` columns
# replaced by cumulative sums of the number of rides for each 
# driver and the `star_rating` column for each driver. These
# represent the actual values at the time each ride occurred.

# Confirm that the rides and stars values in the most recent
# entry for each driver in your result match the values in
# the existing joined data.


# ## Cleanup

spark_disconnect(spark)

# ## References

# [dplyr window functions vignette](https://cran.r-project.org/web/packages/dplyr/vignettes/window-functions.html)
