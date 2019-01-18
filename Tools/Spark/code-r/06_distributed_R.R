# # Distributing R computations on the cluster

# Copyright © 2010–2018 Cloudera. All rights reserved.
# Not to be reproduced or shared without prior written 
# consent from Cloudera.

# sparklyr provides the function `spark_apply()` which
# can be used to run arbitrary R code on the cluster nodes
# where the Spark executors are running.

# This is useful in cases where a function you need
# to apply your data is available only in R or an an
# R package, not in Spark. But note that R functions 
# may not run as efficiently as native Spark functions.

# An example is presented below.
# For full details about `spark_apply()`, see the sparklyr
# [Distributing R Computations guide](https://spark.rstudio.com/articles/guides-distributed-r.html).
# See the information there about partitioning and grouping,
# which are not described in the example below.


# ## Setup

library(sparklyr)
library(dplyr)

config <- spark_config()
config$spark.driver.host <- Sys.getenv("CDSW_IP_ADDRESS")
spark <- spark_connect(
  master = "local",
  app_name = "distributed_R",
  config = config
)

rides <- spark_read_parquet(
  sc = spark,
  name = "rides",
  path = "/duocar/clean/rides/"
)


# ## Example: Haversine distance

# The [haversine distance](https://en.wikipedia.org/wiki/Haversine_formula)
# measures the shortest distance between two points on a sphere.
# This can be used as an approximate lower bound for the 
# driving distance between two points.

# `spark_apply()` applies an R function to a Spark DataFrame.
# The R function must take a data frame as input
# and return a data frame (or something that can be coerced
# into a data frame).

# Below is a function that conforms to these requirements
# and that calculates the haversine distance between
# the ride origin and destination points, using the column
# names that are used in the rides table.
  
haversine_function <- function(e) {
  R <- 6378137 # Earth radius in meters
  dlat <- (e$dest_lat - e$origin_lat) * pi / 180
  dlon <- (e$dest_lon - e$origin_lon) * pi / 180
  lat1 <- e$origin_lat * pi / 180
  lat2 <- e$dest_lat * pi / 180
  a <- sin(dlat / 2)^2 + 
    cos(lat1) * cos(lat2) * sin(dlon / 2.0)^2
  c <- 2.0 * asin(sqrt(a))
  hdist <- round(R * c)
  data.frame(e, hdist)
}

# Now apply this function to the first 1000 records in 
# the rides table:

rides_with_haversine_distance <- rides %>% 
  head(1000) %>%
  spark_apply(
    f = haversine_function,
    columns = c(colnames(rides), "haversine_distance")
  )

# Compare the resulting `haversine_distance` column
# to the actual ride distance in the `distance` column.

rides_with_haversine_distance %>% 
  select(distance, haversine_distance)

# Instead of writing the haversine distance function 
# yourself, you could use an existing haversine distance
# function which is available in the geosphere R package.

if(!"geosphere" %in% rownames(installed.packages())) {
  install.packages("geosphere")
}

library(geosphere)

# But you need to create a wrapper around this function
# `geosphere::distHaversine()` so that it conforms to the
# requirements for a function used with `spark_apply()`:

haversine_function <- function(e) {
  origin <- e[, c("origin_lon", "origin_lat")]
  dest <- e[, c("dest_lon", "dest_lat")]
  hdist <- 
    round(geosphere::distHaversine(origin, dest))
  data.frame(e, hdist)
}

# Then you can call this function the same way:

rides_with_haversine_distance <- rides %>% 
  head(1000) %>%
  spark_apply(
    f = haversine_function,
    columns = c(colnames(rides), "haversine_distance")
  )

# If you run this while connected to Spark on YARN,
# then R must be installed on the cluster nodes where
# the Spark executors are running. sparklyr takes care of
# copying the R packages to the cluster nodes the first time
# you call `spark_apply()` in a Spark session.
# The packages are only copied once per session
# and persist for the duration of the session.
# The total size of the R packages may be large, so the
# first call to `spark_apply()` may take a long time
# when connected to Spark on YARN.
# You can disable this package copying feature by setting
# `packages = FALSE`. 

# Packages are not copied when you are connected to local
# Spark because they are already installed on the system
# where Spark is running.


# ## Cleanup

spark_disconnect(spark)
