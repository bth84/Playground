# # Map of longest ride

# Copyright © 2010–2018 Cloudera. All rights reserved.
# Not to be reproduced or shared without prior written 
# consent from Cloudera.

# Get the route coordinates of the longest ride in the DuoCar
# data and use Leaflet to create a map of the ride route

# Setup
library(sparklyr)
library(dplyr)
library(leaflet)

config <- spark_config()
config$spark.driver.host <- Sys.getenv("CDSW_IP_ADDRESS")
spark <- spark_connect(
  master = "local",
  config = config
)

# Load the rides and ride_routes data from HDFS
rides <- spark_read_parquet(
  sc = spark,
  name = "rides",
  path = "/duocar/clean/rides"
)
ride_routes <- spark_read_parquet(
  sc = spark,
  name = "ride_routes",
  path = "/duocar/clean/ride_routes"
)

# Filter the rides data to the longest ride (by distance),
# join it with the ride_routes data, and collect
longest_ride_route <- rides %>% 
  filter(distance == max(distance)) %>%
  inner_join(ride_routes, by = c("id" = "ride_id")) %>%
  collect()

# Make a Leaflet map
leaflet(longest_ride_route) %>%
  addTiles() %>%
  addPolylines(
    lng = ~lon, lat = ~lat) %>%
  addMarkers(
    lng = ~lon[1],
    lat = ~lat[1],
    popup = "Origin"
  ) %>%
  addMarkers(
    lng = ~lon[length(lon)],
    lat = ~lat[length(lat)],
    popup = "Destination"
)

# Cleanup
spark_disconnect(spark)
