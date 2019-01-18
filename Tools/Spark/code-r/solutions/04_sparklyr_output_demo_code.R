# # Working with sparklyr output - Interactive Demonstration Code

# Copyright © 2010–2018 Cloudera. All rights reserved.
# Not to be reproduced or shared without prior written 
# consent from Cloudera.

# Sample code for the interactive demonstrations
# You can run this standalone, or execute selected
# statements in a wider session.


# ## Setup

library(sparklyr)
library(dplyr)

config <- spark_config()
config$spark.driver.host <- Sys.getenv("CDSW_IP_ADDRESS")
spark <- spark_connect(
  master = "local",
  app_name = "sparklyr_output_demo_code",
  config = config
)

riders <- spark_read_csv(
  sc = spark,
  name = "riders",
  path = "/duocar/raw/riders/"
)


# #### Interactive Demonstration: `show_query()`

# Revisit the lengthy dplyr example below the heading 
# "Chaining dplyr verbs" near the bottom of the dplyr verbs
# module. Show the SQL statement that Spark SQL will 
# execute to return the result of this example.

current_month <- format(Sys.Date(), "%m")
current_day <- format(Sys.Date(), "%d")

riders %>% 
  select(birth_date, student) %>%
  mutate(
    birth_month = substr(birth_date, 6, 7),
    birth_day = substr(birth_date, 9, 10),
    birth_decade = paste0(substr(birth_date, 1, 3), "0s")
  ) %>% 
  filter(
    birth_month == current_month,
    birth_day == current_day
  ) %>% 
  group_by(birth_decade) %>% 
  summarize(
    n = n(),
    students = sum(as.numeric(student))
  ) %>%
  arrange(birth_decade) %>%
  show_query()


# #### Interactive Demonstration: Dynamic bounding box

# Return to the interactive demonstration above where you 
# created a Leaflet map of the bounding box of rider home
# locations. This time, instead of copying and pasting
# the values from the printed sparklyr output, use
# `collect()` to return the output to R as a `tbl_df`
# and produce the same plot by passing that `tbl_df`
# to the `leaflet()` function.

bounds <- riders %>%
  summarise(
    lng1 = min(home_lon),
    lat1 = min(home_lat),
    lng2 = max(home_lon),
    lat2 = max(home_lat)
  ) %>%
  collect()

library(leaflet)
leaflet() %>%
  addTiles() %>%
  addRectangles(bounds$lng1, bounds$lat1, bounds$lng2, bounds$lat2)


# ## Cleanup

spark_disconnect(spark)
