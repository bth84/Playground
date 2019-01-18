# # dplyr verbs - Interactive Demonstration Code

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
  app_name = "dplyr_verbs_demo_code",
  config = config
)

riders <- spark_read_csv(
  sc = spark,
  name = "riders",
  path = "/duocar/raw/riders/"
)


# #### Interactive Demonstration: `select()`

# Review the `?select_helpers` help documentation.
# Use the `select()` verb with the `ends_with()` 
# function to select the four latitude and longitude
# columns from the riders table.
# Explore what happens when you use multiple select 
# helper functions in one `select()` verb.
# How can you control whether they are interpreted as 
# a logical OR or a logical AND?
# Hint: These select helper functions return numeric
# column positions, not logical vectors. Can you use 
# set operations functions with them?

riders %>% select(ends_with("lat"):ends_with("k_lon"))


# #### Interactive Demonstration: `select_if()`

# Read the help documentation page for `?select_all`
# which describes several variants of `select()` and
# `rename()`. Use the `select_if()` verb with the
# `is.numeric()` function to select all the numeric columns
# in the riders table.

select_if(riders, is.numeric)


# #### Interactive Demonstration: `filter()` and `between()`

# Use Google Maps or some other mapping website to find the
# latitude and longitude coordinates of an approximate
# bounding box around the campus of North Dakota State 
# University in Fargo.
# Then use the `filter()` verb with the `between()` function
# to limit the data to the riders whose homes are located
# within this bounding box.

# Manual check for boundaries of the campus yields the numbers
# supplied:

riders %>%
  filter(between(home_lon, -96.808810, 096.798253)) %>%
  filter(between(home_lat, 46.890480, 46.901213))


# #### Interactive Demonstration: `mutate()`

riders %>% 
  mutate(birth_year = substr(birth_date, 1, 4)) %>%
  select(birth_date, birth_year, everything())


# #### Interactive Demonstration: Bounding box

# What are the longitude and latitude coordinates that make 
# the bounding box in which all of the riders with known
# home locations live?
# Once you have found these coordinates, copy and paste
# them into this code (replacing min_lon, min_lat, and so on
# with the numeric values) to draw a rectangle of the 
# bounding box on a Leaflet map:
#
#```r
#leaflet() %>% 
#  addTiles() %>% 
#  addRectangles(min_lon, min_lat, max_lon, max_lat)
#```

riders%>% 
  summarize(
    min_lon = min(home_lon), 
    min_lat = min(home_lat),
    max_lon = max(home_lon),
    max_lat =max(home_lat)
  )

# Paste results into the code fragment provided:
library(leaflet)
leaflet() %>% 
  addTiles() %>% 
  addRectangles(-97.67161, 46.6334, -96.179, 47.23741)


# #### Interactive Demonstration: `group_by()`

# What proportion of riders who reported their sex
# as female are students? How does this compare to the
# proportions of students among self-reported male drivers
# and drivers who did not identify themselves as male or
# female?

riders %>%
  group_by(sex) %>%
  summarise(
    students = sum(student),
    n = n(),
    proportion = sum(student)/n())


# #### Interactive Demonstration: Multiple verbs

# Use `mutate()`, `filter()`, `arrange()`, and `select()` 
# to get the full names (first name, space, last name)
# of all student riders, in ascending order by last name 
# then first name.

riders %>%
  filter(student == 1) %>%
  mutate(full_name = paste(first_name, last_name)) %>%
  arrange(last_name, first_name) %>%
  select(full_name)


# ## Cleanup

spark_disconnect(spark)
