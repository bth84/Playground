# # Window functions

# Copyright © 2010–2018 Cloudera. All rights reserved.
# Not to be reproduced or shared without prior written 
# consent from Cloudera.

# Compute a new dataset with the `driver_rides` and 
# `driver_stars` columns replaced by cumulative sums of 
# the number of rides for each  driver and the `star_rating`
# column for each driver. 


# ## Setup

library(sparklyr)
library(dplyr)

config <- spark_config()
config$spark.driver.host <- Sys.getenv("CDSW_IP_ADDRESS")
spark <- spark_connect(
  master = "local",
  app_name = "window_solutions",
  config = config
)


# ## Load data

joined <- spark_read_parquet(
  sc = spark,
  name = "joined",
  path = "/duocar/joined/"
)


# ## Solution

fixed <- joined %>% 
  group_by(driver_id) %>% 
  arrange(ride_id) %>%
  mutate(
    driver_rides = cumsum(as.integer(!cancelled)),
    driver_stars = cumsum(star_rating)
  ) %>%
  na.replace(driver_stars = 0)


# ## Check solution

# ### Informal check: 

# Compare this "fixed" dataset to the original joined 
# dataset for some arbitrary driver.

joined %>% 
  filter(driver_id == "220200000106") %>% 
  select(ride_id, star_rating, driver_rides, driver_stars) %>%
  as.data.frame()
  
fixed %>% 
  filter(driver_id == "220200000106") %>% 
  select(ride_id, star_rating, driver_rides, driver_stars) %>%
  as.data.frame()

# ### Formal check: 

# Confirm that the rides and stars values in the most recent
# entry for each driver in your result match the values in
# the existing joined data.

original_stars <- joined %>%
  group_by(driver_id) %>%
  filter(ride_id == max(ride_id)) %>%
  ungroup() %>%
  arrange(driver_id) %>%
  pull(driver_stars)

fixed_stars <- fixed %>%
  group_by(driver_id) %>%
  filter(ride_id == max(ride_id)) %>%
  ungroup() %>%
  arrange(driver_id) %>%
  pull(driver_stars)

all.equal(original_stars, fixed_stars)

# Note: `identical()` may return FALSE because 
# `original_stars` is an integer vector and `fixed_stars`
# is a numeric vector. Convert both to integers to make
# `identical()` return TRUE.

identical(
  as.integer(original_stars),
  as.integer(fixed_stars)
)


# ## Cleanup

spark_disconnect(spark)
