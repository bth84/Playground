# # Ridgeline plot example

# Copyright © 2010–2018 Cloudera. All rights reserved.
# Not to be reproduced or shared without prior written 
# consent from Cloudera.

# [Ridgeline plots](http://serialmentor.com/blog/2017/9/15/goodbye-joyplots), or 
# "[Joyplots](http://blog.revolutionanalytics.com/2017/07/joyplots.html)"
# in R are super popular right now, so we just had to make one.

# Let's make a ridgeline plot that visualizes the weather
# data, exploring the relationship between the Events 
# column (which report on weather events that day, like 
# rain, snow, thunderstorms, fog) and the CloudCover 
# column  (which reports cloud cover that day in okta 
# units, which have a min of 0 and a max of 8).

# ## Setup

library(sparklyr)
library(dplyr)
config <- spark_config()
config$spark.driver.host <- Sys.getenv("CDSW_IP_ADDRESS")
spark <- spark_connect(
  master = "local",
  app_name = "ridgeline_plots",
  config = config
)

# Load weather data

weather <- spark_read_parquet(
  sc = spark,
  name = "weather",
  path = "/duocar/clean/weather/"
)

# ## Get the data

# Limit the data to the Events and CloudCover columns,
# replace the NA in the Events column (a character column)
# with "None".

events_cloud_cover <- weather %>% 
  select(Events, CloudCover) %>% 
  na.replace(Events = "None") %>%
  collect() 


# ## Install ggridges

# Now let's install the package 
# [ggridges](https://github.com/clauswilke/ggridges)
# which provides `geom_` functions for drawing ridgeline
# plots using ggplot2:

if(!"ggridges" %in% rownames(installed.packages())) { 
  install.packages("ggridges")
}

# Now load both ggplot2 and ggridges:

library(ggplot2)
library(ggridges)

# ## Making the ridgeline plot

# We use `geom_density_ridges()` and `theme_ridges()` 
# along with `ggplot()` to make the ridgeline plot:

ggplot(
    events_cloud_cover, 
    aes(x = CloudCover, y = Events)
  ) +
  geom_density_ridges(scale = 1) + 
  theme_ridges()


# ## Advanced version

# We could just stop there, but there are some problems
# with that plot:
# * The y axis is in alphabetical order, which is not 
#   meaningful. It would be better to order the y axis
#   by the mean CloudCover for each Event. 
# * There is not enough data to draw curves for the
#   Rain-Snow and Rain-Fog-Snow Events so it would be
#   better to omit those from the plot
# This version fixes those problems:

events_cloud_cover <- weather %>% 
  select(Events, CloudCover) %>% 
  na.replace(Events = "None") %>%
  group_by(Events) %>%
  mutate(
    m = mean(CloudCover),
    n = count()
  ) %>%
  filter(n > 3) %>% 
  arrange(m) %>%
  ungroup() %>%
  select(-m, -n) %>%
  collect() %>%
  mutate(Events=factor(Events, unique(Events)))

# Now let's plot it again, using the same plotting code
# as last time, but using the new data:


ggplot(
    events_cloud_cover, 
    aes(x = CloudCover, y = Events)
  ) +
  geom_density_ridges(scale = 1) + 
  theme_ridges()

# ## Non-graphical version

# The above is a graphical way to represent the same information  
# that could be represented in a pivot table using `sdf_pivot()`:

weather %>% 
  sdf_pivot(Events ~ CloudCover) %>%
  na.replace("None", 0)

# ## Cleanup

spark_disconnect(spark)
