# # Working with sparklyr output

# Copyright © 2010–2018 Cloudera. All rights reserved.
# Not to be reproduced or shared without prior written 
# consent from Cloudera.

# Sample answers.  You can run this standalone, or execute
# selected statements in a wider session.


# ## Setup

library(sparklyr)
library(dplyr)

config <- spark_config()
config$spark.driver.host <- Sys.getenv("CDSW_IP_ADDRESS")
spark <- spark_connect(
  master = "local",
  app_name = "sparklyr_output_solutions",
  config = config
)


# ## Exercises


# Create a `tbl_spark` representing the drivers table.

tbl_change_db(spark, "duocar")
drivers <- tbl(spark, "drivers")


# Examine the structure of the drivers table.

drivers %>% glimpse()


# Collect the first 100 rows of the drivers table into 
# a `tbl_df`.

drivers_100 <- drivers %>% head(100)
drivers_100_tbl <- drivers_100 %>% collect()


# Show the SQL statement that sparklyr generated and Spark
# SQL executed to return the first 100 rows of the drivers 
# table.

drivers_100 %>% show_query()


# Use `pull()` to create a character vector containing the
# makes of all the drivers' vehicles.

makes <- drivers %>% pull(vehicle_make)


# Name this character vector `makes`. Then run the following 
# code, which summarizes the data and produces a data frame
# named `makes_df` listing the makes of vehicles driven by
# more than twenty DuoCar drivers, in decreasing order of 
# frequency:

makes_counts <- table(makes)
makes_df <- data.frame(
  vehicle_make = names(makes_counts),
  count = as.numeric(makes_counts)
)
makes_df <- makes_df[makes_df$count > 20, ]
makes_df <- makes_df[
  order(makes_df$count, decreasing = TRUE),
]
rownames(makes_df) <- NULL
makes_df

# Note: The above R code uses base R idioms instead of 
# the dplyr style.


# What are the top three vehicle makes among DuoCar drivers?

# Ford, Toyota, Chevrolet, in that order.


# Visualize the result using the following ggplot2 code:

library(ggplot2)
ggplot(makes_df, aes(x = vehicle_make, y = count)) + geom_col()


# Now write a dplyr command to obtain a result identical
# to the contents of the data frame `makes_df` above.
# Assign the result to a variable named `makes_tbl`.
# Hint: Use the verbs `group_by()`, `summarise()`, `filter()`,
# and `arrange()`.

makes_tbl <- drivers %>%
  group_by(vehicle_make) %>% 
  summarise(count = n()) %>% 
  filter(count > 20) %>%
  arrange(desc(count))


# Collect `makes_tbl` and visualize it using the same 
# ggplot2 code as above, but with `makes_df` replaced
# by the name of the `tbl_df` containing the collected 
# result.

makes_tbl_df <- makes_tbl %>% collect()

library(ggplot2)
ggplot(makes_tbl_df, aes(x = vehicle_make, y = count)) + geom_col()


# ## Cleanup

spark_disconnect(spark)
