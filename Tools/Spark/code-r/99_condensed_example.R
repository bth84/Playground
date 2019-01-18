# # Condensed R Example

# Copyright © 2010–2018 Cloudera. All rights reserved.
# Not to be reproduced or shared without prior written 
# consent from Cloudera.

# ## Using Apache Spark with sparklyr

# CDSW provides a virtual gateway to the cluster, where
# you can run Apache Spark jobs. Cloudera recommends
# using [sparklyr](https://spark.rstudio.com) as the R
# interface to Spark.

# Install the sparklyr package from CRAN (if it is not
# already installed). This might take several minutes,
# because the package must be built from source code:

if(!"sparklyr" %in% rownames(installed.packages())) {
  install.packages("sparklyr")
}


# ### Connecting to Spark

# Begin by loading the sparklyr package:

library(sparklyr)

# Then use the `spark_connect()` function to create a
# Spark connection

# The argument `master` specifies which Spark instance
# to connect to. Use `master = "local"` to connect to 
# a local Spark instance and `master = "yarn"` to 
# connect to Spark on YARN.

# The optional `app_name` argument changes the default
# name of the Spark session.

# When using the local Spark instance from CDSW, it's
# necessary to set one additional Spark configuration
# parameter through the `config` argument:

spark <- spark_connect(
  master = "local",
  app_name = "condensed-example",
  config = list(
    spark.driver.host = Sys.getenv("CDSW_IP_ADDRESS")
  )
)

# Now you can use the connection object named `spark`
# to read data into Spark.


# ### Reading Data

# Load the drivers dataset from HDFS. This data is in CSV
# format and includes a header row. Spark can infer the
# schema automatically from the data:

drivers <- spark_read_csv(
  sc = spark,
  name = "drivers",
  path = "/duocar/raw/drivers/",
  header = TRUE,
  infer_schema = TRUE
)

# The result is a Spark `DataFrame` named `drivers`.
# Note that this is not an R data frame—it is a pointer
# to a Spark `DataFrame`.


# ### Inspecting Data

# Inspect the Spark `DataFrame` to gain a basic
# understanding of its structure and contents.

# To make the code more readable, the examples below use
# the pipe operator `%>%`.

# Print the column names:

drivers %>% colnames()

# Print the number of rows:

drivers %>% sdf_nrow()

# Print the first 10 rows of data, for as many columns
# as fit on the screen (this is the default behavior):

drivers

# Print the first five rows of data, for as many columns
# as fit on the screen:

drivers %>% print(n = 5)

# Print the first five rows of data, showing all the
# columns even if rows wrap onto multiple lines:

drivers %>% print(n = 5, width = Inf)


# ### Transforming Data Using dplyr Verbs

# sparklyr works together with the popular R package
# [dplyr](http://dplyr.tidyverse.org). sparklyr enables
# you to use dplyr *verbs* to manipulate data with Spark.

# The main dplyr verbs are:
# * `select()` to select columns
# * `filter()` to filter rows
# * `arrange()` to order rows
# * `mutate()` to create new columns
# * `summarise()` to aggregate

# In addition to these main dplyr verbs, there are
# some other less important ones that are variations on
# the main verbs, such as `rename()` and `transmute()`.

# And in addition to verbs, dplyr also has the function
# `group_by()` which allows you to perform operations by
# group.

# Load the dplyr package:

library(dplyr)

# `select()` returns the specified columns:

drivers %>% select(first_name, last_name)

# `distinct()` works like `select()` but returns only
# distinct values:

drivers %>% distinct(vehicle_color)

# `filter()` returns rows that satisfy a Boolean
# expression:

drivers %>% filter(first_name == "Cody")

# `arrange()` returns rows arranged by the specified
# columns:

drivers %>% arrange(birth_date)

# The default sort order is ascending. Use the helper
# function `desc()` to sort by a column in descending
# order:

drivers %>% arrange(desc(birth_date))

# `mutate()` adds new columns or replaces existing
# columns using the specified expressions:

drivers %>% 
  mutate(green_vehicle = vehicle_color == "green")

drivers %>% 
  mutate(full_name = paste(first_name, last_name))

# `summarise()` performs aggregations using the 
# specified expressions.

# Use aggregation functions such as `n()`, `n_distinct()`,
# `sum()`, and `mean()`:

drivers %>% summarise(n = n())

drivers %>%
  summarise(num_makes = n_distinct(vehicle_make))

# `group_by()` groups data by the specified columns, so
# aggregations can be computed by group:

drivers %>%
  group_by(vehicle_make) %>%
  summarise(
    num_drivers = n(),
    avg_star_rating = mean(stars / rides)
  )

# You can chain together multiple dplyr verbs:

drivers %>%
  filter(vehicle_make == "Ford") %>%
  mutate(star_rating = stars / rides) %>%
  group_by(vehicle_model) %>%
  summarise(
    num_drivers = n(),
    avg_stars_rating = mean(star_rating)
  ) %>%
  arrange(desc(num_drivers))


# ### Using SQL Queries

# Instead of using dplyr verbs, you can use a SQL query
# to achieve the same result:

tbl(spark, sql("
  SELECT vehicle_model, 
    COUNT(*) AS num_drivers,
    AVG(stars / rides) AS avg_stars_rating
  FROM drivers
  WHERE vehicle_make = 'Ford'
  GROUP BY vehicle_model
  ORDER BY num_drivers DESC"))


# ### Spark DataFrames Functions

# In addition to the dplyr verbs, there are also some
# other data manipulation functions you can use with
# sparklyr. For example:

# `na.omit()` filters out rows with missing values:

drivers %>% na.omit()

# `sdf_sample()` returns a random sample of rows:

drivers %>%
  sdf_sample(fraction = 0.05, replacement = FALSE)


# ### Visualizing Data from Spark

# You can create data visualizations in CDSW using R
# graphics packages such as ggplot2.

# To visualize data from a Spark `DataFrame` with ggplot2,
# you must first return the data as an R data frame. To do
# this, use the `collect()` function.

# Get the vehicle color and star rating from the
# `drivers` dataset, omit missing records, and return 
# the result as an R data frame:

drivers_df <- drivers %>%
  mutate(star_rating = stars / rides) %>%
  select(vehicle_color, star_rating) %>%
  na.omit() %>%
  collect()

# Caution: When working with a large Spark `DataFrame`,
# you may need to sample, filter, or aggregate before
# using `collect()` to return an R data frame.

# Create a boxpot showing the relationship between
# vehicle color and star rating:

library(ggplot2)

ggplot(drivers_df, aes(x=vehicle_color, y=star_rating)) +
  geom_boxplot()

# The boxplot seems to show that drivers with black
# vehicles have higher average reviews, and drivers
# with yellow vehicles have lower average reviews.


# ### Machine Learning with MLlib

# MLlib is Spark's machine learning library.

# Examine the relationship between black vehicles and
# star rating using a linear regression model.

# First, create a `DataFrame` with the relevant
# columns with missing values removed:

drivers_to_model <- drivers %>%
  mutate(
    black_vehicle = vehicle_color == "black",
    star_rating = stars / rides
  ) %>%
  select(black_vehicle, star_rating) %>%
  na.omit()

# Randomly split the data into a training sample (70% of
# records) and a test sample (30% of records):

samples <- drivers_to_model %>%
  sdf_partition(train = 0.7, test = 0.3)

# Specify the linear regression model and fit it to the
# training sample:

model <- samples$train %>%
  ml_linear_regression(star_rating ~ black_vehicle)

# Examine the model coefficients and other model summary
# information:

summary(model)

# Use the model to generate predictions for the test
# sample:

pred <- model %>%
  sdf_predict(samples$test)

# Evaluate the model on the test sample by computing
# R-squared, which gives the fraction of the variance
# in the test sample that is explained by the model:

pred %>%
  summarise(r_squared = cor(star_rating, prediction)^2)

model %>%
  ml_predict(samples$train) %>%
  ml_regression_evaluator(prediction_col="prediction", label_col="star_rating", metric_name="rmse")

model %>%
  ml_predict(samples$test) %>%
  ml_regression_evaluator(prediction_col="prediction", label_col="star_rating", metric_name="rmse")


# ### Cleanup

# Disconnect from Spark:

spark_disconnect(spark)
