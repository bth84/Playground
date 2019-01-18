# # Working with sparklyr output

# Copyright © 2010–2018 Cloudera. All rights reserved.
# Not to be reproduced or shared without prior written 
# consent from Cloudera.

# ## Setup

library(sparklyr)
library(dplyr)

config <- spark_config()
config$spark.driver.host <- Sys.getenv("CDSW_IP_ADDRESS")
spark <- spark_connect(
  master = "local",
  app_name = "sparklyr_output",
  config = config
)

# Load the riders data from HDFS into a Spark DataFrame

riders <- spark_read_csv(
  sc = spark,
  name = "riders",
  path = "/duocar/raw/riders/"
)


# ## Printing and assigning sparklyr output

# The last module showed how you can start with a Spark 
# DataFrame and apply a series of operations to it using
# dplyr verbs.

# Now let's understand more about the objects returned by
# these series of operations.

# If you do *not* assign the output of one or more dplyr
# operations to a variable, then the output is simply 
# printed to the screen:

riders %>% 
  filter(student == 1) %>% 
  select(home_lat, home_lon)

# You can also assign the output to a variable:

student_rider_homes <- riders %>% 
  filter(student == 1) %>% 
  select(home_lat, home_lon)

# If you like, you can use the right assignment operator `->`.
# This is less common, but it makes your code read from left 
# to right:

riders %>% 
  filter(student == 1) %>% 
  select(home_lat, home_lon) -> 
  student_rider_homes

# Assigning a dplyr result to a variable does not prevent
# you from performing more dplyr operations on it.
# Starting with the result variable, you can apply further 
# dplyr operations:

student_rider_homes %>% 
  filter(
    home_lat > mean(home_lat),
    home_lon < mean(home_lon),
  )

# The result variable (`student_rider_homes` in this example)
# is a `tbl_spark` object:

class(student_rider_homes)

# This object represents the source Spark DataFrame and the
# series of dplyr operations that is to be performed on it.


# ## Translating dplyr operations to SQL

# Look again at the class of the `tbl_spark` object:

class(student_rider_homes)

# Notice this object also inherits the classes `tbl_sql`, 
# `tbl_lazy`, and `tbl`. These are classes defined by the 
# packages tibble and dplyr:
# * `tbl` is an underlying class defined by the tibble 
#   package.
# * `tbl_lazy` means that R uses lazy evaluation on this 
#   object; it does not immediately perform operations but 
#   waits until the result is requested.
# * `tbl_sql` means that R can perform operations on this 
#   object by issuing SQL statements which are processed by
#   some SQL backend (Spark SQL in this case).

# You can see an overview the source Spark DataFrame and the
# series of operations to be performed on it to produce a 
# `tbl_spark` object by looking at its `ops` element:

student_rider_homes$ops

# These operations are translated into a SQL statement which
# is then processed by Spark SQL.

# The work of translating dplyr operations into SQL is done 
# by dplyr, with help from sparklyr and the package 
# dbplyr which is a general SQL backend for dplyr.

# You can see the SQL that will perform these operations
# by using the `show_query()` function:

student_rider_homes %>% show_query()

# #### Interactive Demonstration: `show_query()`

# Revisit the lengthy dplyr example below the heading 
# "Chaining dplyr verbs" near the bottom of the dplyr verbs
# module. Show the SQL statement that Spark SQL will 
# execute to return the result of this example.


# ## Executing dplyr operations on Spark SQL

# When you assign the output of one or more dplyr operations
# to a variable, Spark SQL does not actually execute those 
# operations at that time. Spark does not execute the 
# operations until you force execution.

# What are the ways to force execution on a `tbl_spark`?
# They are several ways:

# You can force execution by printing the `tbl_spark`:

student_rider_homes

# This is equivalent to:

print(student_rider_homes)

# Printing a `tbl_spark` causes Spark to return 
# only the first 10 rows of the result to R.

# You can also print a result at the same time as
# assigning it, by enclosing the assignment operation
# in parentheses:

( student_rider_homes <- riders %>% 
    filter(student == 1) %>% 
    select(home_lat, home_lon) )

# You can also force execution by calling
# the `collect()` function on the `tbl_spark`:

student_rider_homes_tbl_df <- 
  student_rider_homes %>% 
  collect()

# This returns the full result to R.

# Another way to force execution is to call the `compute()`
# function, which makes a temporary table in Spark and stores 
# the full result in it:

riders %>% 
  filter(student == 1) %>% 
  select(home_lat, home_lon) %>% 
  compute("student_rider_homes")

# After calling `compute()`, there is a new temporary table 
# registered in your Spark session named `student_rider_homes`.
# You can retrieve it using `tbl()`:

tbl(spark, "student_rider_homes")

# You could then call `collect()` on this temporary table:

student_rider_homes_tbl_df <- 
  tbl(spark, "student_rider_homes") %>%
  collect()


# ## When to collect dplyr output

# We described above how calling the function `collect()` 
# forces execution of the dplyr operations on Spark SQL
# and returns the full result to R. Now let's understand
# more about the object returned by `collect()` and when
# you need to use `collect()`.

# We will learn throughout this training about how sparklyr 
# and dplyr allow you to do many things with `tbl_spark` 
# objects *without* loading the full result into R.
# But some functions require that the data be loaded in R.
# For example, the functions in graphics packages like 
# ggplot2 and leaflet do *not* work with `tbl_spark` objects.

# For instance, this fails, because `student_rider_homes` is 
# a `tbl_spark`:

#``` r
#library(leaflet)
#
#leaflet(student_rider_homes) %>%
#  addTiles() %>%
#  addMarkers(lng = ~home_lon, lat = ~home_lat)
#```

# To make this work, we could need to use `collect()` to
# return the full result to R.

# Let's use `collect()` on `student_rider_homes` and assign
# the result to a variable named `student_rider_homes_tbl_df`:

student_rider_homes_tbl_df <- 
  student_rider_homes %>% 
  collect()

# **Important:**
# So you should only use `collect()` if the result is small 
# enough to fit in R's memory on the computer you're using!

# Now let's take a closer look at this variable.

# The object returned after calling `collect()` is no longer
# a `tbl_spark`. It's now a `tbl_df`, also called a "data 
# frame tbl" or simply a "tibble".

class(student_rider_homes_tbl_df)

# Notice how this object also inherits `data.frame`.
# It *is* actually a data frame, but it has the `tbl_df` 
# and `tbl` classes also which help make the object do 
# things like print more nicely.

# You can make the object just a raw `data.frame` by 
# calling `as.data.frame()`:

student_rider_homes_df <- 
  as.data.frame(student_rider_homes_tbl_df)

class(student_rider_homes_df)

# Then the output when you print that is different
# and takes up more space in the console:

student_rider_homes_df

# So generally we keep these as `tbl_df` objects.

rm(student_rider_homes_df)


# ## After collecting dplyr output

# Once you use `collect()` to return a dplyr result to R as 
# a local `tbl_df`, then you can use functions and operators
# from base R and other R packages on it:

student_rider_homes_tbl_df$home_lat
student_rider_homes_tbl_df[, "home_lat"]

clusters <- kmeans(student_rider_homes_tbl_df, 2)

student_rider_homes_tbl_df$cluster <- 
  factor(clusters$cluster)

# plot it with ggplot2

library(ggplot2)

ggplot(
    student_rider_homes_tbl_df,
    aes(x = home_lon, y = home_lat)
  ) + 
  geom_point(aes(color = cluster))

# plot it with Leaflet

library(leaflet)

leaflet(student_rider_homes_tbl_df) %>%
  addTiles() %>%
  addMarkers(lng = ~home_lon, lat = ~home_lat)


# #### Interactive Demonstration: Dynamic bounding box

# Return to the interactive demonstration in the dplyr verbs
# module where you created a Leaflet map of the bounding box
# of rider home locations. This time, instead of copying and 
# pasting the values from the printed sparklyr output, use
# `collect()` to return the output to R as a `tbl_df` and 
# produce the same plot by passing that `tbl_df` to the 
# `leaflet()` function.

# You can also perform more dplyr operations on a `tbl_df` that was 
# created using sparklyr and collected, but these operations will 
# not be processed by Spark. They will just be processed locally in R:

student_rider_homes_tbl_df %>% 
  summarise(
    avg_lat = mean(home_lat),
    avg_lon = mean(home_lon)
  )


# ## Exercises

# Create a `tbl_spark` representing the drivers table.

# Examine the structure of the drivers table.

# Collect the first 100 rows of the drivers table into 
# a `tbl_df`.

# Show the SQL statement that sparklyr generated and Spark
# SQL executed to return the first 100 rows of the drivers 
# table.

# Use `pull()` to create a character vector containing the
# makes of all the drivers' vehicles.

# Name this character vector `makes`. Then run the following 
# code, which summarizes the data and produces a data frame
# named `makes_df` listing the makes of vehicles driven by
# more than twenty DuoCar drivers, in decreasing order of 
# frequency:

#```r
#makes_counts <- table(makes)
#makes_df <- data.frame(
#  vehicle_make = names(makes_counts),
#  count = as.numeric(makes_counts)
#)
#makes_df <- makes_df[makes_df$count > 20, ]
#makes_df <- makes_df[
#  order(makes_df$count, decreasing = TRUE), 
#]
#rownames(makes_df) <- NULL
#makes_df
#```

# Note: The above R code uses base R idioms instead of 
# the dplyr style.

# What are the top three vehicle makes among DuoCar drivers?

# Visualize the result using the following ggplot2 code:

#```r
#library(ggplot2)
#ggplot(makes_df, aes(x = vehicle_make, y = count)) + geom_col()
#```

# Now write a dplyr command to obtain a result identical
# to the contents of the data frame `makes_df` above.
# Assign the result to a variable named `makes_tbl`.
# Hint: Use the verbs `group_by()`, `summarise()`, `filter()`,
# and `arrange()`.

# Collect `makes_tbl` and visualize it using the same 
# ggplot2 code as above, but with `makes_df` replaced
# by the name of the `tbl_df` containing the collected 
# result.


# ## Cleanup

spark_disconnect(spark)
