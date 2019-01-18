# # dplyr verbs

# Copyright © 2010–2018 Cloudera. All rights reserved.
# Not to be reproduced or shared without prior written 
# consent from Cloudera.

# At the end of the last module, we saw some examples of 
# `select()` and `distinct()`. These are examples of dplyr 
# *verbs*.  In this module we will look at the common dplyr 
# verbs, and understand how they work in sparklyr and how 
# they can be used together to perform data manipulation 
# tasks.


# ## Setup

library(sparklyr)
library(dplyr)

config <- spark_config()
config$spark.driver.host <- Sys.getenv("CDSW_IP_ADDRESS")
spark <- spark_connect(
  master = "local",
  app_name = "dplyr_verbs",
  config = config
)

# ## Load the riders data from HDFS into a Spark DataFrame

riders <- spark_read_csv(
  sc = spark,
  name = "riders",
  path = "/duocar/raw/riders/"
)


# ## Background about dplyr verbs

# dplyr provides a set of *verbs* to perform the most 
# common data manipulation tasks.
# The main dplyr verbs are:
# * `select()` to select columns
# * `filter()` to filter rows
# * `arrange()` to order rows
# * `mutate()` to create new columns
# * `summarise()` to aggregate
# * `group_by()` to define groups of rows

# `group_by()` is not exactly a verb, but a function that 
# can be used to modify the behavior of other verbs.

# dplyr verbs work on local data frames in R memory.
# Most R users are familiar with this.

# For example: Calculate the rounded average sepal width 
# of iris flowers by species after removing some outlying 
# values, and return the results in increasing order:

avg_sepal_width <- iris %>%
  select(Sepal.Width, Species) %>%
  filter(Sepal.Width > 2 & Sepal.Width < 4.0) %>%
  group_by(Species) %>%
  summarise(Avg.Sepal.Width = mean(Sepal.Width)) %>%
  arrange(Avg.Sepal.Width) %>%
  mutate(Avg.Sepal.Width = round(Avg.Sepal.Width, 1))

# Print the result:

avg_sepal_width

# Many R users have used dplyr in this way.


# ## dplyr verbs with sparklyr

# dplyr verbs also work on Spark DataFrames.
# sparklyr enables this; it provides a backend to dplyr 
# for Spark.

# For example: load the iris flower data into Spark,
# and repeat the same dplyr example as above.

iris_spark <- sdf_copy_to(spark, iris)

avg_sepal_width_spark <- iris_spark %>%
  select(Sepal_Width, Species) %>%
  filter(Sepal_Width > 2.0 & Sepal_Width < 4.0) %>%
  group_by(Species) %>%
  summarise(Avg_Sepal_Width = mean(Sepal_Width)) %>%
  arrange(Avg_Sepal_Width) %>%
  mutate(Avg_Sepal_Width = round(Avg_Sepal_Width, 1))

# This is the same as the example above, except periods 
# in column names are replaced by underscores, because 
# sparklyr automatically renames columns in this way.

# The result is the same as in the example above:

avg_sepal_width_spark


# ## Simple dplyr examples

# These examples use only one dplyr verb at a time, to 
# highlight what each verb does.

# ### `select()`

# `select()` selects one or more columns

riders %>% select(first_name)

riders %>% select(first_name, last_name)

# You can also use `select()` to remove specific columns:

riders %>% select(-first_name)

riders %>% select(-c(first_name, last_name))

# You can use the colon operator (`:`) to specify a range
# of columns by name:

riders %>% select(home_lat:work_lon)

# There are several helper functions you can use inside 
# `select()`, including `starts_with()`, `ends_with()`, 
# `contains()`, and `matches()`:

riders %>% select(ends_with("_name"))

riders %>% select(-ends_with("_name"))

# You can also use `select()` to reorder columns.
# For example, say you want `first_name` and `last_name`
# to be the leftmost columns, and you want to keep all 
# the other columns. To do this, use `select()` with
# the helper function `everything()`:

riders %>% select(
  first_name,
  last_name,
  everything()
)

# For more details about the functions you can use inside
# `select()` see `?select_helpers`.

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

# You can also refer to columns by their numeric positions:

riders %>% select(4:5)

riders %>% select(-(4:5))

# There are several variations on `select()`:

# `distinct()` works like `select()`
# but it returns only distinct values

riders %>% distinct(first_name)

riders %>% distinct(first_name, last_name)

# `pull()` is a shortcut for `select()` then 
# `collect()` for a single column, 
# but it returns a vector instead of a `tbl`:

riders %>% select(first_name) %>% collect()

riders %>% pull(first_name)

# You can use `select()` to rename columns, but it only keeps
# the columns you specify:

riders %>% select(fname = first_name, lname = last_name)

# To rename some columns while keeping all other columns, 
# use `rename()` instead of `select()`:

# Unfortunately this throws an error due to some bug:
#```r
#riders %>% rename(fname = first_name, lname = last_name)
#```

# Workaround: Rename the columns one at a time:
riders %>% 
  rename(fname = first_name) %>%
  rename(lname = last_name)

# #### Interactive Demonstration: `select_if()`

# Read the help documentation page for `?select_all`
# which describes several variants of `select()` and
# `rename()`. Use the `select_if()` verb with the
# `is.numeric()` function to select all the numeric columns
# in the riders table.


# ### `filter()` filters rows of data by one or more conditions

riders %>% filter(first_name == "Skylar")

riders %>% filter(!is.na(sex))

# To filter by more than one condition, you can use `&` 
# or list the conditions separated by commas

riders %>% filter(first_name == "Skylar", sex == "female")

riders %>% filter(first_name == "Skylar" & sex == "female")

# And you can use other logical operators like `|`:

riders %>% filter(first_name == "Skylar" | last_name == "Hayes")


# There are three special variants of `filter()` that you
# can use to apply filtering operations to multiple columns.
# These are `filter_all()`, `filter_at()`, and `filter_if()`.
# These verbs use *predicate expressions* to specify
# which columns to filter on.

# `filter_all()` applies filtering criteria either to
# *any* column, or to *all* columns.
# The pronoun `.` is used to refer to these columns.

# To apply filter criteria to *all* columns, use
# the function `all_vars()`. For example, to filter
# out all rows in which any column has a missing value:

riders %>%
  filter_all(
    all_vars(!is.na(.))
  )

# The remaining rows do not contain any missing values.

# Or to filter to the rows in which the first name, the
# last name, or both are "Ryan":

riders %>% 
  select(first_name, last_name) %>%
  filter_all(
    any_vars(. == "Ryan")
  )

# Another way to do this is to use `filter_at()`
# which applies to predicate expression only to certain
# variables, which are specified using `vars()`:

riders %>% filter_at(
  vars(ends_with("_name")),
  any_vars(. == "Ryan")
)

# The verb `filter_if()` takes this concept further, 
# and allows you to select which variables to filter on
# by using a separate predicate function. For example,
# to filter out missing values in numeric columns, but 
# leave missing values in non-numeric columns:

riders %>% filter_if(
  ~ is.numeric(.),
  all_vars(!is.na(.))
)

# Note: These `filter_*()` variants are a relatively
# exotic topic. Don't worry about remembering exactly
# how to use them; just remember that they exist.
# You can always look at the help documentation to 
# see how to use them.

# #### Interactive Demonstration: `filter()` and `between()`

# Use Google Maps or some other mapping website to find the
# latitude and longitude coordinates of an approximate
# bounding box around the campus of North Dakota State 
# University in Fargo.
# Then use the `filter()` verb with the `between()` function
# to limit the data to the riders whose homes are located
# within this bounding box.

# Instead of filtering based on some conditions, you may
# simply want to return some fixed number of rows. To
# return the first *`x`* rows, use `head(`*`x`*`)`:

riders %>% head(5)

# If you return more than 10 rows, only 10 will be printed:

riders %>% head(20)

# To print more than 10 rows, use `print(n = `*`x`*`)`:

riders %>% head(20) %>% print(n = 20)

# Note that row order in an unordered Spark DataFrame is 
# arbitrary, so you may not get the same rows each time 
# you use `head()` on the same data.

# Another option is to filter to a *random* set of rows.
# This is of course known as *sampling*.
# There are two special dplyr verbs for random sampling:

# `sample_n()` samples a fixed number of rows.
# For example, to sample 100 rows from the riders table:

riders %>% sample_n(100)

# `sample_frac()` samples the specified fraction of rows.
# For example, to sample 1% of the rows from the riders table:

riders %>% sample_frac(0.01)

# When you use these two sampling functions on `tbl_spark`
# objects, they always perform sampling *without replacement*
# and they do not do any grouping or weighting.


# ### `arrange()` orders the rows of the data by the values of variables

riders %>% arrange(birth_date)

riders %>% arrange(last_name, first_name)

# The default sort order is ascending.
# Use `desc()` to sort in descending order:

riders %>% arrange(desc(birth_date))


# ### `mutate()` creates one or more new columns

riders %>% mutate(full_name = paste(first_name, last_name))

# #### Interactive Demonstration: `mutate()`

# Use the `mutate()` verb with the `substr()` function
# to get the birth year of riders.

# You can use `mutate()` to replace existing columns, but
# note that the order of the columns may not be preserved:

riders %>% mutate(
  sex = ifelse(is.na(sex), "other/unknown", sex)
)

# You can use `mutate()` to change the data types of 
# one or more columns:

riders %>% mutate(
  id = as.character(id),
  birth_date = as.date(birth_date),
  student = as.logical(student)
)

# Note that the `birth_date` column still appears in the
# printed R `tbl_spark` to be a string column, but 
# internally Spark now recognizes it as a date column.

# Some R-specific type conversion functions do not work
# with `mutate()` on `tbl_spark` objects, but they do work
# after you `collect()` the `tbl_spark` to a `tbl_df`:

# This fails:

#```r
#riders %>%
#  mutate(
#    birth_date = as.POSIXct(birth_date)
#  )
#```

# This works:

riders %>%
  collect() %>% 
  mutate(
    birth_date = as.POSIXct(birth_date)
  )

# You can refer to columns that you have just created in
# the same `mutate()`:

riders %>% mutate(
  full_name = paste(first_name, last_name),
  uppercase_full_name = toupper(full_name)
)

# A variation on `mutate()` is `transmute()`.
# `transmute()` keeps only the specified columns
# and discards all the other columns:

riders %>% transmute(full_name = paste(first_name, last_name))

riders %>% transmute(
  id,
  full_name = paste(first_name, last_name)
)


# ### `summarise()` applies aggregation functions to the data

# When used by itself, `summarise()` returns a single row:

riders %>% summarise(n = n())

riders %>% summarise(
  num_unique_first_names = n_distinct(first_name)
)

riders %>% summarise(
  prop_students = mean(as.numeric(student)),
  women = sum(as.numeric(sex == "female")),
  men = sum(as.numeric(sex == "male"))
)

# You can also use the American English spelling: `summarize()` (z instead of s):

riders %>% summarize(
  oldest = min(birth_date),
  youngest = max(birth_date)
)

# Tip: `tally()` is a shortcut for `summarise(n = n())`:

riders %>% tally()

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

# `summarise()` is more interesting when combined with 
# `group_by()`:


# ### `group_by()` defines groups of rows

# `group_by()` does nothing by itself
# but it changes what other dplyr verbs do.

# Most importantly: `group_by()` causes the next 
# `summarise()` operation to be performed *by group*.
# So instead of returning one single row, `summarise()`
# returns one row *per group*:

riders %>% 
  group_by(sex) %>% 
  summarise(n = n())

riders %>% 
  group_by(sex, ethnicity) %>%
  summarise(
    n = n(),
    students = sum(as.numeric(student))
  )

# Tip: `count(column)` is a shortcut for 
# `group_by(column) %>% summarise(n = n())`:

riders %>% count(sex)

# #### Interactive Demonstration: `group_by()`

# What proportion of riders who reported their sex
# as female are students? How does this compare to the
# proportions of students among self-reported male drivers
# and drivers who did not identify themselves as male or
# female?


# ### Using variables in dplyr expressions

# You can use local variables in dplyr expressions.
# For example:

first_names <- c("Brian", "Glynn", "Ian")
riders %>% filter(first_name %in% first_names)

# But be careful about cases where the variable name is 
# the same as the name of one of the columns. For example, 
# this returns an unexpected result:

first_name <- "Brian"
riders %>% filter(first_name == first_name)

# This returns *all* rows, because both instances of 
# `first_name` in the expression are interpreted as 
# references to the column named `first_name`, not the 
# variable named `first_name`.

# To make dplyr interpret `first_name` as a variable, use the
# *unquote* operator `!!`:

first_name <- "Brian"
riders %>% filter(first_name == !! first_name)

# This returns the expected result.

# For details about how this works, see the 
# [Programming with dplyr vignette](https://cran.r-project.org/web/packages/dplyr/vignettes/programming.html).


# ## Chaining dplyr verbs

# You can accomplish most common data analysis tasks
# by chaining together these simple dplyr verbs.

# For example, you can find out:
# How many riders and how many student riders have their
# birthday today, broken down by the decade of their
# birth, with the result returned in order by decade:

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
  arrange(birth_decade)

# #### Interactive Demonstration: Multiple verbs

# Use `mutate()`, `filter()`, `arrange()`, and `select()` 
# to get the full names (first name, space, last name)
# of all student riders, in ascending order by last name 
# then first name.


# ## Cleanup

spark_disconnect(spark)
