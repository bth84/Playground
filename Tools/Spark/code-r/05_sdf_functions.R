# # Accessing the Spark DataFrame API directly

# Copyright © 2010–2018 Cloudera. All rights reserved.
# Not to be reproduced or shared without prior written 
# consent from Cloudera.

# The last R module described how sparklyr works with 
# dplyr to turn a series of dplyr verbs into a SQL
# statement that is executed by Spark SQL.

# sparklyr also provides a family of *native* functions 
# that directly access the methods of the Spark DataFrame
# API. Many (but not all) of these functions begin with 
# `sdf_`.
# This module will introduce some of these functions
# and will describe when to use them and when to use
# the dplyr verbs.

# ## Setup

library(sparklyr)
library(dplyr)

config <- spark_config()
config$spark.driver.host <- Sys.getenv("CDSW_IP_ADDRESS")
spark <- spark_connect(
  master = "local",
  app_name = "sdf_functions",
  config = config
)

riders <- spark_read_csv(
  sc = spark,
  name = "riders",
  path = "/duocar/raw/riders/"
)


# ## Spark DataFrame API functions with dplyr equivalents

# Some of sparklyr's Spark DataFrame API functions are 
# directly comparable to dplyr verbs. For example:

# ### `sdf_sort()`

# Like `arrange()` but:
# * Does not use non-standard evaluation; column names 
#   must be specified as a quoted vector of strings.
# * Can only sort in ascending order

# Comparable examples:

riders %>% arrange(birth_date)
riders %>% sdf_sort("birth_date")

riders %>% arrange(last_name, first_name)
riders %>% sdf_sort(c("last_name", "first_name"))

riders %>% arrange(desc(birth_date))
# (There is no `sdf_sort()` command equivalent to this one.)


# ### `sdf_sample()`

# This is similar to the dplyr function `sample_frac()`.

# When you use `sample_frac()` with sparklyr, it always
# performs sampling *without replacement*.

# `sdf_sample()` samples *with replacement* by 
# default. You can sample without replacement by setting
# the argument `replacement` to `FALSE`.

# Comparable examples of sampling without replacement:

riders %>% sample_frac(0.01)

riders %>% sdf_sample(0.01, replacement = FALSE)

# Example of sampling with replacement:

riders %>% sdf_sample(0.01)


# ### `na.omit()`

# Note: Although this function's name does not begin 
# with `sdf_`, it is nevertheless a Spark DataFrame 
# API function, not a dplyr verb.

# Can be used to filter out all rows with missing values.
# You can also use `filter_all()` to do this, but it's 
# more concise to use `na.omit()`.

# Comparable examples:

riders %>% filter_all(all_vars(!is.na(.)))
riders %>% na.omit()


# ### `na.replace()`

# Note: Although this function's name does not begin 
# with `sdf_`, it is nevertheless a Spark DataFrame 
# API function, not a dplyr verb.

# Can be used to replace missing values in specified 
# columns with specified values.
# You can also use `mutate()` to do this, but it's 
# more concise to use `na.replace()`.

# Comparable examples:

riders %>% mutate(
  sex = ifelse(is.na(sex), "other/unknown", sex)
)
riders %>% na.replace(sex = "other/unknown")


# ### Interactive Demonstration: `show_query()`

# Try using `show_query()` with the above dplyr examples  
# and the comparable Spark DataFrame API function examples 
# and see what it prints in each case.


# ### `sdf_len()`

# This provides an efficient way to create a Spark 
# DataFrame with one column named `id` containing numeric
# values in a range from 1 to the specified number.

# Comparable examples:

spark %>% 
  copy_to(
    data.frame(
      id = as.double(seq_len(1000000))
    )
  )

spark %>% sdf_len(1000000)

# Using `sdf_len()` is much more efficient than using
# `copy_to()` to do this.

# `sdf_along()` works like `sdf_len()`, but takes the 
# length from the length of the argument passed to it.

spark %>% sdf_along(seq_len(1000000))


# ### `sdf_copy_to()`

# We used `sdf_copy_to()` in previous sections.
# It is equivalent to `dplyr::copy_to()` but does not
# require that you have dplyr loaded.

# Equivalent examples:

#```r
#iris_tbl_1 <- spark %>% copy_to(iris)
#
#iris_tbl_2 <- spark %>% sdf_copy_to(iris)
#```

# sparklyr also provides the function `sdf_import()`
# which is equivalent to `sdf_copy_to()` 
# but with the order of the first two arguments reversed.
# This can be handy when you're using the pipe operator.


# Comparable examples:

#```r
#iris_tbl_3 <- spark %>% sdf_copy_to(iris)
#
#iris_tbl_4 <- iris %>% sdf_import(spark, "iris")
#```

# ### Interactive Demonstration: Arguments to Copy Functions

# Notice in the call to `sdf_import()` that the name of
# the table (`"iris"`) is passed in as the `name`
# argument. What happens if you don't pass this argument?
# Investigate the optional arguments to these three 
# functions `copy_to()`, `sdf_copy_to()`, and 
# `sdf_import()`.


# ### `sdf_register()`

# This performs the same action as `compute()`:
# It makes a temporary table in Spark and stores 
# the result in it.

# Comparable examples:

student_rider_homes <- riders %>% 
  filter(student == 1) %>% 
  select(home_lat, home_lon)

student_rider_homes %>% compute("temp_1")

student_rider_homes %>% sdf_register("temp_2")


# ## Other Spark DataFrame API functions

# There are a number of other sparklyr functions whose
# names begin with `sdf_` that we are not yet ready to
# discuss. One example is `sdf_mutate()`.
# Although this function is conceptually comparable to 
# `dplyr::mutate()`, it works very differently in
# practice, and the two are used in different contexts. 

# So when would you use these "native" Spark DataFrame API 
# functions like `sdf_mutate()`?
# Primarily when you're using them in conjunction with 
# another native Spark function that is not compatible
# with the dplyr functions.
# For example, you would use `sdf_mutate()` when 
# you need to apply one of Spark's feature transformers 
# to mutate a DataFrame.

# We will discuss this more later in the course
# when we get to the content about machine learning.


# ## Cleanup

spark_disconnect(spark)
