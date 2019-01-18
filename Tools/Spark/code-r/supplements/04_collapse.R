# # The dplyr `collapse()` function

# Copyright © 2010–2018 Cloudera. All rights reserved.
# Not to be reproduced or shared without prior written 
# consent from Cloudera.

# 04_sparklyr_output.R talks about `collect()` and `compute()`,
# but it doesn't mention the related function `collapse()`.

# Whereas `collect()` and `compute()` force computation, 
# `collapse()` does not.
# Instead it forces generation of the SQL query.
# The effect of this is that further dplyr operations
# cannot modify the query. Instead, further dplyr operations
# wrap the query in a subquery and can only affect the outer query.

# Here is an example demonstrating this:

# Set up:

student_rider_homes <- riders %>% 
  filter(student == 1) %>% 
  select(home_lat, home_lon)


# Example without `collapse()`:

student_rider_homes %>%
  head(10) %>% 
  show_query()

# Notice that the `LIMIT 10` is added directly to the original query.


# Example with `collapse()`:

student_rider_homes %>%
  collapse() %>% 
  head(10) %>% 
  show_query()

# Notice that the original query has been put into a subquery, 
# and there is an outer query that applies the limit operation:
# `SELECT * FROM (...) LIMIT 10`


# It is rarely necessary to use `collapse()`, but if you encounter 
# a situation where dplyr and sparklyr are generating bad SQL,
# it can be helpful to use it to work around the problem.
