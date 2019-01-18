# # Using base R functions and operators with Spark DataFrames

# Copyright © 2010–2018 Cloudera. All rights reserved.
# Not to be reproduced or shared without prior written 
# consent from Cloudera.

# Many of the usual functions and operators that work on data
# frames do not work on `tbl_spark` objects. (They may fail 
# or return unexpected output.)

# For example, you cannot use `$` or `[]` to get a column 
# from a `tbl_spark`: 

#```r
#riders$student # returns NULL
#
#riders[, "student"] # fails
#```

# And you cannot use `str()` to examine the structure of a `tbl_df`
# (it returns a mess of technical info, not the column structure)

#```r
#riders %>% str() # returns non-useful output
#```

# However, for many operations, dplyr provides its own ways 
# to work with `tbl_spark` objects.

# For example, instead of using `$` or `[]` to get a single 
# column, dplyr provides the function `pull()` which collects
# just the one column:

riders %>% pull(student)

# And instead of using `str()` to see the structure, you can use `glimpse()`:

riders %>% glimpse()
