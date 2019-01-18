# # rwebhdfs

# Copyright © 2010–2018 Cloudera. All rights reserved.
# Not to be reproduced or shared without prior written 
# consent from Cloudera.

# If for some reason you really can't use Spark with sparklyr,
# or Impala with implyr, there's the R package rwebhdfs 
# that can access files in HDFS through WebHDFS.
# It's not on CRAN, only GitHub at 
# https://github.com/saurfang/rwebhdfs


# to install
if(!"rwebhdfs" %in% rownames(installed.packages())) { 
  devtools::install_github("saurfang/rwebhdfs")
}

# to use
library(rwebhdfs)

hdfs <- webhdfs("master-1", 50070, "username")

ride_reviews_string <- read_file(
  hdfs,
  "/duocar/raw/ride_reviews/ride_reviews_fargo.txt"
)

# `read_file()` returns the full file contents in a
# character vector of length one.
# to make this into a tibble, use the package readr:

library(readr)

ride_reviews <- read_tsv(
  ride_reviews_string,
  col_names = c("ride_id", "review")
)
