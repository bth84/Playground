# # implyr

# Copyright © 2010–2018 Cloudera. All rights reserved.
# Not to be reproduced or shared without prior written 
# consent from Cloudera.

# sparklyr is one dplyr interface package that enables you to
# query data stored in HDFS or S3—but it's not the only one.
# Another such package is 
# [implyr](http://github.com/ianmcook/implyr).
# Just as sparklyr enables you to use dplyr verbs with Apache
# Spark on the back end, implyr enables you to use dplyr 
# verbs with Apache Impala (incubating) on the back end.

# For more information, see the Cloudera Engineering Blog
# post ["implyr: R Interface for Apache Impala"](https://blog.cloudera.com/blog/2017/07/implyr-r-interface-for-apache-impala/).

# Example:

if(!"implyr" %in% rownames(installed.packages())) {
  install.packages(
    "implyr",
    contriburl = "https://s3.amazonaws.com/r-pkgs/bin/linux/redhat"
  )
}
if(!"odbc" %in% rownames(installed.packages())) {
  install.packages(
    "odbc",
    contriburl = "https://s3.amazonaws.com/r-pkgs/bin/linux/redhat"
  )
}

library(DBI)
library(implyr)

drv <- odbc::odbc()
impala <- src_impala(
    drv = drv,
    driver = "Impala ODBC Driver",
    host = "worker-1",
    port = 21050,
    database = "duocar"
  )

src_tbls(impala)

riders <- tbl(impala, "riders")

# Use implyr to answer the question:
# How many riders and how many student riders have their
# birthday today, broken down by the decade of their
# birth, with the result returned in order by decade:

current_month <- as.integer(format(Sys.Date(), "%m"))
current_day <- as.integer(format(Sys.Date(), "%d"))

riders %>% 
  select(birth_date, student) %>%
  mutate(
    birth_month = month(birth_date),
    birth_day = day(birth_date),
    birth_year = year(birth_date),
    birth_decade = paste0(
      as.character(floor(birth_year / 10) * 10), ("s")
    )
  ) %>% 
  filter(
    birth_month == current_month,
    birth_day == current_day
  ) %>% 
  group_by(birth_decade) %>% 
  summarize(
    n = as.integer(n()),
    students = sum(as.numeric(student))
  ) %>%
  arrange(birth_decade)

dbDisconnect(impala)
