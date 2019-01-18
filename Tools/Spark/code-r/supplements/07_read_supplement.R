# # Reading and Writing Data

# Copyright © 2010–2018 Cloudera. All rights reserved.
# Not to be reproduced or shared without prior written 
# consent from Cloudera.


# ## Contents

# * Reading from and writing to a SQLite database


# ## Setup

library(sparklyr)
library(dplyr)


# ## Create a SparkSession

spark <- spark_connect(master = "local", app_name = "read_supplement")


# ## Reading from and writing to a SQLite database

# Add the JDBC driver to the Spark driver and executor Java classpaths by
# adding the following line to the `spark-defaults.conf` file:
# ```
# spark.jars /home/cdsw/resources/sqlite-jdbc-3.23.1.jar
# ```

# Specify the SQLite database parameters:
url <- "jdbc:sqlite:/home/cdsw/data/duocar.db"
driver <- "org.sqlite.JDBC"

# Use the `spark_read_jdbc()` function to read from a SQLite database:
scientists <- spark_read_jdbc(
    sc = spark,
    name = "scientists_table",
    options = list(url = url, dbtable = "data_scientists", driver = driver)
)
scientists

offices <- spark_read_jdbc(
    sc = spark,
    name = "offices_table",
    options = list(url = url, dbtable = "offices", driver = driver)
)
offices

# Use the `spark_write_jdbc()` function to write to a SQLite database:
scientists_enhanced <- scientists %>% left_join(offices, by = "office_id") %>% select(-office_id)
scientists_enhanced
scientists_enhanced %>% spark_write_jdbc(
    name = "scientists_enhanced", 
    mode = "overwrite",
    options = list(url = url, driver = driver)
)


# ## Stop the SparkSession

spark_disconnect(spark)
