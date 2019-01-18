# # Connecting to Spark

# Copyright © 2010–2018 Cloudera. All rights reserved.
# Not to be reproduced or shared without prior written 
# consent from Cloudera.

# ## Background

# There are two R packages that provide R interfaces to 
# Apache Spark:
# **[SparkR](https://spark.apache.org/docs/latest/sparkr.html)**
# and
# **[sparklyr](https://spark.rstudio.com)**.
# SparkR is part of the Apache Spark project; it was created by
# the same people who create Apache Spark.
# sparklyr was created by [RStudio](https://www.rstudio.com).

# SparkR does not follow established idioms of R programming.
# It uses a syntax that is unfamiliar to the typical R user.

# sparklyr follows the idioms of the popular
# [tidyverse](https://www.tidyverse.org) collection of R
# packages. As you will see in later sections, sparklyr
# uses a syntax that is familiar to many R users.

# It has been our experience at Cloudera that data scientists
# who use R are typically more comfortable and more productive 
# using sparklyr. Cloudera generally encourages R users to 
# use sparklyr instead of SparkR.


# ## Installing sparklyr

# Install the sparklyr package from CRAN (if it is not
# already installed). This might take several minutes,
# because the package must be built from source code:

if(!"sparklyr" %in% rownames(installed.packages())) {
  install.packages("sparklyr")
}

# Load sparklyr:

library(sparklyr)


# ## Connecting to a local Spark instance

# Use the `spark_connect()` function to create a Spark connection

#```r
#spark <- spark_connect(master = "local", app_name = "spark (local)")
#```

# The argument `master` specifies which Spark instance to connect to.
# Use `master = "local"` to connect to a local Spark instance.

# The optional `app_name` argument changes the default name of 
# the Spark session.

# In some environments, those are the only two arguments you need
# to pass to `spark_connect()`.
# But when using the local Spark instance from CDSW,
# it's also necessary to set a Spark configuration parameter.

# To set configuration parameters, first use the `spark_config()`
# function to retrieve the default configuration:

config <- spark_config()

# You can view the default configuration. It is a named list:

config

# To use local Spark from CDSW, it is necessary to modify this
# configuration list to set the Spark configuration property 
# `spark.driver.host` to the IP address of the CDSW engine 
# container. This is stored in an environment variable 
# named `CDSW_IP_ADDRESS`.

config$spark.driver.host <- Sys.getenv("CDSW_IP_ADDRESS")

# Then pass this modified `config` object as the `config` argument
# to the `spark_connect()` function:

spark <- spark_connect(
  master = "local",
  app_name = "spark (local)",
  config = config
)

# Now your Spark connection is ready to be used.


# ## Using the Spark connection

# To use the connection to Spark, you pass the Spark connection 
# object to various sparklyr functions. For example:

# Use the `spark_version()` function to get the version of Spark.

spark_version(spark)

# Create a Spark DataFrame by creating an R data frame and
# copying it to Spark, then view it:

df <- sdf_copy_to(spark, data.frame(team = c("brian", "glynn", "ian")))
df

# See if the connection to Spark is still open

spark_connection_is_open(spark)

# Disconnect from Spark

spark_disconnect(spark)

# See if the connection to Spark is still open

spark_connection_is_open(spark)


# ## Connecting to a Spark cluster via YARN

# Wait 10 seconds before reconnecting, to work around 
# sparklyr issue [#919](https://github.com/rstudio/sparklyr/issues/919).

Sys.sleep(10)

# To connect to Spark on YARN, use `master = "yarn"`:

spark <- spark_connect(master = "yarn", app_name = "spark (yarn)")

# If this command fails, you may need to restart your R session,
# load sparklyr, and run it again.


# ## Viewing the Spark Job UI

# Create a simple Spark job to exercise the Spark Job UI:

df <- sdf_len(spark, 10000000)
df

# Access the Spark UI through the Spark UI button in the upper right
# menu in CDSW.

# Disconnect from Spark

spark_disconnect(spark)
