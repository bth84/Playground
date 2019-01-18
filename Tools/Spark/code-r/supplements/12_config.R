# # Configuring sparklyr

# Copyright © 2010–2018 Cloudera. All rights reserved.
# Not to be reproduced or shared without prior written 
# consent from Cloudera.

# sparklyr supports a number of options for configuring both 
# the behavior of the package and the behavior of the 
# underlying Spark engine. These are described in the 
# [sparklyr deployment and configuration guide](https://spark.rstudio.com/articles/deployment-overview.html#configuration).
# This guide describes the `spark_config()` function seen 
# elsewhere in this course. It also describes a method for 
# specifying configuration in a config.yml file:

# ## Using config.yml

# You can configure sparklyr using a YAML script named 
# config.yml located in the current working directory.
# sparklyr uses the 
# [config](https://github.com/rstudio/config)
# package to implement this.

# You can use the config.yml file to define multiple configuration
# profiles—for example, one for local Spark and one for Spark
# on YARN.

# An example config.yml script is:

#```
#default:
#  spark.master: "local"
#  spark.driver.host: !expr Sys.getenv("CDSW_IP_ADDRESS") 
#  
#production:
#  spark.master: "yarn"
#  spark.driver.host: ""
#```

# If you save this config.yml file in the root of your CDSW file
# space (`/home/cdsw`) then you can control whether you will 
# use Spark on YARN or local Spark by setting or unsetting
# the environment variable `R_CONFIG_ACTIVE`:


# ### To use Spark on YARN (the profile named "production"):

library(sparklyr)

Sys.setenv(R_CONFIG_ACTIVE = "production")
spark <- spark_connect()

# ... use Spark on YARN ...

spark_disconnect(spark)


# ### To use local Spark (the default profile):

Sys.unsetenv("R_CONFIG_ACTIVE")
spark <- spark_connect()

# ... use local Spark ...

spark_disconnect(spark)


# Be sure to delete the config.yml file unless you will
# always use this `R_CONFIG_ACTIVE` variable, because 
# the `spark.driver.host` parameter set in the default profile
# will prevent sparklyr from connecting to Spark on YARN.

# For information about Spark configuration parameters, see the 
# [Apache Spark configuration guide](https://spark.apache.org/docs/latest/configuration.html).
