# # Working with Machine Learning Pipelines

# Copyright © 2010–2018 Cloudera. All rights reserved.
# Not to be reproduced or shared without prior written 
# consent from Cloudera.


# ## Setup

library(sparklyr)
library(dplyr)


# ## Connect to Spark

spark <- spark_connect(
  master = "local",
  app_name = "pipeline"
) 


# ## Read the enhanced ride data from HDFS

rides <- spark_read_parquet(
  sc = spark,
  name = "rides_table",
  path = "/duocar/joined/"
)


# ## Create the train and test data

splits <- rides %>% sdf_partition(train = 0.7, test = 0.3, seed = 12345)


# ## Method 1: Specify the entire pipeline

pipeline <- ml_pipeline(spark) %>%
  ft_sql_transformer("select * from __THIS__ where cancelled = 0") %>%
  ft_sql_transformer("select *, review is not null as reviewed from __THIS__") %>%
  ft_string_indexer(input_col="vehicle_color", output_col="vehicle_color_indexed") %>%
  ft_one_hot_encoder(input_col="vehicle_color_indexed", output_col="vehicle_color_encoded")

pipeline

# ## Method 2: Specify the individual pipeline stages

# Filter out the cancelled rides:
filterer <- spark %>% ft_sql_transformer("select * from __THIS__ where cancelled = 0")

# Generate the reviewed feature:
extractor <- spark %>% ft_sql_transformer("select *, review is not null as reviewed from __THIS__")

# Index the vehicle color feature:
indexer <- spark %>% ft_string_indexer(input_col="vehicle_color", output_col="vehicle_color_indexed")

# Encode the vehicle color feature:
encoder <- spark %>% ft_one_hot_encoder(input_col="vehicle_color_indexed", output_col="vehicle_color_encoded")


# ## Specify the pipeline

pipeline <- ml_pipeline(filterer, extractor, indexer, encoder)
pipeline
class(pipeline)
names(pipeline)

# ## Fit the pipeline model

pipeline_model <- pipeline %>% ml_fit(splits$train)
pipeline_model
class(pipeline_model)
names(pipeline_model)


# ## Inspect the pipeline model

pipeline_model$stages
indexer_model <- pipeline_model$stages[[3]]
class(indexer_model)
names(indexer_model)


# ## Save and load the pipeline model

pipeline_model %>% ml_save("models/pipemod", overwrite=TRUE)

# **Note:** This does not work in sparklyr 0.8.4:
# ``` r
# pipeline_model_loaded <- spark %>% ml_load("models/pipemod")
# ```


# ## Apply the pipeline model

transformed <- pipeline_model %>% ml_transform(splits$test)
transformed %>% glimpse()


# ## Disconnect from Spark

spark_disconnect(spark)
