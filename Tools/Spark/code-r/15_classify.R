# # Classification modeling with sparklyr

# Copyright © 2010–2018 Cloudera. All rights reserved.
# Not to be reproduced or shared without prior written 
# consent from Cloudera.

# This module demonstrates how to fit and examine a
# binary classification model using sparklyr's machine
# learning functions, in addition to feature transformers
# and other sparklyr functions.


# ## Setup

library(sparklyr)
library(dplyr)
config <- spark_config()
config$spark.driver.host <- Sys.getenv("CDSW_IP_ADDRESS")
spark <- spark_connect(
  master = "local",
  app_name = "classify",
  config = config
)

rides <- spark_read_parquet(
  sc = spark,
  name = "rides",
  path = "/duocar/joined/"
)


# ## Prepare and split data

# In the regression module we treated `star_rating` as a 
# continuous variable. However, `star_rating` is actually 
# an ordered categorical variable. Rather than try to 
# predict each value, let us see if we can distinguish 
# between five-star and non-five-star ratings.

# Filter out cancelled rides, add a boolean variable 
# `reviewed` indicating whether the rider left a review,
# add an integer variable `five_star` that takes values
# one or zero indicating whether the ride has a five-star 
# review, and apply the string indexer and one-hot encoder
# to the vehicle color column. Finally, randomly split the 
# data into a training set (70%) and a test set (30%).

samples <- rides %>%
  filter(!cancelled) %>%
  mutate(
    reviewed = !is.na(review),
    five_star = as.integer(star_rating == 5)
  ) %>%
  ft_string_indexer(
    input.col = "vehicle_color",
    output.col = "vehicle_color_index"
  ) %>%
  ft_one_hot_encoder(
    input.col = "vehicle_color_index",
    output.col = "vehicle_color_code"
  ) %>% 
  sdf_partition(train = 0.7, test = 0.3)


# ## Specify and fit the logistic regression model

# Now fit a logistic regression model to the data, to try to
# predict five-star ratings based on whether or not the rider 
# wrote a review, the vehicle year, and the vehicle color.

model <- samples$train %>%
  ml_logistic_regression(
    five_star ~ reviewed + vehicle_year + vehicle_color_code
  )


# ## Examine the linear regression model

# You can print the model object:

model

# Or a richer summary of the model object:

summary(model)

# You can extract specific fields from the model object 
# by name:

names(model)


# ## Generate predictions using the model

# Use the `sdf_predict()` function to generate predictions
# for the test set using the model.

pred <- model %>% 
  sdf_predict(samples$test)


# ### Evaluate the logistic regression model

# Calculate the area under the ROC curve (the AUC) using
# the predictions from test set.

pred %>%
  ml_binary_classification_eval(
    "five_star",
    "prediction"
  )


# ## Cleanup

spark_disconnect(spark)
