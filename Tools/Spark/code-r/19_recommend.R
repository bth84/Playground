# # Fitting and Evaluating Recommendation Systems

# Copyright © 2010–2018 Cloudera. All rights reserved.
# Not to be reproduced or shared without prior written 
# consent from Cloudera.


# ## Introduction

# * The ALS algorithm takes a DataFrame with user, item, and rating columns as input.


# ## Scenario


# ## Setup

library(sparklyr)
library(dplyr)


# ## Connect to Spark

spark <- spark_connect(master = "local", app_name = "recommend")


# ## Read the data

# **Important:** Run `19_recommend.py` before running this script.

recommendation_data <- spark_read_parquet(spark, "recommendation_data_table", "data/recommendation_data")
recommendation_data %>% print(n = 20)


# ## Create train and test data

splits <- recommendation_data %>% sdf_partition(train = 0.75, test = 0.25, seed = 12345)


# ## Specify and fit an ALS model

als_model <- splits$train %>% ml_als(
    user_col = "user",
    item_col = "artist",
    rating_col = "playcount",
    implicit_prefs = TRUE,
    cold_start_strategy = "drop",
    seed = 23456
)


# ## Examine the model

# Examine each user's weights with respect to the latent factors:
als_model$user_factors %>% glimpse()

# Examine each item's weights with respect to the latent factors:
als_model$item_factors %>% glimpse()


# ## Apply the model

test_with_predictions <- als_model %>% ml_transform(splits$test)
test_with_predictions %>% glimpse()

test_with_predictions <- als_model %>% ml_predict(splits$test)
test_with_predictions %>% glimpse()


# ## Evaluate the model

test_with_predictions %>% ml_regression_evaluator(label_col="playcount")


# ## Generate recommendations

# Recommend the top $n$ items for each user:
recommended_items <- als_model$recommend_for_all_users(5) %>%
    arrange(user) %>%
    head(5) %>%
    collect()

# Each recommendation is a list of lists of lists:
recommended_items
recommended_items$recommendations[1]

# Recommend the top $n$ users for each item:
als_model$recommend_for_all_items(5)


# ## Disconnect from Spark

spark_disconnect(spark)
