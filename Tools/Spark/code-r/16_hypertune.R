# # Tuning Algorithm Hyperparameters Using Grid Search

# Copyright © 2010–2018 Cloudera. All rights reserved.
# Not to be reproduced or shared without prior written 
# consent from Cloudera.


# ## Introduction

# Most machine learning algorithms have a set of user-specified parameters that
# govern the behavior of the algorithm.  These parameters are called
# *hyperparameters* to distinguish them from the model parameters such as the
# intercept and coefficients in linear and logistic regression.  In this module
# we show how to use grid search and cross validation in Spark MLlib to
# determine a reasonable regularization parameter for [$l1$ lasso linear
# regression](https://en.wikipedia.org/wiki/Lasso_%28statistics%29).


# ## Setup

library(sparklyr)
library(dplyr)


# ## Create a SparkSession

spark <- spark_connect(
  master = "local",
  app_name = "hypertune",
)


# ## Read the enhanced ride data

rides <- spark_read_parquet(
  sc = spark,
  name = "rides_table",
  path = "data/modeling_data"
)
rides %>% glimpse()


# ## Create train and test datasets

splits <- rides %>% sdf_partition(train = 0.7, test = 0.3, seed=12345)


# ## Requirements for hyperparameter tuning

# We need to specify four components to perform hyperparameter tuning using
# grid search:
# * Estimator (i.e. machine learning algorithm)
# * Hyperparameter grid
# * Evaluator
# * Validation method


# ## Specify the estimator

estimator <- spark %>% ml_linear_regression(features_col="features", label_col="star_rating")


# ## Specify the hyperparameter grid

param_grid <- list(
  linear_regression = list(
    elastic_net_param = c(1.0),
    reg_param = c(0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
  )
)


# ## Specify the evaluator

evaluator <- spark %>% ml_regression_evaluator(label_col="star_rating")


# ## Specify the validator (train-validation split)

tvs <- spark %>% ml_train_validation_split(estimator, param_grid, evaluator, train_ratio=0.75, seed=54321)


# ## Tuning the hyperparameters

tvs_model <- tvs %>% ml_fit(splits$train)
class(tvs_model)
names(tvs_model)


# ## Examine the validation metrics

tvs_model$validation_metrics_df


# ## Inspect the best model object

class(tvs_model$best_model)
names(tvs_model$best_model)
tvs_model$best_model$intercept
tvs_model$best_model$coefficients


# ## Inspect the best model summary object

class(tvs_model$best_model$summary)
names(tvs_model$best_model$summary)
tvs_model$best_model$summary$root_mean_squared_error


# ## Apply the best model to the test dataset

test_with_prediction <- tvs_model$best_model %>% ml_transform(splits$test)
test_with_prediction %>% glimpse()


# ## Evaluate the best model on the test dataset

evaluator %>% ml_evaluate(test_with_prediction)


# ## Stop the SparkSession

spark_disconnect(spark)
