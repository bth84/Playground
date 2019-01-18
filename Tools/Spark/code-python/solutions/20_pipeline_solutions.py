# # Working with Machine Learning Pipelines - Solutions

# Copyright © 2010–2018 Cloudera. All rights reserved.
# Not to be reproduced or shared without prior written 
# consent from Cloudera.


# ## Introduction

# In the previous modules we have established a workflow in which we load some
# data; preprocess this data; extract, transform, and select features; and fit
# and evaluate a machine learning model.  In this module we show how we can
# encapsulate this workflow into a [Spark MLlib
# Pipeline](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#module-pyspark.ml)
# that we can reuse in our development process or production environment.


# ## Setup

# Import useful packages, modules, classes, and functions:
from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Create a SparkSession

from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local").appName("pipeline_solutions").getOrCreate()


# ## Load the data

# Read the enhanced ride data from HDFS:
rides = spark.read.parquet("/duocar/joined_all/")


# ## Create the train and test DataFrames

# Create the train and test DataFrames *before* specifying the pipeline:
(train, test) = rides.randomSplit([0.7, 0.3], 12345)


# ## Specify the pipeline stages

# A *Pipeline* is a sequence of stages that implement a data engineering or
# machine learning workflow.  Each stage in the pipeline is either a
# *Transformer* or an *Estimator*.  Recall that a
# [Transformer](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.Transformer)
# takes a DataFrame as input and returns a DataFrame as output.  Recall that an
# [Estimator](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.Estimator)
# takes a DataFrame as input and returns a Transformer (e.g., model) as output.
# We begin by specifying the stages in our machine learning workflow.

# Filter out the cancelled rides:
from pyspark.ml.feature import SQLTransformer
filterer = SQLTransformer(statement="SELECT * FROM __THIS__ WHERE cancelled == 0")

# Generate the reviewed feature:
extractor = SQLTransformer(statement="SELECT *, review IS NOT NULL AS reviewed FROM __THIS__")

# Index `vehicle_color`:
from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol="vehicle_color", outputCol="vehicle_color_indexed")

# Create dummy variables for `vehicle_color_indexed`:
from pyspark.ml.feature import OneHotEncoder
encoder = OneHotEncoder(inputCol="vehicle_color_indexed", outputCol="vehicle_color_encoded")

# Select and assemble the features:
from pyspark.ml.feature import VectorAssembler
features = ["reviewed", "vehicle_year", "vehicle_color_encoded", "CloudCover"]
assembler = VectorAssembler(inputCols=features, outputCol="features")

# Specify the estimator (i.e., classification algorithm):
from pyspark.ml.classification import RandomForestClassifier
classifier = RandomForestClassifier(featuresCol="features", labelCol="star_rating")
print(classifier.explainParams())

# Specify the hyperparameter grid:
from pyspark.ml.tuning import ParamGridBuilder
maxDepthList = [5, 10, 20]
numTreesList = [20, 50, 100]
subsamplingRateList = [0.5, 1.0]
paramGrid = ParamGridBuilder() \
  .addGrid(classifier.maxDepth, maxDepthList) \
  .addGrid(classifier.numTrees, numTreesList) \
  .addGrid(classifier.subsamplingRate, subsamplingRateList) \
  .build()

# Specify the evaluator:
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(labelCol="star_rating", metricName="accuracy")

# **Note:** We are treating `star_rating` as a multiclass label.

# Specify the validator:
from pyspark.ml.tuning import TrainValidationSplit
validator = TrainValidationSplit(estimator=classifier, estimatorParamMaps=paramGrid, evaluator=evaluator)


# ## Specify the pipeline 

# A
# [Pipeline](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.Pipeline)
# itself is an `Estimator`:
from pyspark.ml import Pipeline
stages = [filterer, extractor, indexer, encoder, assembler, validator]
pipeline = Pipeline(stages=stages)


# ## Fit the pipeline model

# The `fit` method produces a
# [PipelineModel](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.PipelineModel),
# which is a `Transformer`:
%time pipeline_model = pipeline.fit(train)


# ## Query the PipelineModel

# Access the stages of a `PipelineModel` using its `stages` attribute:
pipeline_model.stages

# We can access each stage as necessary:
indexer_model = pipeline_model.stages[2]
indexer_model.labels

# The best model is an instance of the
# [RandomForestClassificationModel](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.RandomForestClassificationModel)
# class:
validator_model = pipeline_model.stages[5]
type(validator_model.bestModel)

# Get the best hyperparameters:
validator_model.bestModel._java_obj.getMaxDepth()
validator_model.bestModel.getNumTrees
validator_model.bestModel._java_obj.getSubsamplingRate()

# **Note:** We accessed the values for `maxDepth` and `subsamplingRate`
# from the underlying Java object.

# Plot feature importances:
def plot_feature_importances(fi):
  fi_array = fi.toArray()
  plt.figure()
  sns.barplot(range(len(fi_array)), fi_array)
  plt.title("Feature Importances")
  plt.xlabel("Feature")
  plt.ylabel("Importance")
plot_feature_importances(validator_model.bestModel.featureImportances)


# ## Apply the PipelineModel

# We can use the `transform` method of our `PipelineModel` to apply it to our
# test DataFrame:
classified = pipeline_model.transform(test)
classified.printSchema()

# Let us generate a confusion matrix to see how well we did:
classified \
  .crosstab("prediction", "star_rating") \
  .orderBy("prediction_star_rating") \
  .show()

# Create the baseline prediction (always predict five-star rating):
from pyspark.sql.functions import lit
classified_with_baseline = classified.withColumn("prediction_baseline", lit(5.0))

# Evaluate the baseline model and the random forest model::
evaluator = MulticlassClassificationEvaluator(labelCol="star_rating", metricName="accuracy")
evaluator.setPredictionCol("prediction_baseline").evaluate(classified_with_baseline)
evaluator.setPredictionCol("prediction").evaluate(classified_with_baseline)

# We have more work to do!


# ## Exercises

# (1) Import the
# [RFormula](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.RFormula)
# class from the `pyspark.ml.feature` module.

from pyspark.ml.feature import RFormula

# (2) Create an instance of the `RFormula` class with the R formula
# `star_rating ~ reviewed + vehicle_year + vehicle_color`.

rformula = RFormula(formula = "star_rating ~ reviewed + vehicle_year + vehicle_color")

# (3) Specify a pipeline consisting of the `filterer`, `extractor`, and the
# RFormula instance specified above.

pipeline = Pipeline(stages=[filterer, extractor, rformula])

# (4) Fit the pipeline on the `train` DataFrame.

pipeline_model = pipeline.fit(train)

# (5) Use the `save` method to save the pipeline model to the
# `models/pipeline_model` directory in HDFS.

pipeline_model.write().overwrite().save("models/pipeline_model")

# (6) Import the `PipelineModel` class from the `pyspark.ml` package.

from pyspark.ml import PipelineModel

# (7) Use the `load` method of `PipelineModel` class to load the saved pipeline
# model.

pipeline_model_loaded = PipelineModel.load("models/pipeline_model")
                                           
# (8) Apply the loaded pipeline model to the test DataFrame and examine the
# resulting DataFrame.

test_transformed = pipeline_model.transform(test)
test_transformed.printSchema()
test_transformed.select("features", "label").show(truncate=False)


# ## References

# [Spark Documentation - ML
# Pipelines](http://spark.apache.org/docs/latest/ml-pipeline.html)

# [Spark Python API - pyspark.ml
# package](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html)


# ## Stop the SparkSession

spark.stop()
