# # Saving and Loading Machine Learning Pipelines

# Copyright © 2010–2018 Cloudera. All rights reserved.
# Not to be reproduced or shared without prior written 
# consent from Cloudera.

# In this module we show how save, load, and apply machine learning pipelines.


# ## Setup

# Create a SparkSession:
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local").appName("deploy").getOrCreate()


# ## Load the data

# Read the enhanced ride data from HDFS:
rides = spark.read.parquet("/duocar/joined/")


# ## Specify the machine learning pipeline

# Remove the cancelled rides:
from pyspark.ml.feature import SQLTransformer
filterer = SQLTransformer(statement="select * from __THIS__ where cancelled = 0")

# Cast `star_rating` to double for the Binarizer:
converter = SQLTransformer(statement="select *, cast(star_rating as double) as star_rating_double from __THIS__")

# Binarize `star_rating_double`:
from pyspark.ml.feature import Binarizer
binarizer = Binarizer(inputCol="star_rating_double", outputCol="five_star_rating", threshold=4.5)

# Extract the `reviewed` feature:
extractor = SQLTransformer(statement="select *, review is not null as reviewed from __THIS__")

# Assemble the features:
from pyspark.ml.feature import VectorAssembler
selected = ["reviewed"]
assembler = VectorAssembler(inputCols=selected, outputCol="features")

# Specify the decision tree classifier:
from pyspark.ml.classification import DecisionTreeClassifier
classifier = DecisionTreeClassifier(featuresCol="features", labelCol="five_star_rating")

# Specify the pipeline:
from pyspark.ml import Pipeline
stages = [filterer, converter, binarizer, extractor, assembler, classifier]
pipeline = Pipeline(stages=stages)


# ## Save and load the machine learning pipeline

# Save the `Pipeline` instance to our local directory in HDFS:
pipeline.write().overwrite().save("models/pipeline")

# **Note**: We can use Hue to explore the saved object.

# If we do not want to overwrite existing objects, then we can use the following convenience method:
#```python
#pipeline.save("models/pipeline")
#```

# Load the Pipeline object from our local directory in HDFS:
pipeline_loaded = Pipeline.read().load("models/pipeline")

# We can also use the following convenience method:
#```python
#pipeline_loaded = Pipeline.load("models/pipeline")
#```


# ## Fit the machine learning pipeline model

pipeline_model = pipeline.fit(rides)


# ## Save and load the machine learning pipeline model

# Save the pipeline model to our local directory in HDFS:
pipeline_model.write().overwrite().save("models/pipeline_model")

# **Note**: We can use Hue to explore the saved object.

# Load the pipeline model from our local directory in HDFS:
from pyspark.ml import PipelineModel
pipeline_model_loaded = PipelineModel.read().load("models/pipeline_model")


# ## Examine and evaluate the decision tree classifier

# Extract the decision tree classifier from the `stages` attribute:
classifier_model = pipeline_model.stages[5]
type(classifier_model)

# Use the `toDebugString` attribute to print the decision tree classifier:
print(classifier_model.toDebugString)

# Use the `transform` method to apply the pipeline model to a DataFrame:
classified = pipeline_model.transform(rides)

# Use the `persist` method to cache the classified DataFrame in memory:
classified.persist()

# Examine the classified DataFrame:
classified.printSchema()

classified.select("review", "reviewed", "features").show(10)

classified.select("star_rating", "star_rating_double", "five_star_rating").show(10)

classified.select("probability", "prediction", "five_star_rating").show(10, truncate=False)

# Compute the confusion matrix:
classified \
  .crosstab("prediction", "five_star_rating") \
  .orderBy("prediction_five_star_rating") \
  .show()

# Compute baseline classifier accuracy (always predict five-star rating):
from pyspark.sql.functions import col
classified.filter(1.0 == col("five_star_rating")).count() / \
float(classified.count())

# Compute decision tree classifier accuracy:
classified.filter(col("prediction") == col("five_star_rating")).count() / \
float(classified.count())

# Unpersist the classified DataFrame:
classified.unpersist()

# We certainly have room for improvement, but let us look at how we can load
# our machine learning pipeline into a Scala workflow.


# ## Exercises

# None


# ## References

# [Spark Python API - MLReadable class](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.util.MLReadable)

# [Spark Python API - MLReader class](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.util.MLReader)

# [Spark Python API - MLWriteable class](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.util.MLWritable)

# [Spark Python API - MLWriter class](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.util.MLWriter)


# ## Cleanup

# Stop the SparkSession:
spark.stop()
