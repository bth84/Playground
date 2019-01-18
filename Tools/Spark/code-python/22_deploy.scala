/* # Saving and Loading Machine Learning Pipelines */

/* Copyright © 2010–2018 Cloudera. All rights reserved.
   Not to be reproduced or shared without prior written 
   consent from Cloudera. */

/*
In this module we demonstrate how to load and use the Pipeline and
PipelineModel objects that we saved in our Python session.  In addition, we
give you a glimpse of the Spark Scala API.  You will find that it looks quite
similar to the Python API.
*/

/*
**Note**: // comments do not render Markdown.
*/


/*
## Setup
*/

/*
In a Scala session, the SparkSession is created for us:
*/
spark

/*
Note that Spark is running in yarn client mode by default:
*/
spark.conf.get("spark.master")
spark.conf.get("spark.submit.deployMode")


/*
## Load the data
*/

/*
Let us read the enhanced ride data from HDFS:
*/
val rides = spark.read.parquet("/duocar/joined_all/")

/*
**Note:** The Spark keyword `val` indicates that `rides` is an immutable
object.
*/


/*
## Load and apply a PipelineModel object
*/
  
/*
If we simply want to apply our existing classifier to new data, then we load
and use our PipelineModel instance:
*/
import org.apache.spark.ml.PipelineModel
val pipelineModel = PipelineModel.load("models/pipeline_model")

/*
Use the `transform` method to apply the pipeline model to our new DataFrame:
*/
val classified = pipelineModel.transform(rides)

/*
**Important:** The input DataFrame must include the required columns:
*/
classified.printSchema()

/*
Use the `persist` method to cache our classified DataFrame in (worker) memory:
*/
classified.persist()

/*
Compute the confusion matrix:
*/
classified.
  groupBy("prediction").
  pivot("five_star_rating").
  count().
  show()

/*
**Note**: Scala does not provide a `crosstab` method, so we use the `pivot`
method instead.
*/

/*
**Note**: We must put the dots at the end of the line rather than the
beginning of the line.
*/

/*
Compute the classifier accuracy using Spark DataFrame operations:
*/
import org.apache.spark.sql.functions.col
val total = classified.count()
val correct = classified.filter(col("prediction") === col("five_star_rating")).count()
println(s"The accuracy of our classifier is ${correct.toFloat / total}.")

/*
**Note:** We have to use `===` to compare column expressions in Scala.
*/

/*
Compute the classifier accuracy using `MulticlassClassificationEvaluator`:
*/
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
val evaluator = new MulticlassClassificationEvaluator().
  setPredictionCol("prediction").
  setLabelCol("five_star_rating").
  setMetricName("accuracy")
println(s"The accuracy of our classifier is ${evaluator.evaluate(classified)}.")

/*
Unpersist our DataFrame:
*/
classified.unpersist()


/*
## Load and apply a Pipeline object
*/

/*
If instead we want to rebuild our decision tree classifier on new data, then we
load and use our Pipeline instance:
*/
import org.apache.spark.ml.Pipeline
val pipeline = Pipeline.load("models/pipeline")

/*
Use the `fit` method to retrain the classifier on new data:
*/
val pipelineModel = pipeline.fit(rides)

/*
The pipeline stages are stored in the `stages` attribute:
*/
pipelineModel.stages

/*
Access the decision tree classifier using the `stages` attribute:
*/
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
val classifier = pipelineModel.stages(5).asInstanceOf[DecisionTreeClassificationModel]

/*
Use the `toDebugString` method to print the decision tree classifier:
*/
classifier.toDebugString

/*
This simple decision tree classifier implements the following rules: If the
ride does not has a review, then predict a five-star rating; otherwise, predict
a non-five-star rating.
*/


/*
## Exercises

None
*/


/*
## References

[Spark Scala API](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.package)
*/

  
/*
## Cleanup

Stop the SparkSession:
```scala
spark.stop()
```
*/
