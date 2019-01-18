# # The RFormula Transformer

# Copyright © 2010–2018 Cloudera. All rights reserved.
# Not to be reproduced or shared without prior written 
# consent from Cloudera.

# The RFormula Transformer will look familiar to R users.  It can be useful in
# some cases.  In this module we demonstrate the basic functionality of the
# RFormula Transformer.

# **Note:** RFormula is experimental as of Spark 2.2.

from pyspark.sql import SparkSession
from pyspark.ml.feature import RFormula

spark = SparkSession.builder.master("local").appName("rformula").getOrCreate()

rides = spark.read.parquet("/duocar/joined/")

formula = "star_rating ~ vehicle_year + vehicle_color"

rformula = RFormula(formula=formula)

rformula_model = rformula.fit(rides)

transformed = rformula_model.transform(rides)

transformed.select("vehicle_year", "vehicle_color", "star_rating", "features", "label").show(truncate=False)

# **Question:** How do we get the mapping for `vehicle_color`?

spark.stop()

# ## References

# [RFormula class](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.RFormula)

# [RFormulaModel class](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.RFormulaModel)

# [R formula](http://stat.ethz.ch/R-manual/R-patched/library/stats/html/formula.html)
