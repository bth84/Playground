# # Clustering with sparklyr

# Copyright © 2010–2018 Cloudera. All rights reserved.
# Not to be reproduced or shared without prior written 
# consent from Cloudera.

# This module demonstrates how to perform k-means clustering
# using sparklyr.

# Scenario: The Fargo metropolitan area is home to two
# major universities: North Dakota State University and 
# Minnesota State University Moorhead. DuoCar wants to
# send targeted promotions to student riders containing
# different content based on which university the student
# attends. ("Go Bison!" or "Go Dragons!")
# But DuoCar does not ask student riders which university 
# they attend. Can you develop a simple machine learning
# model to predict which university a student attends
# based on the data DuoCar has?


# ## Setup

library(sparklyr)
library(dplyr)
config <- spark_config()
config$spark.driver.host <- Sys.getenv("CDSW_IP_ADDRESS")
spark <- spark_connect(
  master = "local",
  app_name = "cluster",
  config = config
)

riders <- spark_read_parquet(
  sc = spark,
  name = "riders",
  path = "/duocar/clean/riders/"
)


# ## K-means clustering

# Cluster student riders into two groups based on their
# home coordinates

student_riders <- riders %>% filter(student)


# ### Specify and fit the k-means clustering model:

model <- student_riders %>% 
  ml_kmeans(
    centers = 2,
    features = c("home_lat", "home_lon")
  )


# ###  Examine the k-means clustering model

# Print the model object:

model

# See the names of the elements in the k-means model 
# object, and look at some of those elements:

names(model)

model$cost

model$centers

# See which records the model assigns to which clusters

predictions <- model %>% 
  sdf_predict(student_riders)

predictions %>% select(prediction)

predictions_df <- predictions %>% collect()

# It looks like the prediction values of 0 represent the
# cluster toward the northwest (larger positive latitude, 
# larger negative longitude) and the prediction values of
# 1 represent the cluster toward the southeast.

# Let's visualize the model output to make this clearer.


# ### Create a map visualization

# Load the leaflet package

library(leaflet)

# Use `colorFactor()` to create a function that maps the 
# cluster IDs to the two university colors:

pal <- colorFactor(
  palette = c("#006633", "#A6192E"), # red, green
  domain = c(0, 1) 
)

# Create the map visualization, with separate layers
# for the students' home coordinates and the cluster 
# centers, using matching colors:

leaflet(predictions_df) %>%
  addTiles() %>%
  addCircleMarkers(
    lng = ~home_lon,
    lat = ~home_lat,
    radius = 5,
    stroke = FALSE,
    color = ~pal(prediction)
  ) %>% 
  addCircleMarkers(
    lng = model$centers$home_lon,
    lat = model$centers$home_lat,
    radius = 15,
    fill = FALSE,
    color = c("#006633","#A6192E"),
  )


# ## Exercises

# Use the function `sdf_predict()` to predict which 
# universities the students with these five pairs of home
# coordinates attend:

new_data <- tribble(
  ~home_lat,~home_lon,
  46.869668,-96.765317,
  46.887471,-96.792376,
  46.887082,-96.800197,
  46.860673,-96.777554,
  46.885192,-96.783540
)

# Plot these points on a map, colored by the predicted
# university. Hint: Use the `opacity` and/or `fillOpacity`
# arguments to make the points more easily visible.

# There is a third notable (but smaller) university in the
# Fargo metropolitan area: Concordia College. (Go Cobbers!)
# Try changing the number of clusters from two to three
# and re-fitting the k-means model to the data.
# See if the third cluster identifies Concordia College
# students. If not, how else could you use k-means 
# clustering to identify Concordia College students? Hint:
# you could apply the k-means model twice.

# Concordia College's color is maroon (#701C45).


# ## Cleanup

spark_disconnect(spark)
