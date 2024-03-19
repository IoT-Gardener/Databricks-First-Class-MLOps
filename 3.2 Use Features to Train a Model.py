# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: left; line-height: 0; padding-top: 9px; padding-left:150px">
# MAGIC   <img src="https://static1.squarespace.com/static/5bce4071ab1a620db382773e/t/5d266c78abb6d10001e4013e/1562799225083/appliedazuredatabricks3.png" alt="Databricks Learning" style="width: 600px; height: 163px">
# MAGIC </div>
# MAGIC
# MAGIC # *Feature Stores*. Presented by <a href="www.advancinganalytics.co.uk">Advancing Analytics</a>

# COMMAND ----------

# MAGIC %md ## Train a model
# MAGIC
# MAGIC This section illustrates how to train a model using the pickup and dropoff features stored in Feature Store. It trains a LightGBM model to predict taxi fare.

# COMMAND ----------

# MAGIC %md ### Helper functions

# COMMAND ----------

from pyspark.sql import *
from pyspark.sql.functions import current_timestamp, lit
from pyspark.sql.types import IntegerType
import math
from datetime import timedelta
import mlflow.pyfunc


def rounded_unix_timestamp(dt, num_minutes=15):
    """
    Ceilings datetime dt to interval num_minutes, then returns the unix timestamp.
    """
    nsecs = dt.minute * 60 + dt.second + dt.microsecond * 1e-6
    delta = math.ceil(nsecs / (60 * num_minutes)) * (60 * num_minutes) - nsecs
    return int((dt + timedelta(seconds=delta)).timestamp())


rounded_unix_timestamp_udf = udf(rounded_unix_timestamp, IntegerType())


def rounded_taxi_data(taxi_data_df):
    # Round the taxi data timestamp to 15 and 30 minute intervals so we can join with the pickup and dropoff features
    # respectively.
    taxi_data_df = (
        taxi_data_df.withColumn(
            "rounded_pickup_datetime",
            rounded_unix_timestamp_udf(taxi_data_df["tpep_pickup_datetime"], lit(15)),
        )
        .withColumn(
            "rounded_dropoff_datetime",
            rounded_unix_timestamp_udf(taxi_data_df["tpep_dropoff_datetime"], lit(30)),
        )
        .drop("tpep_pickup_datetime")
        .drop("tpep_dropoff_datetime")
    )
    taxi_data_df.createOrReplaceTempView("taxi_data")
    return taxi_data_df
  
def get_latest_model_version(model_name):
  latest_version = 1
  mlflow_client = MlflowClient()
  for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
    version_int = int(mv.version)
    if version_int > latest_version:
      latest_version = version_int
  return latest_version

# COMMAND ----------

# MAGIC %md ### Read taxi data for training

# COMMAND ----------

raw_data = spark.read.format("delta").load("/databricks-datasets/nyctaxi-with-zipcodes/subsampled")
taxi_data = rounded_taxi_data(raw_data)

display(taxi_data)

# COMMAND ----------

from databricks import feature_store
from databricks.feature_store import FeatureLookup

fs = feature_store.FeatureStoreClient()

pickup_features_table = "feature_store_taxi_example.trip_pickup_features"
dropoff_features_table = "feature_store_taxi_example.trip_dropoff_features"

pickup_feature_lookups = [
   FeatureLookup( 
     table_name = pickup_features_table,
     feature_names = ["mean_fare_window_1h_pickup_zip", "count_trips_window_1h_pickup_zip"],
     lookup_key = ["pickup_zip", "rounded_pickup_datetime"],
   ),
]

dropoff_feature_lookups = [
   FeatureLookup( 
     table_name = dropoff_features_table,
     feature_names = ["count_trips_window_30m_dropoff_zip", "dropoff_is_weekend"],
     lookup_key = ["dropoff_zip", "rounded_dropoff_datetime"],
   ),
]

# COMMAND ----------

# MAGIC %md ### Create a Training Dataset
# MAGIC
# MAGIC When `fs.create_training_set(..)` is invoked below, the following steps will happen:
# MAGIC
# MAGIC 1. A `TrainingSet` object will be created, which will select specific features from Feature Store to use in training your model. Each feature is specified by the `FeatureLookup`'s created above. 
# MAGIC
# MAGIC 1. Features are joined with the raw input data according to each `FeatureLookup`'s `lookup_key`.
# MAGIC
# MAGIC The `TrainingSet` is then transformed into a DataFrame to train on. This DataFrame includes the columns of taxi_data, as well as the features specified in the `FeatureLookups`.

# COMMAND ----------

# End any existing runs (in the case this notebook is being run for a second time)
mlflow.end_run()

# Start an mlflow run, which is needed for the feature store to log the model
mlflow.start_run() 

# Since the rounded timestamp columns would likely cause the model to overfit the data 
# unless additional feature engineering was performed, exclude them to avoid training on them.
exclude_columns = ["rounded_pickup_datetime", "rounded_dropoff_datetime"]

# Create the training set that includes the raw input data merged with corresponding features from both feature tables
training_set = fs.create_training_set(
  taxi_data,
  feature_lookups = pickup_feature_lookups + dropoff_feature_lookups,
  label = "fare_amount",
  exclude_columns = exclude_columns
)

# Load the TrainingSet into a dataframe which can be passed into sklearn for training a model
training_df = training_set.load_df()

# COMMAND ----------

# Display the training dataframe, and note that it contains both the raw input data and the features from the Feature Store, like `dropoff_is_weekend`
display(training_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Train a LightGBM model on the data returned by `TrainingSet.to_df`, then log the model with `FeatureStoreClient.log_model`. The model will be packaged with feature metadata.

# COMMAND ----------

from sklearn.model_selection import train_test_split
from mlflow.tracking import MlflowClient
import lightgbm as lgb
import mlflow.lightgbm
from mlflow.models.signature import infer_signature

features_and_label = training_df.columns

# Collect data into a Pandas array for training
data = training_df.toPandas()[features_and_label]

train, test = train_test_split(data, random_state=123)
X_train = train.drop(["fare_amount"], axis=1)
X_test = test.drop(["fare_amount"], axis=1)
y_train = train.fare_amount
y_test = test.fare_amount

mlflow.lightgbm.autolog()

train_lgb_dataset = lgb.Dataset(X_train, label=y_train.values)
test_lgb_dataset = lgb.Dataset(X_test, label=y_test.values)

param = {"num_leaves": 32, "objective": "regression", "metric": "rmse"}
num_rounds = 100

# Train a lightGBM model
model = lgb.train(
  param, train_lgb_dataset, num_rounds
)

# COMMAND ----------

# Log the trained model with MLflow and package it with feature lookup information. 
fs.log_model(
  model,
  artifact_path="model_packaged",
  flavor=mlflow.lightgbm,
  training_set=training_set,
  registered_model_name="taxi_example_fare_packaged"
)

# COMMAND ----------

# MAGIC %md ## Scoring: batch inference

# COMMAND ----------

# MAGIC %md Suppose another data scientist now wants to apply this model to a different batch of data.

# COMMAND ----------

new_taxi_data = rounded_taxi_data(raw_data)

# COMMAND ----------

# MAGIC %md Display the data to use for inference, reordered to highlight the `fare_amount` column, which is the prediction target.

# COMMAND ----------

cols = ['fare_amount', 'trip_distance', 'pickup_zip', 'dropoff_zip', 'rounded_pickup_datetime', 'rounded_dropoff_datetime']
new_taxi_data_reordered = new_taxi_data.select(cols)
display(new_taxi_data_reordered)

# COMMAND ----------

# MAGIC %md
# MAGIC Use the `score_batch` API to evaluate the model on the batch of data, retrieving needed features from FeatureStore. 

# COMMAND ----------

# Get the model URI
latest_model_version = get_latest_model_version("taxi_example_fare_packaged")
model_uri = f"models:/taxi_example_fare_packaged/{latest_model_version}"

# Call score_batch to get the predictions from the model
with_predictions = fs.score_batch(model_uri, new_taxi_data)

# COMMAND ----------

# MAGIC %md <img src="https://docs.databricks.com/_static/images/machine-learning/feature-store/taxi_example_score_batch.png"/>

# COMMAND ----------

# MAGIC %md ### View the taxi fare predictions
# MAGIC
# MAGIC This code reorders the columns to show the taxi fare predictions in the first column.  Note that the `predicted_fare_amount` roughly lines up with the actual `fare_amount`, although more data and feature engineering would be required to improve the model accuracy.
# MAGIC

# COMMAND ----------

import pyspark.sql.functions as func

cols = ['prediction', 'fare_amount', 'trip_distance', 'pickup_zip', 'dropoff_zip', 
        'rounded_pickup_datetime', 'rounded_dropoff_datetime', 'mean_fare_window_1h_pickup_zip', 
        'count_trips_window_1h_pickup_zip', 'count_trips_window_30m_dropoff_zip', 'dropoff_is_weekend']

with_predictions_reordered = (
    with_predictions.select(
        cols,
    )
    .withColumnRenamed(
        "prediction",
        "predicted_fare_amount",
    )
    .withColumn(
      "predicted_fare_amount",
      func.round("predicted_fare_amount", 2),
    )
)

display(with_predictions_reordered)
