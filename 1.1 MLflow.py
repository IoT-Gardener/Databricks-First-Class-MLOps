# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: left; line-height: 0; padding-top: 9px; padding-left:150px">
# MAGIC   <img src="https://static1.squarespace.com/static/5bce4071ab1a620db382773e/t/5d266c78abb6d10001e4013e/1562799225083/appliedazuredatabricks3.png" alt="Databricks Learning" style="width: 600px; height: 163px">
# MAGIC </div>
# MAGIC
# MAGIC # *MLflow*. Presented by <a href="www.advancinganalytics.co.uk">Advancing Analytics</a>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # MLflow Demo Notebook
# MAGIC This notebook will demonstrate how to use MLflow in databricks to train a machine learning model, log the parameters & metrics, register a model, before finally preparing the model for production.
# MAGIC ## MLflow Architecture
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/mlflow-tracking.png" style="height: 400px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Helper Functions
# MAGIC Just a few helper functions to format get the data ready for model training! 

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
    """
    Round the taxi data timestamp to 15 and 30 minute intervals so we can join with the pickup and dropoff features respectively.
    """
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

# COMMAND ----------

# MAGIC %md 
# MAGIC # Load the dataset
# MAGIC Load the NYC taxi dataset which will be used for training a regression model to predict taxi fares.

# COMMAND ----------

# Load the raw NYC taxi dataset from the selection default databricks datasets. 
raw_data = spark.read.format("delta").load("/databricks-datasets/nyctaxi-with-zipcodes/subsampled")
# Use helper function to round the data to 15 and 30 minute intervals
taxi_data = rounded_taxi_data(raw_data)

# Display the dataset
display(taxi_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare the data for training
# MAGIC The most barebones steps are taken to prepare the data for training a model, splitting into a training and test set.
# MAGIC
# MAGIC We will look more at what can be done with the data in the next section on feature store.

# COMMAND ----------

from sklearn.model_selection import train_test_split

# Get the feature and label names
features_and_label = taxi_data.columns

# Collect data into a Pandas array for training
data = taxi_data.toPandas()[features_and_label]

# Split the data into train and test sets
train, test = train_test_split(data, random_state=42)

# Separate the features by dropping the label column
X_train = train.drop(["fare_amount"], axis=1)
X_test = test.drop(["fare_amount"], axis=1)

# Separate the labels by selecting only the relevant column
y_train = train.fare_amount
y_test = test.fare_amount

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train an SKLearn Random Forest Regressor
# MAGIC Here we will train a Random Forest regressor, logging the parameters and the model performance metrics to MLflow.

# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# Start mlflow autologging
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="Taxi-RF") as run:
    # Define the parameters for the random forest
    n_estimators = 100
    max_depth = 6
    max_features = 3
  
    # Create and train a random forest regresor using the predefined parameters
    rf_model = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features)
    rf_model.fit(X_train, y_train)
  
    # Get predictions for the test data
    rf_predictions = rf_model.predict(X_test)

    # Calculate performance metrics
    rf_mae = mean_absolute_error(y_test, rf_predictions)
    rf_mse = mean_squared_error(y_test, rf_predictions)

        # Log the performance metrics to mlflow
    mlflow.log_metric("Mean absolute error", rf_mae)
    mlflow.log_metric("Mean squared error", rf_mse)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train a Light GBM
# MAGIC Here we will train a second model, a Light GBM, logging the parameters and the model performance metrics to MLflow. We can then easily compare the performance of the two models to one another.

# COMMAND ----------

import lightgbm as lgb
import mlflow.lightgbm

# Start mlflow autologging
mlflow.lightgbm.autolog()

# Prepare training dataset for use in the gbm
train_lgb_dataset = lgb.Dataset(X_train, label=y_train.values)

with mlflow.start_run(run_name="Taxi-GBM") as run:
    # Define parameters for the gbm
    lgb_param = {"num_leaves": 32, "objective": "regression", "metric": "rmse"}
    lgb_num_rounds = 100

    # Train a gbm using the predefined parameters
    lgb_model = lgb.train(lgb_param, train_lgb_dataset, lgb_num_rounds)

    # Get predictions for the test data
    lgb_predictions = lgb_model.predict(X_test)

    # Calculate performance metrics
    lgb_mae = mean_absolute_error(y_test, lgb_predictions)
    lgb_mse = mean_squared_error(y_test, lgb_predictions)

    # Log the performance metrics to ML
    mlflow.log_metric("Mean absolute error", lgb_mae)
    mlflow.log_metric("Mean squared error", lgb_mse)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Querying Past Runs Programmatically
# MAGIC
# MAGIC You can query past runs programmatically in order to use this data back in Python.  The pathway to doing this is an **`MlflowClient`** object.

# COMMAND ----------

from mlflow.tracking import MlflowClient

# Create an MLflow client and search all experiments
client = MlflowClient()
display(client.search_experiments())

# COMMAND ----------

# MAGIC %md
# MAGIC You can also use `search_runs` <a href="https://mlflow.org/docs/latest/search-syntax.html" target="_blank">(documentation)</a> to find all runs for a given experiment.

# COMMAND ----------

# Get the experimnent id from the run (most recently trained model in this case)
experiment_id = run.info.experiment_id

# List all runs for given experiment
runs_df = mlflow.search_runs(experiment_id)
display(runs_df)

# Get the runs and order by date
runs = client.search_runs(experiment_id, order_by=["attributes.start_time desc"])
# Take the most recent run
print(runs[0].data.metrics)
print(runs[0].info.run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Registry
# MAGIC
# MAGIC The MLflow Model Registry component is a centralized model store, set of APIs, and UI, to collaboratively manage the full lifecycle of an MLflow Model. 
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/model-registry.png" style="height: 400px; margin: 20px"/></div>
# MAGIC
# MAGIC ### Programatically Register a Model
# MAGIC Create a unique model name so you don't clash with other workspace users. 
# MAGIC
# MAGIC Note that a registered model name must be a non-empty UTF-8 string and cannot contain forward slashes(/), periods(.), or colons(:).

# COMMAND ----------

# Set a name for the model (it does not matter if it already exists)
model_name = f"Taxi-Fare-Predictor"
print(f"Model Name: {model_name}")

# Get the run id and get the model path
run_id = run.info.run_id
model_uri = f"runs:/{run_id}/model"

# Register the model
model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Update Model
# MAGIC You can check the status of a model, update the description, and even transition it to production.

# COMMAND ----------

# Get the model detailos from the name
model_version_details = client.get_model_version(name=model_name, version=1)
model_version_details.status

# Update the model description
client.update_registered_model(
    name=model_details.name,
    description="This model forecasts wine quality based on various listing inputs."
)

# Update the description for a given version
client.update_model_version(
    name=model_details.name,
    version=model_details.version,
    description="This model version was built using OLS linear regression with sklearn."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploying a Model
# MAGIC
# MAGIC The MLflow Model Registry defines several model stages: **`None`**, **`Staging`**, **`Production`**, and **`Archived`**. Each stage has a unique meaning. For example, **`Staging`** is meant for model testing, while **`Production`** is for models that have completed the testing or review processes and have been deployed to applications. 
# MAGIC
# MAGIC Users with appropriate permissions can transition models between stages.

# COMMAND ----------

# Transition model to production
client.transition_model_version_stage(
    name=model_details.name,
    version=model_details.version,
    stage="Staging"
)

# Check the model version
model_version_details = client.get_model_version(
    name=model_details.name,
    version=model_details.version
)
print(f"The current model stage is: '{model_version_details.current_stage}'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load and Test Production Model
# MAGIC Fetch the latest model using a **`pyfunc`**.  Loading the model in this way allows us to use the model regardless of the package that was used to train it.

# COMMAND ----------

import mlflow.pyfunc

# Get the URI of the Staging model
model_version_uri = f"models:/{model_name}/Staging"

# Load the model
model_version_1 = mlflow.pyfunc.load_model(model_version_uri)

# Make predictions using the model
model_version_1.predict(X_test)

# COMMAND ----------


