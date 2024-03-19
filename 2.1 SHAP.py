# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: left; line-height: 0; padding-top: 9px; padding-left:150px">
# MAGIC   <img src="https://static1.squarespace.com/static/5bce4071ab1a620db382773e/t/5d266c78abb6d10001e4013e/1562799225083/appliedazuredatabricks3.png" alt="Databricks Learning" style="width: 600px; height: 163px">
# MAGIC </div>
# MAGIC
# MAGIC # *SHAP*. Presented by <a href="www.advancinganalytics.co.uk">Advancing Analytics</a>

# COMMAND ----------

# MAGIC %md 
# MAGIC 💡 In this lesson you will grasp the following concepts:
# MAGIC * Create Tree based SHAP explainer for PySpark and Scikit-learn models
# MAGIC * Create summary plots and force plots
# MAGIC * Parrallelise SHAP value calculations

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Libraries
# MAGIC Import all of the libraries used in the notebook

# COMMAND ----------

import shap
from typing import Iterator
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql.types import StructType, StructField, FloatType
from xgboost import XGBRegressor
import numpy as np

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

# COMMAND ----------

from pyspark.sql.types import IntegerType, FloatType
from pyspark.sql.functions import col, when, percentile_approx
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the Data
# MAGIC This notebook will use the wine quality dataset to look at what affects the quality of wine!

# COMMAND ----------

df = (
    spark.read.format("csv")
    .option("inferSchema", True)
    .option("header", True)
    .option("sep", ";")
    .load("dbfs:/databricks-datasets/wine-quality/winequality-red.csv")
)
display(df)

label = "quality"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Split the dataframe into a test and train dataset
# MAGIC Split the data 70/30 into train/test

# COMMAND ----------

seed = 42
trainDF, testDF = df.randomSplit([0.7, 0.3], seed=42)

print("We have %d training examples and %d test examples." % (trainDF.count(), testDF.count()))

# COMMAND ----------

num_feats = df.drop(label).columns
vectorAssembler = VectorAssembler(
    inputCols=num_feats, outputCol="rawFeatures", handleInvalid="skip"
)
model = DecisionTreeRegressor(featuresCol="rawFeatures", labelCol=label)  # maxIter=100)

# COMMAND ----------

pipeline = Pipeline().setStages([vectorAssembler, model])
trans = Pipeline().setStages([vectorAssembler])

# COMMAND ----------

input_df = trans.fit(trainDF).transform(testDF)
pmodel = pipeline.fit(trainDF)
predictionDF = pmodel.transform(testDF)

# COMMAND ----------

evaluatorrmse = (
    RegressionEvaluator()
    .setMetricName("rmse")
    .setPredictionCol("prediction")
    .setLabelCol(label)
)
evaluatorr2 = (
    RegressionEvaluator()
    .setMetricName("r2")
    .setPredictionCol("prediction")
    .setLabelCol(label)
)

rmse = evaluatorrmse.evaluate(predictionDF)
r2 = evaluatorr2.evaluate(predictionDF)
print("Test RMSE = %f" % rmse)
print("Test R2 = %f" % r2)

# COMMAND ----------

display(predictionDF)

# COMMAND ----------

rfc = model.fit(input_df)

# COMMAND ----------

#######
# Create pandas df from a dictionary of distributed rows.
# Current arrow conversion doesn't support exploding and converting ml vectors.

rows_list = []
for row in input_df.collect():
    dict1 = {}
    dict1.update({k: v for k, v in zip(input_df.columns, row.rawFeatures)})
    rows_list.append(dict1)

pandas_df = pd.DataFrame(rows_list)

# COMMAND ----------

#######
# Use shap to explain the fitted model.

explainer = shap.TreeExplainer(rfc)
shap_values = explainer.shap_values(pandas_df, check_additivity=False)

# Take shap_values[0] here to type-hint the singular class classification (binary probabilities are equivalent to a single class)
# The pandas dataframe is used here to allow shap calculations across OHE vectors (ordering is retained)
p = shap.summary_plot(shap_values, pandas_df, show=False)
display(p)

# COMMAND ----------

#######
# shap_values[0] here is used to identify a singular row from the shap values for the **first** class.
shap_display = shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    feature_names=pandas_df.columns,
    matplotlib=True,
)
display(shap_display)

# COMMAND ----------

model = XGBRegressor()
model.fit(pandas_df, input_df.select(label).toPandas().values.ravel())

# COMMAND ----------

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(pandas_df, check_additivity=False)
p = shap.summary_plot(shap_values, pandas_df, show=False)
display(p)

# COMMAND ----------

explainer = shap.TreeExplainer(model)
columns_for_shap_calculation = pandas_df.columns


def calculate_shap(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    for X in iterator:
        yield pd.DataFrame(
            explainer.shap_values(np.array(X), check_additivity=False),
            columns=columns_for_shap_calculation,
        )


return_schema = StructType()
for feature in columns_for_shap_calculation:
    return_schema = return_schema.add(StructField(feature, FloatType()))

shap_values = (
    spark.createDataFrame(pandas_df)
    .mapInPandas(calculate_shap, schema=return_schema)
    .toPandas()
    .values
)

# COMMAND ----------

p = shap.summary_plot(shap_values, pandas_df, show=False)
display(p)

# COMMAND ----------


