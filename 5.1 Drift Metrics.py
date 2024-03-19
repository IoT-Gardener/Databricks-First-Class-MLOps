# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: left; line-height: 0; padding-top: 9px; padding-left:150px">
# MAGIC   <img src="https://static1.squarespace.com/static/5bce4071ab1a620db382773e/t/5d266c78abb6d10001e4013e/1562799225083/appliedazuredatabricks3.png" alt="Databricks Learning" style="width: 600px; height: 163px">
# MAGIC </div>
# MAGIC
# MAGIC # *Drift Metrics*. Presented by <a href="www.advancinganalytics.co.uk">Advancing Analytics</a>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Import Libraries

# COMMAND ----------

from pyspark.sql.functions import to_date, year, month, col
import numpy as np
import pandas as pd

from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Functions
# MAGIC The code below implements a simple programatic implementation of PSI and KL Divergence

# COMMAND ----------

def psi(x, y):
    x = np.clip(x, a_min=0.0001, a_max=None) / sum(x)
    y = np.clip(y, a_min=0.0001, a_max=None) / sum(y)
    return sum((x-y)*np.log(x/y))

def kl_div(x, y):
    x = np.clip(x, a_min=0.0001, a_max=None) / sum(x)
    y = np.clip(y, a_min=0.0001, a_max=None) / sum(y)
    return sum(rel_entr(x, y))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Data

# COMMAND ----------

data = (
    spark.read.format("csv")
    .option("inferSchema", True)
    .option("header", True)
    .load("dbfs:/databricks-datasets/weather/high_temps")
).withColumn("Date", to_date("Date", format="d/M/yyyy")).withColumn("year", year("Date")).withColumn("month", month("Date"))
label = "temp"
df = data.select("Date", "year", label).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Format Data

# COMMAND ----------

# Create bins for categories
bins = [33.0, 40.0, 46.0, 52.0, 58.0, 64.0, 70.0, 76.0, 82.0, 88.0, 96.0]

# Split data into groups by year
years = (
    df.groupby(["year"])
    .apply(
        lambda x: pd.Series(np.histogram(x[[label]], bins=bins), index=["dist", "bins"])
    )
    .reset_index()
)

# COMMAND ----------

baseline = years.iloc[0].dist

# COMMAND ----------

years['wasserstein_distance'] = years.apply(lambda x: wasserstein_distance(baseline/sum(baseline), x['dist']/sum(x['dist'])), axis=1)
years['js'] = years.apply(lambda x: jensenshannon(baseline, x['dist']), axis=1)
years['kl_div'] = years.apply(lambda x: kl_div(baseline, x['dist']), axis=1)
years['psi'] = years.apply(lambda x: psi(baseline, x['dist']), axis=1)

# COMMAND ----------

years

# COMMAND ----------

for cnt, y in years.iterrows():
    plt.plot(y.bins[:-1], y.dist/y.dist.sum(), label=y.year)
    plt.legend()
plt.show()

# COMMAND ----------


