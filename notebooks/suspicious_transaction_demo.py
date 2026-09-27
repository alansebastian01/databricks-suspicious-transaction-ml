# Databricks notebook source
# MAGIC %md
# MAGIC # Suspicious Transaction Detection — End-to-End ML Demo
# MAGIC Synthetic-data demo: Delta tables → feature engineering → Random Forest → predictions → MLflow → Unity Catalog.
# MAGIC
# MAGIC **Demo only.** The suspicious patterns are intentionally strong and model metrics are not representative of production fraud detection.

# COMMAND ----------

CATALOG = "main"
SCHEMA = "demo"
EXPERIMENT_PATH = "/Shared/suspicious_transaction_demo"
REGISTERED_MODEL_NAME = f"{CATALOG}.{SCHEMA}.suspicious_transaction_model"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.transaction (
    txn_id STRING,
    user_id STRING,
    merchant_id STRING,
    merchant_name STRING,
    txn_timestamp TIMESTAMP,
    txn_amount DOUBLE,
    txn_currency STRING,
    txn_status STRING,
    txn_type STRING,
    from_country STRING,
    to_country STRING,
    is_cross_border INT,
    payment_method STRING,
    device_type STRING
)
USING DELTA
""")

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DoubleType, IntegerType
from datetime import datetime, timedelta
import random

schema = StructType([
    StructField("txn_id", StringType(), False), StructField("user_id", StringType(), False),
    StructField("merchant_id", StringType(), False), StructField("merchant_name", StringType(), False),
    StructField("txn_timestamp", TimestampType(), False), StructField("txn_amount", DoubleType(), False),
    StructField("txn_currency", StringType(), False), StructField("txn_status", StringType(), False),
    StructField("txn_type", StringType(), False), StructField("from_country", StringType(), False),
    StructField("to_country", StringType(), False), StructField("is_cross_border", IntegerType(), False),
    StructField("payment_method", StringType(), False), StructField("device_type", StringType(), False),
])

merchants = [("M001","Starbucks"),("M002","Apple Store"),("M003","Amazon"),("M004","Tokyo Camera"),
             ("M005","Best Electronics"),("M006","Nike"),("M007","Target"),("M008","Walmart"),
             ("M009","Seoul Fashion"),("M010","Taipei Electronics")]
countries = ["US", "JP", "TW", "KR", "CN"]
payment_methods = ["CARD", "QR", "WALLET"]
device_types = ["MOBILE", "WEB", "POS"]

random.seed(42)
start_time = datetime(2026, 9, 1)
rows = []
for i in range(10_000):
    user_id = f"U{random.randint(1, 1000):04d}"
    merchant_id, merchant_name = random.choice(merchants)
    txn_timestamp = start_time + timedelta(minutes=random.randint(0, 30 * 24 * 60))
    txn_amount = round(random.uniform(5, 500), 2)
    from_country = random.choice(countries)
    to_country = from_country if random.random() < 0.85 else random.choice(countries)
    rows.append((f"TXN{i+1:07d}", user_id, merchant_id, merchant_name, txn_timestamp, txn_amount,
                 "USD", "COMPLETED", "PAYMENT", from_country, to_country,
                 int(from_country != to_country), random.choice(payment_methods), random.choice(device_types)))

transaction_df = spark.createDataFrame(rows, schema=schema)
spark.sql(f"TRUNCATE TABLE {CATALOG}.{SCHEMA}.transaction")
transaction_df.write.mode("append").insertInto(f"{CATALOG}.{SCHEMA}.transaction")
print("Loaded 10,000 normal transactions")

# COMMAND ----------

# Add 25 suspicious users x 4 rapid, high-value, cross-border transactions.
random.seed(100)
suspicious_rows = []
suspicious_merchants = [("M004","Tokyo Camera"),("M005","Best Electronics"),("M010","Taipei Electronics")]
base_time = datetime(2026, 9, 20, 1, 0, 0)
txn_counter = 10_001

for user_num in range(1, 26):
    user_id = f"SUSP_USER_{user_num:03d}"
    merchant_id, merchant_name = random.choice(suspicious_merchants)
    sequence_start = base_time + timedelta(hours=user_num)
    for sequence_no in range(4):
        suspicious_rows.append((
            f"TXN{txn_counter:07d}", user_id, merchant_id, merchant_name,
            sequence_start + timedelta(minutes=sequence_no), round(random.uniform(1500, 5000), 2),
            "USD", "COMPLETED", "PAYMENT", "US", random.choice(["JP","TW","KR"]), 1,
            random.choice(["QR","WALLET"]), "MOBILE"
        ))
        txn_counter += 1

spark.createDataFrame(suspicious_rows, schema=schema).write.mode("append").insertInto(f"{CATALOG}.{SCHEMA}.transaction")
print("Added 100 suspicious transactions")

# COMMAND ----------

# Validation
spark.sql(f"""
SELECT COUNT(*) total_transactions,
       COUNT(DISTINCT user_id) unique_users,
       COUNT(DISTINCT merchant_id) unique_merchants,
       ROUND(AVG(txn_amount), 2) avg_transaction_amount
FROM {CATALOG}.{SCHEMA}.transaction
""").show()

# COMMAND ----------

# Labels are synthetic and deliberately tied to the generated suspicious-user convention.
spark.sql(f"""
CREATE OR REPLACE TABLE {CATALOG}.{SCHEMA}.transaction_labels AS
SELECT txn_id,
       CASE WHEN user_id LIKE 'SUSP_USER_%' THEN 1 ELSE 0 END AS suspicious_label
FROM {CATALOG}.{SCHEMA}.transaction
""")

spark.sql(f"SELECT suspicious_label, COUNT(*) txn_count FROM {CATALOG}.{SCHEMA}.transaction_labels GROUP BY suspicious_label ORDER BY suspicious_label").show()

# COMMAND ----------

# Feature engineering.
# NOTE: user/merchant aggregate features below use the complete demo dataset. This is acceptable for illustrating
# the workflow, but production features must be point-in-time correct to avoid future-information leakage.
spark.sql(f"""
CREATE OR REPLACE TABLE {CATALOG}.{SCHEMA}.transaction_ml_features AS
WITH base AS (
    SELECT t.txn_id, t.user_id, t.merchant_id, t.merchant_name, t.txn_timestamp,
           t.txn_amount, t.is_cross_border, t.payment_method, t.device_type,
           l.suspicious_label,
           HOUR(t.txn_timestamp) AS txn_hour,
           COUNT(*) OVER (PARTITION BY t.user_id) AS user_txn_count,
           AVG(t.txn_amount) OVER (PARTITION BY t.user_id) AS user_avg_amount,
           MAX(t.txn_amount) OVER (PARTITION BY t.user_id) AS user_max_amount,
           COUNT(*) OVER (PARTITION BY t.merchant_id) AS merchant_txn_count,
           AVG(t.txn_amount) OVER (PARTITION BY t.merchant_id) AS merchant_avg_amount,
           COUNT(*) OVER (
               PARTITION BY t.user_id
               ORDER BY CAST(t.txn_timestamp AS LONG)
               RANGE BETWEEN 600 PRECEDING AND CURRENT ROW
           ) AS transactions_last_10_min
    FROM {CATALOG}.{SCHEMA}.transaction t
    INNER JOIN {CATALOG}.{SCHEMA}.transaction_labels l ON t.txn_id = l.txn_id
)
SELECT *, ROUND(txn_amount / NULLIF(user_avg_amount, 0), 2) AS amount_vs_user_avg
FROM base
""")

# COMMAND ----------

feature_columns = [
    "txn_amount", "is_cross_border", "txn_hour", "user_txn_count",
    "merchant_txn_count", "merchant_avg_amount", "transactions_last_10_min"
]

ml_df = spark.table(f"{CATALOG}.{SCHEMA}.transaction_ml_features")
selected_df = ml_df.select(*feature_columns, "suspicious_label").dropna()
selected_df.groupBy("suspicious_label").count().show()

pdf = selected_df.toPandas()
X = pdf[feature_columns]
y = pdf["suspicious_label"]

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100, max_depth=6, random_state=42, class_weight="balanced"
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, predictions))

# COMMAND ----------

import pandas as pd
feature_importance_df = pd.DataFrame({
    "feature": feature_columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)
display(feature_importance_df)

# COMMAND ----------

# Batch-score all demo transactions.
score_base = (
    spark.table(f"{CATALOG}.{SCHEMA}.transaction_ml_features")
    .select("txn_id", "user_id", "merchant_name", "txn_timestamp", "txn_amount",
            "is_cross_border", "txn_hour", "user_txn_count", "merchant_txn_count",
            "merchant_avg_amount", "transactions_last_10_min", "suspicious_label")
    .dropna()
)
score_pdf = score_base.toPandas()
score_pdf["predicted_label"] = model.predict(score_pdf[feature_columns])
score_pdf["suspicious_probability"] = model.predict_proba(score_pdf[feature_columns])[:, 1]

suspicious_output = score_pdf[score_pdf["predicted_label"] == 1].sort_values(
    "suspicious_probability", ascending=False
)
display(suspicious_output.head(50))

spark.createDataFrame(suspicious_output).write.format("delta").mode("overwrite").saveAsTable(
    f"{CATALOG}.{SCHEMA}.transaction_predictions"
)

# COMMAND ----------

# MLflow tracking with signature/input example for Unity Catalog compatibility.
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

mlflow.set_experiment(EXPERIMENT_PATH)
metrics = {
    "accuracy": accuracy_score(y_test, predictions),
    "precision": precision_score(y_test, predictions, zero_division=0),
    "recall": recall_score(y_test, predictions, zero_division=0),
    "f1_score": f1_score(y_test, predictions, zero_division=0),
}
input_example = X_train.iloc[[0]]
signature = infer_signature(X_train, model.predict(X_train))

with mlflow.start_run(run_name="random_forest_with_signature") as run:
    mlflow.log_params({"model_type": "RandomForest", "n_estimators": 100, "max_depth": 6,
                       "class_weight": "balanced", "random_state": 42})
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=input_example,
    )
    run_id = run.info.run_id

print("MLflow run ID:", run_id)
print("Metrics:", metrics)

# COMMAND ----------

# Register the logged model in Unity Catalog.
# If your workspace uses a manual governance/approval flow, skip this cell and register from the MLflow UI.
mlflow.set_registry_uri("databricks-uc")
model_uri = f"runs:/{run_id}/model"
registered = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
print(f"Registered {REGISTERED_MODEL_NAME}, version {registered.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Production notes
# MAGIC - Replace synthetic labels with governed outcomes such as confirmed fraud/investigation results.
# MAGIC - Build point-in-time correct features; do not use future transactions in historical features.
# MAGIC - Prefer temporal validation for transaction streams.
# MAGIC - Tune the decision threshold for the business cost of false positives vs. false negatives.
# MAGIC - Add PR-AUC/ROC-AUC, calibration, drift/model monitoring, explainability, and retraining controls.
