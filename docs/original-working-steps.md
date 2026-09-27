
# Databricks ML Demo - Suspicious Transaction Detection

## Goal

Build a small end-to-end ML demo in Databricks using synthetic transaction data.

Flow:

```text
main.demo.transaction
        ↓
main.demo.transaction_labels
        ↓
main.demo.transaction_ml_features
        ↓
Random Forest Model
        ↓
Predictions
        ↓
MLflow
        ↓
Unity Catalog Model Registry
```

---

# 1. Create Demo Schema

Run in a Databricks SQL cell:

```sql
%sql

CREATE SCHEMA IF NOT EXISTS main.demo;
```

---

# 2. Create Base Transaction Table

```sql
%sql

CREATE TABLE IF NOT EXISTS main.demo.transaction (
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
USING DELTA;
```

---

# 3. Generate Normal Synthetic Transactions

Python cell:

```python
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    TimestampType,
    DoubleType,
    IntegerType
)

from datetime import datetime, timedelta
import random

random.seed(42)

schema = StructType([
    StructField("txn_id", StringType(), False),
    StructField("user_id", StringType(), False),
    StructField("merchant_id", StringType(), False),
    StructField("merchant_name", StringType(), False),
    StructField("txn_timestamp", TimestampType(), False),
    StructField("txn_amount", DoubleType(), False),
    StructField("txn_currency", StringType(), False),
    StructField("txn_status", StringType(), False),
    StructField("txn_type", StringType(), False),
    StructField("from_country", StringType(), False),
    StructField("to_country", StringType(), False),
    StructField("is_cross_border", IntegerType(), False),
    StructField("payment_method", StringType(), False),
    StructField("device_type", StringType(), False)
])

merchants = [
    ("M001", "Starbucks"),
    ("M002", "Apple Store"),
    ("M003", "Amazon"),
    ("M004", "Tokyo Camera"),
    ("M005", "Best Electronics"),
    ("M006", "Nike"),
    ("M007", "Target"),
    ("M008", "Walmart"),
    ("M009", "Seoul Fashion"),
    ("M010", "Taipei Electronics")
]

countries = ["US", "JP", "TW", "KR", "CN"]

payment_methods = [
    "CARD",
    "QR",
    "WALLET"
]

device_types = [
    "MOBILE",
    "WEB",
    "POS"
]

start_time = datetime(2026, 9, 1)

rows = []

for i in range(10000):

    user_id = f"U{random.randint(1, 1000):04d}"

    merchant_id, merchant_name = random.choice(merchants)

    txn_timestamp = start_time + timedelta(
        minutes=random.randint(
            0,
            30 * 24 * 60
        )
    )

    txn_amount = round(
        random.uniform(5, 500),
        2
    )

    from_country = random.choice(countries)

    if random.random() < 0.85:
        to_country = from_country
    else:
        to_country = random.choice(countries)

    is_cross_border = int(
        from_country != to_country
    )

    rows.append((
        f"TXN{i+1:07d}",
        user_id,
        merchant_id,
        merchant_name,
        txn_timestamp,
        txn_amount,
        "USD",
        "COMPLETED",
        "PAYMENT",
        from_country,
        to_country,
        is_cross_border,
        random.choice(payment_methods),
        random.choice(device_types)
    ))

transaction_df = spark.createDataFrame(
    rows,
    schema=schema
)

transaction_df.printSchema()

display(transaction_df)
```

---

# 4. Load Base Data

```python
spark.sql("""
TRUNCATE TABLE main.demo.transaction
""")

transaction_df.write \
    .mode("append") \
    .insertInto("main.demo.transaction")

print("Base transaction data loaded.")
```

---

# 5. Validate Base Data

```sql
%sql

SELECT
    COUNT(*) AS total_transactions,
    COUNT(DISTINCT user_id) AS unique_users,
    COUNT(DISTINCT merchant_id) AS unique_merchants,
    ROUND(AVG(txn_amount), 2) AS avg_transaction_amount
FROM main.demo.transaction;
```

Expected:

```text
total_transactions       10000
unique_users             1000
unique_merchants         10
avg_transaction_amount   ~250.66
```

---

# 6. Generate Suspicious Transactions

Create 25 suspicious users with 4 rapid high-value transactions each.

```python
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    TimestampType,
    DoubleType,
    IntegerType
)

from datetime import datetime, timedelta
import random

random.seed(100)

suspicious_rows = []

suspicious_schema = StructType([
    StructField("txn_id", StringType(), False),
    StructField("user_id", StringType(), False),
    StructField("merchant_id", StringType(), False),
    StructField("merchant_name", StringType(), False),
    StructField("txn_timestamp", TimestampType(), False),
    StructField("txn_amount", DoubleType(), False),
    StructField("txn_currency", StringType(), False),
    StructField("txn_status", StringType(), False),
    StructField("txn_type", StringType(), False),
    StructField("from_country", StringType(), False),
    StructField("to_country", StringType(), False),
    StructField("is_cross_border", IntegerType(), False),
    StructField("payment_method", StringType(), False),
    StructField("device_type", StringType(), False)
])

suspicious_merchants = [
    ("M004", "Tokyo Camera"),
    ("M005", "Best Electronics"),
    ("M010", "Taipei Electronics")
]

base_time = datetime(2026, 9, 20, 1, 0, 0)

txn_counter = 10001

for user_num in range(1, 26):

    user_id = f"SUSP_USER_{user_num:03d}"

    merchant_id, merchant_name = random.choice(
        suspicious_merchants
    )

    sequence_start = base_time + timedelta(
        hours=user_num
    )

    for sequence_no in range(4):

        txn_timestamp = sequence_start + timedelta(
            minutes=sequence_no
        )

        txn_amount = round(
            random.uniform(1500, 5000),
            2
        )

        suspicious_rows.append((
            f"TXN{txn_counter:07d}",
            user_id,
            merchant_id,
            merchant_name,
            txn_timestamp,
            txn_amount,
            "USD",
            "COMPLETED",
            "PAYMENT",
            "US",
            random.choice(["JP", "TW", "KR"]),
            1,
            random.choice(["QR", "WALLET"]),
            "MOBILE"
        ))

        txn_counter += 1

suspicious_df = spark.createDataFrame(
    suspicious_rows,
    schema=suspicious_schema
)

display(suspicious_df)
```

---

# 7. Append Suspicious Transactions

```python
suspicious_df.write \
    .mode("append") \
    .insertInto("main.demo.transaction")

print("Suspicious transactions added.")
```

---

# 8. Validate Suspicious Data

```sql
%sql

SELECT
    user_id,
    merchant_name,
    txn_timestamp,
    txn_amount,
    from_country,
    to_country,
    is_cross_border,
    payment_method
FROM main.demo.transaction
WHERE user_id LIKE 'SUSP_USER_%'
ORDER BY user_id, txn_timestamp;
```

Example pattern:

```text
SUSP_USER_001
02:00  $3108
02:01  $4061
02:02  $3365
02:03  $2421
```

Characteristics:

```text
Same user
Same merchant
Rapid transactions
High transaction amount
Cross-border
QR / Wallet
```

---

# 9. Create Transaction Labels

```sql
%sql

CREATE OR REPLACE TABLE main.demo.transaction_labels AS

SELECT
    txn_id,

    CASE
        WHEN user_id LIKE 'SUSP_USER_%'
        THEN 1
        ELSE 0
    END AS suspicious_label

FROM main.demo.transaction;
```

---

# 10. Validate Labels

```sql
%sql

SELECT
    suspicious_label,
    COUNT(*) AS txn_count
FROM main.demo.transaction_labels
GROUP BY suspicious_label
ORDER BY suspicious_label;
```

Expected:

```text
0   10000
1     100
```

---

# 11. Create ML Feature Table

```sql
%sql

CREATE OR REPLACE TABLE main.demo.transaction_ml_features AS

WITH base AS (

    SELECT
        t.txn_id,
        t.user_id,
        t.merchant_id,
        t.merchant_name,
        t.txn_timestamp,
        t.txn_amount,
        t.is_cross_border,
        t.payment_method,
        t.device_type,
        l.suspicious_label,

        HOUR(t.txn_timestamp) AS txn_hour,

        COUNT(*) OVER (
            PARTITION BY t.user_id
        ) AS user_txn_count,

        AVG(t.txn_amount) OVER (
            PARTITION BY t.user_id
        ) AS user_avg_amount,

        MAX(t.txn_amount) OVER (
            PARTITION BY t.user_id
        ) AS user_max_amount,

        COUNT(*) OVER (
            PARTITION BY t.merchant_id
        ) AS merchant_txn_count,

        AVG(t.txn_amount) OVER (
            PARTITION BY t.merchant_id
        ) AS merchant_avg_amount,

        COUNT(*) OVER (
            PARTITION BY t.user_id
            ORDER BY CAST(t.txn_timestamp AS LONG)
            RANGE BETWEEN 600 PRECEDING AND CURRENT ROW
        ) AS transactions_last_10_min

    FROM main.demo.transaction t

    INNER JOIN main.demo.transaction_labels l
        ON t.txn_id = l.txn_id
)

SELECT
    *,

    ROUND(
        txn_amount /
        NULLIF(user_avg_amount, 0),
        2
    ) AS amount_vs_user_avg

FROM base;
```

---

# 12. Compare Normal vs Suspicious Features

```sql
%sql

SELECT
    suspicious_label,

    COUNT(*) AS txn_count,

    ROUND(
        AVG(txn_amount),
        2
    ) AS avg_txn_amount,

    ROUND(
        AVG(user_txn_count),
        2
    ) AS avg_user_txn_count,

    ROUND(
        AVG(transactions_last_10_min),
        2
    ) AS avg_velocity_10min,

    ROUND(
        AVG(is_cross_border),
        2
    ) AS cross_border_ratio,

    ROUND(
        AVG(amount_vs_user_avg),
        2
    ) AS avg_amount_vs_user_avg

FROM main.demo.transaction_ml_features

GROUP BY suspicious_label

ORDER BY suspicious_label;
```

Observed example:

```text
label 0
avg amount        ~250
velocity          ~1
cross border      ~0.12

label 1
avg amount        ~3300
velocity          ~2.5
cross border      1.00
```

---

# 13. Select ML Features

Python cell:

```python
feature_columns = [
    "txn_amount",
    "is_cross_border",
    "txn_hour",
    "user_txn_count",
    "merchant_txn_count",
    "merchant_avg_amount",
    "transactions_last_10_min"
]
```

---

# 14. Load Feature Dataset

```python
ml_df = spark.table(
    "main.demo.transaction_ml_features"
)

selected_df = ml_df.select(
    *feature_columns,
    "suspicious_label"
).dropna()

display(selected_df)
```

---

# 15. Validate Training Labels

```python
selected_df.groupBy(
    "suspicious_label"
).count().show()
```

Expected:

```text
0   10000
1     100
```

---

# 16. Convert to Pandas

Because this demo dataset is small:

```python
pdf = selected_df.toPandas()

X = pdf[feature_columns]

y = pdf["suspicious_label"]
```

---

# 17. Train / Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training rows:", len(X_train))
print("Testing rows :", len(X_test))
```

Expected approximately:

```text
Training rows: 8080
Testing rows : 2020
```

---

# 18. Train Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    random_state=42,
    class_weight="balanced"
)

model.fit(
    X_train,
    y_train
)

print("Model training completed.")
```

---

# 19. Generate Predictions

```python
predictions = model.predict(
    X_test
)

probabilities = model.predict_proba(
    X_test
)[:, 1]
```

---

# 20. Evaluate Model

```python
from sklearn.metrics import classification_report

print(
    classification_report(
        y_test,
        predictions
    )
)
```

Synthetic data may return:

```text
precision    1.00
recall       1.00
f1-score     1.00
accuracy     1.00
```

Important:

This does NOT represent realistic production accuracy.

The synthetic suspicious records were intentionally created with strong patterns.

Real production data will have overlap between normal and suspicious behavior.

---

# 21. Feature Importance

```python
import pandas as pd

feature_importance_df = pd.DataFrame({
    "feature": feature_columns,
    "importance": model.feature_importances_
}).sort_values(
    "importance",
    ascending=False
)

display(feature_importance_df)
```

This helps explain which features the model relied on most.

Examples:

```text
txn_amount
is_cross_border
transactions_last_10_min
user_txn_count
merchant_txn_count
```

---

# 22. Score All Transactions

```python
score_base = (
    spark.table(
        "main.demo.transaction_ml_features"
    )
    .select(
        "txn_id",
        "user_id",
        "merchant_name",
        "txn_timestamp",
        "txn_amount",
        "is_cross_border",
        "txn_hour",
        "user_txn_count",
        "merchant_txn_count",
        "merchant_avg_amount",
        "transactions_last_10_min",
        "suspicious_label"
    )
    .dropna()
)

score_pdf = score_base.toPandas()
```

---

# 23. Generate Risk Probability

```python
score_pdf["predicted_label"] = model.predict(
    score_pdf[feature_columns]
)

score_pdf["suspicious_probability"] = model.predict_proba(
    score_pdf[feature_columns]
)[:, 1]
```

---

# 24. Show Predicted Suspicious Transactions

```python
suspicious_output = score_pdf[
    score_pdf["predicted_label"] == 1
].sort_values(
    "suspicious_probability",
    ascending=False
)

display(
    suspicious_output[
        [
            "txn_id",
            "user_id",
            "merchant_name",
            "txn_timestamp",
            "txn_amount",
            "is_cross_border",
            "transactions_last_10_min",
            "suspicious_label",
            "predicted_label",
            "suspicious_probability"
        ]
    ].head(50)
)
```

This is a good demo screen.

Example:

```text
txn_id
user_id
merchant
amount
velocity
actual label
predicted label
risk probability
```

---

# 25. Save Predictions

```python
prediction_spark_df = spark.createDataFrame(
    suspicious_output
)

prediction_spark_df.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(
        "main.demo.transaction_predictions"
    )
```

---

# 26. Validate Predictions

```sql
%sql

SELECT
    txn_id,
    user_id,
    merchant_name,
    txn_timestamp,
    txn_amount,
    suspicious_probability

FROM main.demo.transaction_predictions

ORDER BY suspicious_probability DESC

LIMIT 50;
```

---

# 27. Log Experiment to MLflow

```python
import mlflow
import mlflow.sklearn

from mlflow.models import infer_signature

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

mlflow.set_experiment(
    "/Shared/suspicious_transaction_demo"
)

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions
)

recall = recall_score(
    y_test,
    predictions
)

f1 = f1_score(
    y_test,
    predictions
)

input_example = X_train.iloc[[0]]

signature = infer_signature(
    X_train,
    model.predict(X_train)
)

with mlflow.start_run(
    run_name="random_forest_with_signature"
) as run:

    mlflow.log_param(
        "model_type",
        "RandomForest"
    )

    mlflow.log_param(
        "n_estimators",
        100
    )

    mlflow.log_param(
        "max_depth",
        6
    )

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=input_example
    )

    print(
        "Run ID:",
        run.info.run_id
    )

print(
    "Model logged with signature successfully."
)
```

---

# 28. View MLflow Experiment

In Databricks:

```text
Experiments
    ↓
suspicious_transaction_demo
    ↓
random_forest_with_signature
```

Review:

```text
Parameters
---------
model_type
n_estimators
max_depth

Metrics
-------
accuracy
precision
recall
f1_score

Artifacts
---------
model
```

---

# 29. Register Model

Open the successful MLflow run.

Click:

```text
Register model
```

Use:

```text
main.demo.suspicious_transaction_model
```

Important:

Unity Catalog requires a model signature.

That is why the model was logged using:

```python
signature = infer_signature(
    X_train,
    model.predict(X_train)
)
```

---

# Final Demo Architecture

```text
                 DATA ENGINEERING

              main.demo.transaction
                       │
                       ▼
             transaction_labels
                       │
                       ▼
          transaction_ml_features
                       │
                       │
                       ▼
                MACHINE LEARNING

                Random Forest
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
   prediction                 probability
                                  │
                                  ▼
                    transaction_predictions
                                  │
                                  ▼
                              MLflow
                                  │
                                  ▼
                         Model Registry
```

---

# Suggested Demo Explanation

## 1. Raw Data

> We start with a normal transaction table containing user, merchant, amount, timestamp, geography, payment method, and device information.

## 2. Synthetic Suspicious Behavior

> We intentionally create a small number of rapid, high-value, cross-border transactions to simulate suspicious activity.

## 3. Feature Engineering

> Raw transactions alone are not enough for ML. We calculate behavioral features such as transaction velocity, user activity, merchant activity, and cross-border behavior.

## 4. Train Model

> We use a Random Forest classifier to learn the relationship between these behavioral features and the suspicious transaction label.

## 5. Prediction

> The model produces both a classification and a suspicious probability.

## 6. MLflow

> MLflow tracks model parameters, metrics, artifacts, and versions so the experiment is reproducible.

## 7. Unity Catalog

> Once validated, the model can be registered in Unity Catalog and governed like other enterprise data assets.

---

# Important Note

This is a demonstration dataset.

The suspicious patterns were intentionally made easy to distinguish.

A production suspicious transaction model should use:

* confirmed fraud / investigation labels
* richer user history
* merchant behavior
* device behavior
* geography
* transaction velocity
* amount deviation
* historical patterns
* model monitoring
* retraining
* threshold tuning
* precision / recall tradeoff


