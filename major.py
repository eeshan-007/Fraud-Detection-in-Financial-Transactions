import pandas as pd

train_trans = pd.read_csv("train_transaction.csv")
train_id = pd.read_csv("train_identity.csv")

print(train_trans.shape)
print(train_id.shape)
df = pd.merge(
    train_trans,
    train_id,
    on="TransactionID",
    how="left"
)
y = df["isFraud"]
print(df.shape)
X = df.select_dtypes(include=["number"])

X = X.drop(
    ["isFraud", "TransactionID"],
    axis=1,
    errors="ignore"
)

print("Features:", len(X.columns))

# train test split 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# missing values here
X_train = X_train.fillna(
    X_train.median()
)

X_test = X_test.fillna(
    X_train.median()
)

print(
    "Missing Train:",
    X_train.isnull().sum().sum()
)

print(
    "Missing Test:",
    X_test.isnull().sum().sum()
)

# imbalance 

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale = negative / positive

print("Scale Weight =", scale)

# Catboost model
from catboost import CatBoostClassifier

model = CatBoostClassifier(
    iterations=1000,
    depth=8,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="AUC",
    class_weights=[1, 10],
    random_seed=42,
    verbose=100
)
# train model 
model.fit(
    X_train,
    y_train
)
y_prob = model.predict_proba(X_test)[:, 1]

# t(threshold) testing 

from sklearn.metrics import classification_report

for t in [0.10, 0.15, 0.20, 0.25]:

    y_pred = (y_prob >= t).astype(int)

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    print(
        f"\nThreshold={t:.2f}"
        f" Precision={report['1']['precision']:.3f}"
        f" Recall={report['1']['recall']:.3f}"
        f" F1={report['1']['f1-score']:.3f}"
    )

# ROC AUC 

from sklearn.metrics import roc_auc_score

auc = roc_auc_score(
    y_test,
    y_prob
)

print("\nROC AUC =", auc)

#PR

from sklearn.metrics import average_precision_score

pr_auc = average_precision_score(
    y_test,
    y_prob
)

print("PR AUC =", pr_auc)

#feature importance 
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)
print("ROC AUC =", auc)
print("PR AUC =", pr_auc)
print("\nTop 20 Features")
print(importance.head(20))
import joblib

joblib.dump(model, "fraud_model.pkl")