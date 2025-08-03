"""
Basic income prediction using RandomForest
No explainability or SHAP used
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
import os
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------
# 1. Check for the data file
# -----------------------------------
print("Current working directory:", os.getcwd())
if not os.path.exists("adult.data.txt"):
    print("❌ File 'adult.data.txt' not found in this folder.")
    exit()
else:
    print("✅ Found 'adult.data.txt'")

# -----------------------------------
# 2. Define column names and load data
# -----------------------------------
COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'income'
]

print("📥 Loading and preparing dataset...")
data = pd.read_csv("adult.data.txt", names=COLUMNS, na_values=' ?', skipinitialspace=True)
data.dropna(inplace=True)

# Encode categorical variables
print("🔠 Encoding categorical features...")
for col in data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# -----------------------------------
# 3. Prepare features and labels
# -----------------------------------
X = data.drop('income', axis=1)
y = data['income']  # 0 = <=50K, 1 = >50K

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------------
# 4. Train the RandomForest model
# -----------------------------------
print("🚀 Training RandomForestClassifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# -----------------------------------
# 5. Evaluate the model
# -----------------------------------
print("📊 Evaluating the model...")
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("\n=== ✅ Model Performance (No XAI) ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")
print("\nClassification report:\n", classification_report(y_test, y_pred))

# -----------------------------------
# 6. Predict a new sample (optional)
# -----------------------------------
print("🤖 Predicting a sample (black-box)...")
sample = X_test.iloc[[0]]
prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0][1]
print(f"Predicted label: {prediction}  |  Probability >50K: {probability:.3f}")

print("\n✅ Script finished successfully!")
import joblib
joblib.dump(model, "income_model.pkl")
