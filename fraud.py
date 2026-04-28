import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib

# ML Imports
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Setup
warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# --- STEP 1: LOAD DATA ---
# Using the full dataset (not filtered) to get the best recall
df = pd.read_csv("Project_1\\AIML Dataset.csv")

# --- STEP 2: PRELIMINARY DATA INSPECTION ---
print("Data Head:")
print(df.head())
print("\nFraud Counts:")
print(df["isFraud"].value_counts())

# --- STEP 3: FEATURE ENGINEERING ---
# Adding the balance difference columns
df['diff_Org'] = df['oldbalanceOrg'] - df['newbalanceOrig']
df['diff_Dest'] = df['oldbalanceDest'] - df['newbalanceDest']

# --- STEP 4: FEATURE SELECTION ---
# X = Everything except the targets and names
# y = IsFraud
X = df.drop(['isFraud', 'nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
y = df['isFraud']

# --- STEP 5: DATA SPLIT ---
# 70/30 split with stratify to keep fraud ratios consistent
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --- STEP 6: PREPROCESSOR ---
numeric_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'diff_Org', 'diff_Dest']
categorical_features = ['type']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# --- STEP 7: TRAIN THE MODEL ---
# Using max_iter=1000 and class_weight='balanced' to hit that >90% recall
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])

print("\nTraining Logistic Regression model... (This might take a minute with 6M rows)")
model_pipeline.fit(X_train, y_train)

# --- STEP 8: EVALUATION ---
y_pred = model_pipeline.predict(X_test)

print("\n--- FINAL CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred))

print("--- CONFUSION MATRIX ---")
print(confusion_matrix(y_test, y_pred))

# --- STEP 9: EXPORT ---
joblib.dump(model_pipeline, "fraud_model_logistic.pkl")
print("\nSuccess! Full pipeline saved to 'fraud_model_logistic.pkl'")