import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import os

os.makedirs('models', exist_ok=True)

print("="*50)
print("ALGORITMA 1: LOGISTIC REGRESSION (dengan SMOTE)")
print("="*50)

# Load data
df = pd.read_csv('data/PCOS_extended_dataset.csv')
target = 'PCOS (Y/N)'
X = df.drop(columns=[target, 'Sl. No', 'Patient File No.'])
y = df[target]

# Numeric only
X = X.select_dtypes(include=[np.number])

# Handle missing values
for col in X.columns:
    if X[col].isnull().any():
        X[col] = X[col].fillna(X[col].median())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardisasi
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Before SMOTE - PCOS: {sum(y_train)}, Non-PCOS: {len(y_train)-sum(y_train)}")

# SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print(f"After SMOTE - PCOS: {sum(y_train_balanced)}, Non-PCOS: {len(y_train_balanced)-sum(y_train_balanced)}")

# Training
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_balanced, y_train_balanced)

# Evaluasi
y_pred = model.predict(X_test_scaled)
print(f"\n📊 HASIL EVALUASI:")
print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
print(f"  Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"  F1-Score : {f1_score(y_test, y_pred):.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"  True Negatif : {cm[0,0]}")
print(f"  False Positif: {cm[0,1]}")
print(f"  False Negatif: {cm[1,0]}")
print(f"  True Positif : {cm[1,1]}")

# Simpan
joblib.dump(model, 'models/logistic_regression.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print("\n✅ Model dan scaler disimpan di folder 'models/'")