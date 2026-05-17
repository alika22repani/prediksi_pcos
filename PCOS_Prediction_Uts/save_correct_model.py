import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

print("="*50)
print("TRAINING MODEL BARU YANG BENAR")
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

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Standardisasi
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Training
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train_balanced)

# Evaluasi
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_pred = model.predict(X_test_scaled)
print(f"\n📊 EVALUASI MODEL:")
print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
print(f"  Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"  F1-Score : {f1_score(y_test, y_pred):.4f}")

# Simpan model dan scaler (OVERWRITE yang lama)
joblib.dump(model, 'models/logistic_regression.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("\n✅ Model dan scaler baru BERHASIL disimpan!")
print("   File: models/logistic_regression.pkl")
print("   File: models/scaler.pkl")

# Test dengan data PCOS
test_data = X_test[y_test == 1].iloc[0]
test_scaled = scaler.transform(test_data.values.reshape(1, -1))
proba = model.predict_proba(test_scaled)[0][1]
print(f"\n🔬 Test dengan data PCOS asli:")
print(f"   Probabilitas PCOS: {proba*100:.2f}% -> {'POSITIF' if proba > 0.5 else 'NEGATIF'}")