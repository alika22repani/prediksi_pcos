import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

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

print("📊 Fitur yang digunakan dalam training:")
for i, col in enumerate(X.columns):
    print(f"  {i}: {col}")

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Standardisasi
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Training model sederhana
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train_balanced)

# Test dengan data PCOS
test_data = X_test[y_test == 1].iloc[0]  # ambil 1 data PCOS dari test set
print(f"\n🔬 Test dengan data PCOS asli dari dataset:")
print(f"  Label asli: PCOS POSITIF")
print(f"  Data: {test_data.values[:5]}...")

# Standardisasi
test_scaled = scaler.transform(test_data.values.reshape(1, -1))

# Prediksi
pred = model.predict(test_scaled)[0]
proba = model.predict_proba(test_scaled)[0][1]

print(f"\n📊 HASIL PREDIKSI MODEL BARU:")
print(f"  Prediksi: {'PCOS POSITIF' if pred == 1 else 'PCOS NEGATIF'}")
print(f"  Probabilitas PCOS: {proba:.4f} ({proba*100:.2f}%)")

# Cek feature importance
print(f"\n📈 Feature Importance (10 fitur terpenting):")
coef = model.coef_[0]
feature_importance = list(zip(X.columns, coef))
feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
for feat, imp in feature_importance[:10]:
    print(f"  {feat}: {imp:.4f}")