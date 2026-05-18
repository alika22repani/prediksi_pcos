import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

os.makedirs('models', exist_ok=True)

print("="*50)
print("ALGORITMA 2: ARTIFICIAL NEURAL NETWORK (ANN) dengan SMOTE")
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

# Standardisasi
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Before SMOTE - PCOS: {sum(y_train)}, Non-PCOS: {len(y_train)-sum(y_train)}")

# SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print(f"After SMOTE - PCOS: {sum(y_train_balanced)}, Non-PCOS: {len(y_train_balanced)-sum(y_train_balanced)}")

# Build model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_balanced.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Training
history = model.fit(X_train_balanced, y_train_balanced, 
                    epochs=100, batch_size=32, 
                    validation_split=0.2, verbose=1)

# Evaluasi
y_pred = (model.predict(X_test_scaled, verbose=0) > 0.5).astype(int)
print(f"\n📊 HASIL EVALUASI:")
print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
print(f"  Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"  F1-Score : {f1_score(y_test, y_pred):.4f}")

model.save('models/ann_model.h5')
print("\n✅ Model ANN disimpan di 'models/ann_model.h5'")