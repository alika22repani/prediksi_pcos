import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import joblib
import os

os.makedirs('models', exist_ok=True)

print("="*50)
print("ALGORITMA 5: BACKPROPAGATION (dengan SMOTE)")
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

# Backpropagation class
class Backpropagation:
    def __init__(self, input_size, hidden_size=16, learning_rate=0.01):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, 1) * 0.5
        self.b2 = np.zeros((1, 1))
        self.lr = learning_rate
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def train(self, X, y, epochs=300, verbose=True):
        y = y.values.reshape(-1, 1)
        for epoch in range(epochs):
            output = self.forward(X)
            error = y - output
            d_output = error * output * (1 - output)
            d_hidden = np.dot(d_output, self.W2.T) * self.a1 * (1 - self.a1)
            self.W2 += self.lr * np.dot(self.a1.T, d_output)
            self.b2 += self.lr * np.sum(d_output, axis=0, keepdims=True)
            self.W1 += self.lr * np.dot(X.T, d_hidden)
            self.b1 += self.lr * np.sum(d_hidden, axis=0, keepdims=True)
            if verbose and (epoch+1) % 100 == 0:
                loss = np.mean(error**2)
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")

# Training
bp = Backpropagation(input_size=X_train_balanced.shape[1], hidden_size=16, learning_rate=0.05)
bp.train(X_train_balanced, y_train_balanced, epochs=300, verbose=True)

# Evaluasi
y_pred = (bp.forward(X_test_scaled) > 0.5).astype(int)
print(f"\n📊 HASIL EVALUASI:")
print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
print(f"  Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"  F1-Score : {f1_score(y_test, y_pred):.4f}")

joblib.dump(bp, 'models/backpropagation.pkl')
print("\n✅ Model Backpropagation disimpan di 'models/backpropagation.pkl'")