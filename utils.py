# utils.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess():
    """Load dataset, preprocessing, dan return data siap training"""
    print("Loading dataset...")
    df = pd.read_csv('data/PCOS_extended_dataset.csv')
    
    # Target
    target = 'PCOS (Y/N)'
    
    # Drop kolom ID dan target
    X = df.drop(columns=[target, 'Sl. No', 'Patient File No.'])
    y = df[target].copy()
    
    # Ambil hanya kolom numeric
    X = X.select_dtypes(include=[np.number]).copy()
    
    # ========== HANDLING MISSING VALUES - VERSION 2 (AMAN) ==========
    print("\nCek missing values sebelum handling:")
    missing_before = X.isnull().sum()
    print(f"  Total missing: {missing_before.sum()}")
    
    # Cara aman: isi semua missing dengan median per kolom (tanpa inplace)
    for col in X.columns:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            print(f"  Mengisi kolom '{col}' dengan median: {median_val:.2f}")
    
    # Cek lagi
    missing_after = X.isnull().sum().sum()
    print(f"\nTotal missing setelah handling: {missing_after}")
    
    if missing_after > 0:
        print("⚠️ MASIH ADA MISSING! Drop baris yang bermasalah...")
        X = X.dropna()
        y = y[X.index]
        print(f"  Shape setelah drop: {X.shape}")
    
    # ========== SPLIT DATA ==========
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # ========== STANDARDISASI ==========
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Simpan scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print(f"\n✅ Preprocessing selesai!")
    print(f"  Train size: {X_train_scaled.shape}")
    print(f"  Test size: {X_test_scaled.shape}")
    print(f"  PCOS di train (1): {y_train.sum()} dari {len(y_train)}")
    print(f"  PCOS di test (1): {y_test.sum()} dari {len(y_test)}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns