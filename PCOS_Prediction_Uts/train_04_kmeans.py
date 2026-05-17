# train_04_kmeans.py
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import joblib
from utils import load_and_preprocess

print("="*50)
print("ALGORITMA 4: K-MEANS CLUSTERING")
print("="*50)

# Load data (K-Meams tidak pakai y_train)
X_train, X_test, y_train, y_test, feature_names = load_and_preprocess()

# Training
print("\nTraining K-Means Clustering...")
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X_train)

# Evaluasi pada test set
y_pred_kmeans = kmeans.predict(X_test)

# Metrik untuk clustering
silhouette = silhouette_score(X_test, y_pred_kmeans)
rand_score = adjusted_rand_score(y_test, y_pred_kmeans)

print(f"\nHasil Evaluasi Clustering:")
print(f"  Silhouette Score    : {silhouette:.4f}")
print(f"  Adjusted Rand Score : {rand_score:.4f}")
print(f"\nCluster distribution pada test set:")
print(f"  Cluster 0: {sum(y_pred_kmeans == 0)} data")
print(f"  Cluster 1: {sum(y_pred_kmeans == 1)} data")
print(f"\nCluster vs Label sebenarnya:")
print(f"  Cluster 0 -> PCOS=0: {sum((y_pred_kmeans == 0) & (y_test == 0))}")
print(f"  Cluster 0 -> PCOS=1: {sum((y_pred_kmeans == 0) & (y_test == 1))}")
print(f"  Cluster 1 -> PCOS=0: {sum((y_pred_kmeans == 1) & (y_test == 0))}")
print(f"  Cluster 1 -> PCOS=1: {sum((y_pred_kmeans == 1) & (y_test == 1))}")

# Simpan model
joblib.dump(kmeans, 'models/kmeans.pkl')
print("\n✅ Model disimpan di: models/kmeans.pkl")