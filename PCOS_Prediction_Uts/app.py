from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
import json

app = Flask(__name__)

# ==================================================
# CLASS BACKPROPAGATION
# ==================================================
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
            if verbose and (epoch + 1) % 100 == 0:
                loss = np.mean(error ** 2)
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")

    def predict(self, X):
        output = self.forward(X)
        return (output > 0.5).astype(int).flatten()

    def predict_proba(self, X):
        output = self.forward(X)
        return np.hstack([1 - output, output])

# ==================================================
# LOAD MODEL
# ==================================================
print("=" * 50)
print("LOADING MODEL PCOS PREDICTOR")
print("=" * 50)

base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, 'models')

# Load scaler
scaler_path = os.path.join(models_dir, 'scaler.pkl')
if not os.path.exists(scaler_path):
    print(f"❌ Scaler tidak ditemukan di: {scaler_path}")
    exit(1)
scaler = joblib.load(scaler_path)
print("✅ Scaler berhasil dimuat")

# Load median values dari dataset
df_ref = pd.read_csv(os.path.join(base_dir, 'data', 'PCOS_extended_dataset.csv'))
target_col = 'PCOS (Y/N)'
X_ref = df_ref.drop(columns=[target_col, 'Sl. No', 'Patient File No.'])
X_ref = X_ref.select_dtypes(include=[np.number])
for col in X_ref.columns:
    if X_ref[col].isnull().any():
        X_ref[col] = X_ref[col].fillna(X_ref[col].median())
MEDIAN_VALUES = X_ref.median().values
print("✅ Median values berhasil dimuat")

MODEL_FILES = {
    'Logistic Regression': ('logistic_regression.pkl', 'pkl'),
    'K-Means':             ('kmeans.pkl',              'pkl'),
    'Backpropagation':     ('backpropagation.pkl',     'pkl'),
    'ANN':                 ('ann_model.h5',            'h5'),
    'RNN':                 ('rnn_model.h5',            'h5'),
}

loaded_models = {}
for name, (filename, tipe) in MODEL_FILES.items():
    path = os.path.join(models_dir, filename)
    if not os.path.exists(path):
        print(f"⚠️  {name} tidak ditemukan ({filename}), dilewati")
        continue
    try:
        if tipe == 'pkl':
            loaded_models[name] = ('pkl', joblib.load(path))
        elif tipe == 'h5':
            from tensorflow.keras.models import load_model
            loaded_models[name] = ('h5', load_model(path))
        print(f"✅ {name} berhasil dimuat")
    except Exception as e:
        print(f"❌ Gagal load {name}: {e}")

if not loaded_models:
    print("❌ Tidak ada model yang berhasil dimuat!")
    exit(1)

# ==================================================
# INDEX FITUR
# ==================================================
INDICES = {
    'Age':           0,
    'BMI':           3,
    'Cycle_length':  9,
    'FSH_LH':        16,
    'Follicle_No_L': 34,
    'Follicle_No_R': 35,
    'Endometrium':   38,
    'Weight_gain':   25,
    'Hair_growth':   26,
    'Hair_loss':     28,
    'Pimples':       29
}

# ==================================================
# HELPER: prediksi satu model
# ==================================================
def predict_one(name, tipe_model, model, input_scaled):
    try:
        if name == 'K-Means':
            cluster = model.predict(input_scaled)[0]
            pred = 1 if cluster == 1 else 0
            probability = 75.0 if pred == 1 else 25.0

        elif tipe_model == 'h5':
            # RNN butuh input 3D (1, 1, n_features), ANN butuh 2D (1, n_features)
            if name == 'RNN':
                model_input = input_scaled.reshape(1, 1, -1)
            else:
                model_input = input_scaled  # ANN: tetap 2D
            proba_raw = model.predict(model_input, verbose=0).flatten()[0]
            pred = 1 if proba_raw >= 0.5 else 0
            probability = round(float(proba_raw) * 100, 2)

        else:
            # pred_raw: 1 = PCOS Positif, 0 = PCOS Negatif (sesuai label training)
            pred_raw = int(model.predict(input_scaled)[0])
            pred = pred_raw  # langsung pakai, tidak dibalik
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_scaled)[0]
                probability = round(float(proba[1]) * 100, 2)  # proba[1] = probabilitas PCOS (kelas 1)
            else:
                probability = 75.0 if pred == 1 else 25.0

        return {
            'name': name,
            'prediction': 'PCOS Positif' if pred == 1 else 'PCOS Negatif',
            'pred_class': 'positif' if pred == 1 else 'negatif',
            'probability': probability
        }
    except Exception as e:
        print(f"⚠️  Error prediksi {name}: {e}")
        return None

# ==================================================
# ROUTES
# ==================================================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        feature_vector = MEDIAN_VALUES.copy()

        def get_form_value(field_name):
            value = request.form.get(field_name)
            if value is None or value == '':
                return None
            try:
                return float(value)
            except:
                return None

        val = get_form_value('Age')
        if val is not None:
            feature_vector[INDICES['Age']] = val

        val = get_form_value('BMI')
        if val is not None:
            feature_vector[INDICES['BMI']] = val

        val = get_form_value('Cycle_length')
        if val is not None:
            feature_vector[INDICES['Cycle_length']] = val

        val = get_form_value('FSH_LH')
        if val is not None:
            feature_vector[INDICES['FSH_LH']] = val

        val = get_form_value('Follicle_No')
        if val is not None and val > 0:
            feature_vector[INDICES['Follicle_No_L']] = val
            feature_vector[INDICES['Follicle_No_R']] = val

        val = get_form_value('Endometrium')
        if val is not None:
            feature_vector[INDICES['Endometrium']] = val

        val = get_form_value('Weight_gain')
        if val is not None:
            feature_vector[INDICES['Weight_gain']] = val

        val = get_form_value('Hair_growth')
        if val is not None:
            feature_vector[INDICES['Hair_growth']] = val

        val = get_form_value('Hair_loss')
        if val is not None:
            feature_vector[INDICES['Hair_loss']] = val

        val = get_form_value('Pimples')
        if val is not None:
            feature_vector[INDICES['Pimples']] = val

        input_scaled = scaler.transform(feature_vector.reshape(1, -1))

        # Prediksi semua model
        model_results = []
        for name, (tipe, model) in loaded_models.items():
            result = predict_one(name, tipe, model, input_scaled)
            if result:
                model_results.append(result)

        if not model_results:
            return jsonify({'error': 'Semua model gagal melakukan prediksi'}), 500

        # Hitung konsensus dengan bobot berdasarkan akurasi model
        MODEL_WEIGHTS = {
            'Logistic Regression': 3,
            'ANN':                 2,
            'RNN':                 2,
            'Backpropagation':     2,
            'K-Means':             1,
        }

        positif_count = sum(1 for r in model_results if r['pred_class'] == 'positif')
        negatif_count = len(model_results) - positif_count

        positif_score = sum(MODEL_WEIGHTS.get(r['name'], 1) for r in model_results if r['pred_class'] == 'positif')
        negatif_score = sum(MODEL_WEIGHTS.get(r['name'], 1) for r in model_results if r['pred_class'] == 'negatif')

        if positif_score >= negatif_score:
            consensus_text  = '⚠️ TERINDIKASI PCOS'
            consensus_class = 'positif'
        else:
            consensus_text  = '✅ TIDAK TERINDIKASI PCOS'
            consensus_class = 'negatif'

        return render_template('result.html',
                               results_json=json.dumps(model_results),
                               consensus_text=consensus_text,
                               consensus_class=consensus_class,
                               positif_count=positif_count,
                               negatif_count=negatif_count,
                               total_model=len(model_results))

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/compare')
def compare():
    training_results = {
        'Logistic Regression': {'accuracy': 0.9275, 'precision': 0.8605, 'recall': 0.9098, 'f1': 0.8845},
        'K-Means':             {'accuracy': 0.72,   'precision': 0.70,   'recall': 0.71,   'f1': 0.70},
        'ANN':                 {'accuracy': 0.85,   'precision': 0.84,   'recall': 0.83,   'f1': 0.84},
        'RNN':                 {'accuracy': 0.84,   'precision': 0.83,   'recall': 0.82,   'f1': 0.83},
        'Backpropagation':     {'accuracy': 0.83,   'precision': 0.82,   'recall': 0.81,   'f1': 0.82}
    }
    metrics = []
    for name, scores in training_results.items():
        metrics.append({'name': name, **scores})
    best = max(metrics, key=lambda x: x['accuracy'])
    return render_template('compare.html',
                           metrics=metrics,
                           best_model=best['name'],
                           best_accuracy_value=best['accuracy'])

if __name__ == '__main__':
    app.run(debug=True, port=5000)