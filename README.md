# Website Prediksi Polycystic Ovary Syndrome (PCOS) Menggunakan 5 Algoritma Machine Learning

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Flask](https://img.shields.io/badge/Flask-3.x-lightgrey)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange)
![Railway](https://img.shields.io/badge/Deploy-Railway-purple)
![Status](https://img.shields.io/badge/Status-Active-green)

> Mata Kuliah: Praktikum Kecerdasan Buatan  
> Universitas Bale Bandung

🌐 **Live Demo**: [pcosscan.my.id](https://pcosscan.my.id)

---

## 📌 Deskripsi

Website ini merupakan sistem prediksi dini **Polycystic Ovary Syndrome (PCOS)** berbasis web yang menggunakan 5 algoritma machine learning secara bersamaan. Hasil prediksi ditentukan melalui mekanisme **voting berbobot** dari seluruh model untuk menghasilkan keputusan yang lebih akurat dan andal.

PCOS adalah gangguan hormonal yang umum dialami wanita usia reproduktif. Deteksi dini sangat penting untuk mencegah komplikasi jangka panjang seperti diabetes, infertilitas, dan penyakit jantung.

---

## 🤖 Algoritma yang Digunakan

| No | Algoritma | Akurasi | Tipe |
|----|-----------|---------|------|
| 1 | Logistic Regression | 92.75% | Supervised |
| 2 | Artificial Neural Network (ANN) | 85.00% | Supervised |
| 3 | Recurrent Neural Network (RNN) | 84.00% | Supervised |
| 4 | Backpropagation | 83.00% | Supervised |
| 5 | K-Means Clustering | 72.00% | Unsupervised |

> Prediksi akhir menggunakan **voting berbobot** berdasarkan akurasi masing-masing model.

---

## 🩺 Fitur Input

Website menerima input parameter klinis berikut:

- **Usia** (Age)
- **BMI** (Body Mass Index)
- **Panjang Siklus Menstruasi** (Cycle Length)
- **Rasio FSH/LH**
- **Jumlah Folikel** (Follicle Number)
- **Ketebalan Endometrium**
- **Kenaikan Berat Badan** (Weight Gain)
- **Pertumbuhan Rambut Berlebih** (Hair Growth)
- **Kerontokan Rambut** (Hair Loss)
- **Jerawat** (Pimples)

---

## 🗂️ Struktur Folder

```
├── app.py                          # Main Flask application
├── utils.py                        # Preprocessing & helper functions
├── requirements.txt                # Python dependencies
├── Procfile                        # Railway deployment config
├── train_01_logistic_regression.py # Training Logistic Regression
├── train_02_ann.py                 # Training ANN
├── train_03_rnn.py                 # Training RNN
├── train_04_kmeans.py              # Training K-Means
├── train_05_backpropagation.py     # Training Backpropagation
├── models/
│   ├── scaler.pkl
│   ├── logistic_regression.pkl
│   ├── kmeans.pkl
│   ├── backpropagation.pkl
│   ├── ann_model.h5
│   └── rnn_model.h5
├── data/
│   └── PCOS_extended_dataset.csv
├── templates/
│   ├── index.html
│   ├── result.html
│   └── compare.html
└── static/
```

---

## ⚙️ Cara Menjalankan Secara Lokal

**1. Clone repository**
```bash
git clone https://github.com/username/prediksi_pcos.git
cd prediksi_pcos
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Jalankan aplikasi**
```bash
python app.py
```

**4. Buka browser**
```
http://localhost:5000
```

---

## 📦 Dependencies

```
flask
numpy
pandas
scikit-learn
imbalanced-learn
tensorflow-cpu==2.20.0
joblib
gunicorn
```

---

## 🚀 Deployment

Website di-deploy menggunakan **Railway** dengan domain custom **pcosscan.my.id**.

---

## 📊 Dataset

Dataset yang digunakan adalah **PCOS Extended Dataset** yang berisi data klinis pasien dengan dan tanpa PCOS. Data di-preprocess menggunakan:
- Penanganan missing values dengan median
- SMOTE untuk mengatasi imbalanced dataset
- StandardScaler untuk normalisasi fitur

---

## ⚠️ Disclaimer

Website ini dibuat untuk keperluan **akademis** dan tidak menggantikan diagnosis medis profesional. Hasil prediksi hanya bersifat indikatif. Konsultasikan selalu dengan dokter atau tenaga medis untuk diagnosis yang akurat.
