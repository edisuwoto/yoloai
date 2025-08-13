
# Pelatihan Computer Vision dengan Python, OpenCV, TensorFlow, YOLO, dan Google Colab

Repositori ini berisi kumpulan notebook pelatihan **Computer Vision** berbasis YOLO dan deep learning, disusun untuk memudahkan pembelajaran mulai dari dasar hingga implementasi studi kasus.

📌 **Repo GitHub:** [https://github.com/edisuwoto/yoloai](https://github.com/edisuwoto/yoloai)

---

## 📂 Struktur Modul

### 1. **Dasar Computer Vision**
- Pengenalan citra digital (RGB, Grayscale, dll)
- Operasi dasar: cropping, resizing, filtering
- Visualisasi dengan Matplotlib dan OpenCV

### 2. **Pengenalan YOLO (You Only Look Once)**
- Arsitektur YOLO
- Perbandingan YOLOv3, YOLOv4, YOLOv5, YOLOv8
- Demo inference dengan model pre-trained

### 3. **Persiapan Dataset**
- Labeling data dengan LabelImg/Roboflow
- Augmentasi data
- Struktur folder untuk training YOLO

### 4. **Training Model YOLO**
- Konfigurasi hyperparameter
- Transfer learning
- Training di Google Colab / lokal GPU

### 5. **Evaluasi Model**
- mAP (Mean Average Precision)
- Precision-Recall Curve
- Confusion Matrix

### 6. **Optimisasi & Deployment**
- Konversi model ke format ONNX / TensorRT
- Optimasi inference speed
- Deployment ke server / edge device

### 7. **Studi Kasus 1: Deteksi Kebakaran**
- Dataset deteksi api
- Training model YOLO untuk kebakaran
- Integrasi ke sistem CCTV

### 8. **Studi Kasus 2: Deteksi Kerutan Wajah & Analisa Usia**
- Dataset kerutan wajah & age prediction
- Training multi-task YOLO (deteksi area + klasifikasi usia)
- Demo real-time dengan webcam

---

## 📌 Demo Deteksi Kerutan Wajah & Analisa Usia

Notebook ini memanfaatkan YOLO untuk mendeteksi wajah dan CNN tambahan untuk:
- Menghitung tingkat kerutan
- Memprediksi rentang usia

**Contoh Output:**
- Bounding box pada wajah
- Label: "Wrinkle Level: Medium, Age: 32-38"

---

## 🚀 Cara Menjalankan di Google Colab

1. **Buka link Colab** sesuai modul  
   Contoh:
- [1 Pengenalan Computer Vision](https://colab.research.google.com/github/edisuwoto/yoloai/blob/main/Module_1_Intro_Computer_Vision.ipynb) 
- [2 Operasi Dasar Gambar](https://colab.research.google.com/github/edisuwoto/yoloai/blob/main/Module_2_Basic_Image_Operations.ipynb) 
- [3 Filtering & Edge Detection](https://colab.research.google.com/github/edisuwoto/yoloai/blob/main/Module_3_Filtering_Edge_Detection.ipynb) 
- [4 Haar Cascade Detection](https://colab.research.google.com/github/edisuwoto/yoloai/blob/main/Module_4_Haar_Cascade_Detection.ipynb) 
- [5 Pengenalan YOLO](https://colab.research.google.com/github/edisuwoto/yoloai/blob/main/Module_5_Intro_YOLO.ipynb) 
- [6 Training YOLO Custom Dataset](https://colab.research.google.com/github/edisuwoto/yoloai/blob/main/Module_6_Train_YOLO_Custom_Dataset.ipynb) 
- [7 Testing & Evaluasi YOLO](https://colab.research.google.com/github/edisuwoto/yoloai/blob/main/Module_7_Test_Evaluate_YOLO.ipynb) 

- [Module 8 - Wrinkle Detection & Age Analysis](https://colab.research.google.com/github/edisuwoto/yoloai/blob/main/module8_wrinkle_age.ipynb)

2. **Klik "Copy to Drive"** untuk membuat salinan
3. **Jalankan sel kode** dari atas ke bawah

---

## 📦 Instalasi Lokal
```bash
git clone https://github.com/edisuwoto/yoloai.git
cd yoloai
pip install -r requirements.txt

📧 **Author:** Edi Suwoto  
🔗 **Repo:** [https://github.com/edisuwoto/yoloai](https://github.com/edisuwoto)
