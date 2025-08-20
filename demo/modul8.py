"""
Module 8 – Wrinkle Detection & Age Estimation (End-to-End, Offline Friendly)

Pipeline:
1) Generate dummy dataset wajah sintetis + label (wrinkle: 0/1, age: 18–70)
2) Train CNN multi-output (wrinkle classification + age regression)
3) (Optional) Face detection:
   - Try YOLOv8-face (if ultralytics available)
   - Else try Haar Cascade (OpenCV)
   - Else fallback: use full image as one face
4) Predict on a test image (dummy or real), draw bbox + label (Wrinkle/No Wrinkle, Age)

Run:
    python module8_wrinkle_age_end2end.py
"""

import os
import sys
import math
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Tuple

# ====== TensorFlow imports ======
import tensorflow as tf
from tensorflow.keras import layers, models

# -----------------------------
# Global configs
# -----------------------------
IMG_SIZE = 128
NUM_SAMPLES = 300        # kecil agar cepat training
EPOCHS = 3               # tambah jika ingin hasil lebih baik
BATCH_SIZE = 16
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# =====================================================================
# 1) Dummy Dataset Generator (wajah sintetis + kerutan + usia)
# =====================================================================

def draw_synthetic_face(canvas: np.ndarray, wrinkle: bool) -> np.ndarray:
    """
    Gambar "wajah" sederhana pada canvas:
    - oval untuk wajah
    - dua mata
    - mulut
    - garis-garis horizontal sebagai 'wrinkles' jika wrinkle=True
    """
    h, w, _ = canvas.shape
    # latar belakang abu
    canvas[:] = np.random.randint(130, 180, size=canvas.shape, dtype=np.uint8)

    # koordinat wajah (oval)
    center = (w//2, h//2)
    axes = (w//3, h//2 - 10)
    face_color = (220, 200, 180)  # skin-ish
    cv2.ellipse(canvas, center, axes, 0, 0, 360, face_color, thickness=-1)

    # mata
    eye_y = h//2 - 20
    eye_dx = 25
    cv2.circle(canvas, (center[0] - eye_dx, eye_y), 8, (30, 30, 30), -1)
    cv2.circle(canvas, (center[0] + eye_dx, eye_y), 8, (30, 30, 30), -1)

    # mulut
    cv2.ellipse(canvas, (center[0], h//2 + 25), (25, 10), 0, 0, 180, (50, 50, 50), 2)

    # rambut sederhana
    cv2.ellipse(canvas, (center[0], h//2 - axes[1]//2 - 30), (axes[0], 20), 0, 180, 360, (50, 40, 30), -1)

    if wrinkle:
        # Tambah beberapa garis horizontal sebagai "kerutan"
        n_lines = np.random.randint(5, 12)
        for _ in range(n_lines):
            y = np.random.randint(h//2 - 5, h//2 + 20)
            x1 = center[0] - np.random.randint(10, axes[0] - 5)
            x2 = center[0] + np.random.randint(10, axes[0] - 5)
            cv2.line(canvas, (x1, y), (x2, y), (60, 60, 60), 1)
    return canvas


def generate_dummy_dataset(num_samples: int = NUM_SAMPLES) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    images, wrinkles, ages = [], [], []
    for _ in range(num_samples):
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        wrinkle_label = np.random.randint(0, 2)       # 0=no wrinkle, 1=wrinkle
        age_label = np.random.randint(18, 70)         # pseudo age

        img = draw_synthetic_face(img, wrinkle=bool(wrinkle_label))

        images.append(img)
        wrinkles.append(wrinkle_label)
        ages.append(age_label)

    images = np.array(images).astype("float32") / 255.0
    wrinkles = np.array(wrinkles).astype(np.float32)
    ages = np.array(ages).astype(np.float32)
    return images, wrinkles, ages


# =====================================================================
# 2) Model: CNN Multi-Output (wrinkle cls + age regression)
# =====================================================================

def build_wrinkle_age_model(input_shape=(IMG_SIZE, IMG_SIZE, 3)) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    wrinkle_out = layers.Dense(1, activation='sigmoid', name='wrinkle')(x)
    age_out = layers.Dense(1, activation='linear', name='age')(x)

    model = models.Model(inputs=inputs, outputs=[wrinkle_out, age_out])
    model.compile(
        optimizer='adam',
        loss={'wrinkle': 'binary_crossentropy', 'age': 'mse'},
        metrics={'wrinkle': 'accuracy', 'age': 'mae'}
    )
    return model


# =====================================================================
# 3) (Optional) Face Detector: YOLOv8-face → Haar Cascade → Mock
# =====================================================================

def try_import_ultralytics():
    try:
        from ultralytics import YOLO  # type: ignore
        return YOLO
    except Exception:
        return None


def detect_faces_yolo(image_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    YOLO = try_import_ultralytics()
    boxes = []
    if YOLO is None:
        return boxes
    try:
        # pakai model face ringan (butuh internet saat pertama kali untuk unduh)
        model = YOLO("yolov8n-face.pt")
        results = model.predict(image_bgr, verbose=False)
        for b in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, b[:4])
            boxes.append((x1, y1, x2, y2))
    except Exception:
        # jika gagal (mis: offline), return kosong
        pass
    return boxes


def detect_faces_haar(image_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    boxes = []
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                              minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in faces:
            boxes.append((x, y, x + w, y + h))
    except Exception:
        pass
    return boxes


def detect_faces(image_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    # 1) YOLO jika tersedia
    boxes = detect_faces_yolo(image_bgr)
    if boxes:
        return boxes
    # 2) Haar Cascade
    boxes = detect_faces_haar(image_bgr)
    if boxes:
        return boxes
    # 3) Fallback: anggap seluruh gambar adalah satu wajah
    h, w = image_bgr.shape[:2]
    return [(0, 0, w, h)]


# =====================================================================
# 4) Utils Plot & Inference
# =====================================================================

def plot_grid(images: np.ndarray, titles: List[str] = None, cols: int = 4, savepath: str = None):
    n = len(images)
    rows = math.ceil(n / cols)
    plt.figure(figsize=(cols * 3, rows * 3))
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(cv2.cvtColor((images[i] * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        if titles:
            plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=150)
        print(f"Saved: {savepath}")
    plt.show()


def draw_boxes_with_labels(image_bgr: np.ndarray,
                           boxes: List[Tuple[int, int, int, int]],
                           labels: List[str]) -> np.ndarray:
    img = image_bgr.copy()
    for (x1, y1, x2, y2), label in zip(boxes, labels):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(img, label, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    return img


# =====================================================================
# 5) Main: train + demo inference
# =====================================================================

def main():
    os.makedirs("outputs", exist_ok=True)

    # ---- Generate dummy dataset ----
    print("[1/5] Generating dummy dataset...")
    X, y_wrinkle, y_age = generate_dummy_dataset(NUM_SAMPLES)

    # preview
    plot_grid(X[:8], titles=[f"W:{int(y_wrinkle[i])}, Age:{int(y_age[i])}" for i in range(8)],
              savepath="outputs/dummy_preview.png")

    # ---- Train/val split ----
    print("[2/5] Splitting dataset...")
    n = len(X)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(0.8 * n)
    tr, va = idx[:split], idx[split:]
    X_train, X_val = X[tr], X[va]
    w_train, w_val = y_wrinkle[tr], y_wrinkle[va]
    a_train, a_val = y_age[tr], y_age[va]

    # ---- Build & Train model ----
    print("[3/5] Building and training model...")
    model = build_wrinkle_age_model(input_shape=(IMG_SIZE, IMG_SIZE, 3))
    model.summary()

    history = model.fit(
        X_train, {"wrinkle": w_train, "age": a_train},
        validation_data=(X_val, {"wrinkle": w_val, "age": a_val}),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    model.save("outputs/wrinkle_age_model.h5")
    print("Saved model: outputs/wrinkle_age_model.h5")

    # ---- Plot training curves ----
    plt.figure()
    plt.plot(history.history["wrinkle_accuracy"], label="wrinkle_acc")
    plt.plot(history.history["val_wrinkle_accuracy"], label="val_wrinkle_acc")
    plt.legend()
    plt.title("Wrinkle Accuracy")
    plt.savefig("outputs/train_wrinkle_acc.png", dpi=150)
    plt.show()

    plt.figure()
    plt.plot(history.history["age_mae"], label="age_mae")
    plt.plot(history.history["val_age_mae"], label="val_age_mae")
    plt.legend()
    plt.title("Age MAE")
    plt.savefig("outputs/train_age_mae.png", dpi=150)
    plt.show()

    # ---- Demo inference ----
    print("[4/5] Running demo inference...")
    # Buat 1 "foto" wajah (dummy)
    test_img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    test_img = draw_synthetic_face(test_img, wrinkle=bool(np.random.randint(0, 2)))
    test_bgr = (test_img).astype(np.uint8)  # already 0..255 BGR-ish drawing
    test_rgb = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2RGB)

    # Deteksi wajah (YOLO → Haar → Fallback)
    boxes = detect_faces(test_bgr)

    labels = []
    for (x1, y1, x2, y2) in boxes:
        crop = test_bgr[max(y1,0):max(y2,0), max(x1,0):max(x2,0)]
        if crop.size == 0:
            # jika bbox invalid, pakai full image
            crop = test_bgr.copy()
        crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
        crop_in = (crop.astype(np.float32) / 255.0)[None, ...]
        pred_wrinkle, pred_age = model.predict(crop_in, verbose=0)
        wr = "Wrinkled" if pred_wrinkle[0,0] > 0.5 else "No Wrinkle"
        age = int(np.clip(pred_age[0,0], 0, 100))
        labels.append(f"{wr}, Age {age}")

    result = draw_boxes_with_labels(test_bgr, boxes, labels)
    cv2.imwrite("outputs/demo_result.jpg", result)
    print("Saved demo: outputs/demo_result.jpg")

    # tampilkan
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title("Demo Result")
    plt.axis("off")
    plt.show()

    print("[5/5] Done ✅")


if __name__ == "__main__":
    # Optional: tips untuk Mac ARM – gunakan tensorflow-macos + tensorflow-metal
    # pip install tensorflow-macos tensorflow-metal
    main()
