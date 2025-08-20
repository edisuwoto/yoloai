from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2   # <--- ini yang kemarin belum ada

# Load model YOLO
model = YOLO("yolov8n.pt")

# Inference pada gambar
results = model("bus.jpg")

# Ambil hasil visualisasi (bounding box, label, dll)
img = results[0].plot()

# Tampilkan pakai matplotlib (portable di Mac/Windows/Linux)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
