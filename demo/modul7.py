import cv2
import matplotlib.pyplot as plt
import os
import urllib.request

# URL video sample dari opencv_extra
video_url = "https://raw.githubusercontent.com/opencv/opencv_extra/4.x/testdata/cv/tracking/faceocc2/data/faceocc2.webm"
video_path = "faceocc2.webm"

# Unduh video jika belum ada
if not os.path.exists(video_path):
    try:
        print("Mengunduh video contoh...")
        urllib.request.urlretrieve(video_url, video_path)
        print("Download selesai.")
    except Exception as e:
        print("Gagal download:", e)
        video_path = 0  # fallback ke webcam
        print("Gunakan stream webcam sebagai fallback.")

cap = cv2.VideoCapture(video_path)

# Pilih tracker
tracker = cv2.legacy.TrackerCSRT_create()

ret, frame = cap.read()
if not ret:
    print("Tidak bisa membuka video. Coba buka webcam.")
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Gagal membuka webcam juga.")

# ROI manual via mouse (GUI) atau hardcode
try:
    bbox = cv2.selectROI("Pilih Objek", frame, False)
    cv2.destroyAllWindows()
except:
    bbox = (300, 200, 100, 150)  # default frame crop, sesuaikan
    print("ROI tidak dapat dipilih secara manual — menggunakan area default.")

tracker.init(frame, bbox)

frames = []
for _ in range(30):
    ret, frame = cap.read()
    if not ret:
        break
    success, box = tracker.update(frame)
    if success:
        x, y, w, h = map(int, box)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()

# Tampilkan beberapa frame hasil tracking
plt.figure(figsize=(15,6))
for i in range(min(len(frames), 6)):
    plt.subplot(2,3,i+1)
    plt.imshow(frames[i])
    plt.axis("off")
plt.show()
