import cv2
import matplotlib.pyplot as plt
import urllib.request

# Ganti dengan URL gambar yang valid
url = 'https://raw.githubusercontent.com/edisuwoto/yoloai/main/face/image4.jpg'
urllib.request.urlretrieve(url, 'image4.jpg')

# Load face detector Haarcascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Baca gambar
img = cv2.imread('image4.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Deteksi wajah
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Gambar kotak pada wajah yang terdeteksi
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Tampilkan hasil
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
