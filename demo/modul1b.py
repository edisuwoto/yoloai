import cv2
import matplotlib.pyplot as plt
import urllib.request

# 1. Download gambar dari internet
url = 'https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png'
urllib.request.urlretrieve(url, 'sample.png')

# 2. Baca gambar dengan OpenCV (BGR)
img = cv2.imread('sample.png')

# 3. Konversi ke RGB agar sesuai Matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 4. Ubah gambar ke grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 5. Deteksi tepi (Canny)
edges = cv2.Canny(img_gray, 100, 200)

# 6. Cari kontur dari hasil tepi
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 7. Gambar kontur di atas gambar asli
img_contours = img_rgb.copy()
cv2.drawContours(img_contours, contours, -1, (255, 0, 0), 2)  # warna biru, tebal 2px

# 8. Bisa juga gambar bounding box tiap kontur
img_boxes = img_rgb.copy()
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img_boxes, (x,y), (x+w,y+h), (0,255,0), 2)  # hijau

# 9. Tampilkan hasil
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(edges, cmap='gray')
plt.title('Edge Detection (Canny)')
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(img_contours)
plt.title('Contours')
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(img_boxes)
plt.title('Bounding Boxes')
plt.axis('off')

plt.show()
