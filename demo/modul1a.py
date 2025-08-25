import cv2
import matplotlib.pyplot as plt
import urllib.request

# 1. Download gambar dari internet
url = 'https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png'
urllib.request.urlretrieve(url, 'sample.png')

# 2. Baca gambar dengan OpenCV (BGR)
img = cv2.imread('sample.png')

# 3. Konversi ke RGB agar sesuai dengan Matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 4. Ubah gambar jadi grayscale untuk deteksi tepi
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 5. Deteksi tepi dengan Canny
edges = cv2.Canny(img_gray, threshold1=100, threshold2=200)

# 6. Tampilkan hasil
plt.figure(figsize=(10,5))

# Gambar asli
plt.subplot(1,2,1)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

# Hasil deteksi tepi
plt.subplot(1,2,2)
plt.imshow(edges, cmap='gray')
plt.title('Edge Detection (Canny)')
plt.axis('off')

plt.show()
