import cv2
import matplotlib.pyplot as plt
import urllib.request

# URL valid untuk gambar Lenna
url = "https://raw.githubusercontent.com/edisuwoto/yoloai/main/face/lena.png"
urllib.request.urlretrieve(url, "lenna.png")

# Baca gambar dalam grayscale
img = cv2.imread("lenna.png", cv2.IMREAD_GRAYSCALE)

# Aplikasikan Gaussian Blur
blur = cv2.GaussianBlur(img, (5,5), 0)

# Tampilkan hasil
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(blur, cmap="gray")
plt.title("Gaussian Blur")
plt.axis("off")

plt.show()