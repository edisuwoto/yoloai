import cv2
import matplotlib.pyplot as plt
import urllib.request

url = 'https://raw.githubusercontent.com/edisuwoto/yoloai/main/face/Lenna.png'
urllib.request.urlretrieve(url, 'Lenna.png')

img = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(img, 100, 200)

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.subplot(1,2,2)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edges')
plt.show()