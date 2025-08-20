import cv2
import matplotlib.pyplot as plt
import urllib.request

url = 'https://raw.githubusercontent.com/edisuwoto/yoloai/main/face/Lenna.png'
urllib.request.urlretrieve(url, 'Lenna.png')

img = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)
ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.subplot(1,2,2)
plt.imshow(thresh, cmap='gray')
plt.title('Thresholded')
plt.show()