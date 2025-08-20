import cv2
import matplotlib.pyplot as plt
import urllib.request

url = 'https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png'
urllib.request.urlretrieve(url, 'sample.png')

img = cv2.imread('sample.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.title('Sample Image')
plt.axis('off')
plt.show()