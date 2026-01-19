import os

import cv2
import matplotlib.pyplot as plt

from tools import custom_convolution, gaussian_kernel, showGray

dir = "C:/Users/xcrea/Documents/YandexPython/proj3/ranghigs/lab7"
f1v1 = os.path.join(dir, "foto1_var1.jpg")
f1v2 = os.path.join(dir, "foto1_var2.jpg")
f2v1 = os.path.join(dir, "foto6_var1.jpg")
f2v2 = os.path.join(dir, "foto6_var2.jpg")

# Загрузка изображений в градациях серого
img1 = cv2.imread(f2v1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(f2v2, cv2.IMREAD_GRAYSCALE)


kernel_size = 15
sigma = 3.0

gauss_kernel = gaussian_kernel(kernel_size, sigma)

image_blurred1 = custom_convolution(img1, gauss_kernel)
showGray(image_blurred1, maxheight=500)

image_blurred2 = custom_convolution(img2, gauss_kernel)
showGray(image_blurred2, maxheight=500)

# Инициализация SIFT
sift = cv2.SIFT_create(nfeatures=128)
keypoints1 = sift.detect(img1, None)
keypoints2 = sift.detect(img2, None)

imgKeyPoints1 = cv2.drawKeypoints(
    img1, keypoints1, img1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)
imgKeyPoints2 = cv2.drawKeypoints(
    img2, keypoints2, img2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(imgKeyPoints1, cmap="gray")
plt.title("Изображение 1")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(imgKeyPoints2, cmap="gray")
plt.title("Изображение 2")
plt.axis("off")

plt.tight_layout()
plt.show()

# Нахождение ключевых точек и дескрипторов
kp1, des1 = sift.detectAndCompute(image_blurred1, None)
kp2, des2 = sift.detectAndCompute(image_blurred2, None)

# Сопоставление дескрипторов с помощью BFMatcher
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Применение ratio test по Лоу
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:  # порог соответствий
        good_matches.append(m)

# Отрисовка совпадений
result = cv2.drawMatches(
    img1,
    kp1,
    img2,
    kp2,
    good_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)


plt.figure(figsize=(16, 8))
plt.imshow(result)
plt.title("SIFT matching")
plt.axis("off")
plt.show()
