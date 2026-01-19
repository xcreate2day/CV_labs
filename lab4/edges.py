# %%
import cv2
import numpy as np
from kernels import *
from tools import *


# Loading image
# working = "C:/Users/xcrea/Documents/YandexPython/pythonProject/img/cats2.jpg"
# I = cv2.cvtColor(cv2.imread(working), cv2.COLOR_RGB2GRAY)

working = load_image("C:/Users/xcrea/Documents/YandexPython/pythonProject/img/cats2.jpg")
I = rgb2gray(working)

WITH_FIELDS = False

# showGrayInput(I)

# Roberts
method = robertsKernels
Gx, Gy = grad(I, method)

# G1, G2, Magnitude = showGrayGrad(Gx, Gy, WITH_FIELDS, (10, 10), original=I)
# res1 = showGray(edgesGrad(I, method))
res1 = edgesGrad(I, method)

# images = [G1, G2, Magnitude, res1]
# titles = ['Свертка х', 'Свертка у', 'Сумма', 'результат (Roberts)']
# show_images(images, titles)


# # Prewitt
method = prewittKernels
Gx, Gy = grad(I, method)

# G1, G2, Magnitude = showGrayGrad(Gx, Gy, WITH_FIELDS, (10, 10), original=I)
# res2 = showGray(edgesGrad(I, method))
res2 = edgesGrad(I, method)

# images = [G1, G2, Magnitude, res2]
# titles = ['Свертка х', 'Свертка у', 'Сумма', 'результат (Prewitt)']
# show_images(images, titles)


# # Sobel
method = sobelKernels
Gx, Gy = grad(I, method)

# G1, G2, Magnitude = showGrayGrad(Gx, Gy, WITH_FIELDS, (10, 10), original=I)
# res3 = showGray(edgesGrad(I, method))
res3 = edgesGrad(I, method)

# images = [G1, G2, Magnitude, res3]
# titles = ['Свертка х', 'Свертка у', 'Сумма', 'результат (Sobel)']
# show_images(images, titles)


# # Scharr
method = scharrKernels
Gx, Gy = grad(I, method)

# G1, G2, Magnitude = showGrayGrad(Gx, Gy, WITH_FIELDS, (10, 10), original=I)
# res4 = showGray(edgesGrad(I, method))
res4 = edgesGrad(I, method)

# images = [G1, G2, Magnitude, res4]
# titles = ['Свертка х', 'Свертка у', 'Сумма', 'результат (Scharr)']
# show_images(images, titles)


# Laplace

# working = "C:/Users/xcrea/Documents/YandexPython/pythonProject/img/cats2.jpg"
# I = cv2.cvtColor(cv2.imread(working), cv2.COLOR_RGB2GRAY)

LK = laplaceKernel()
L1 = laplacian_edge_detection(I, 8, LK)
# showGrayInput(L1)

# working = "C:/Users/xcrea/Documents/YandexPython/pythonProject/img/cats2.jpg"
# I = cv2.cvtColor(cv2.imread(working), cv2.COLOR_RGB2GRAY)

LK = laplaceKernel()
L2 = laplacian_edge_detection(I, 12, LK)
# showGrayInput(L2)


# Canny

# img = cv2.imread('messi5.jpg', cv2.IMREAD_GRAYSCALE)
# assert I is not None, "file could not be read, check with os.path.exists()"
edges = cv2.Canny(I, 120, 250)
# plt.figure(figsize=(14, 6))
# plt.subplot(121), plt.imshow(I, cmap='gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(edges, cmap='gray')
# plt.title('Edge Image (Canny)'), plt.xticks([]), plt.yticks([])

# plt.show()

# Отображение результатов
fig, axes = plt.subplots(2, 4, figsize=(24, 8))
fig.suptitle("Все вместе, итоговые", fontsize=16, fontweight='bold')

axes[0][0].imshow(res1, cmap='gray')
axes[0][0].set_title("Робертс", fontsize=12)
axes[0][0].axis('off')

axes[0][1].imshow(res2, cmap='gray')
axes[0][1].set_title("Прюитт", fontsize=12)
axes[0][1].axis('off')

axes[0][2].imshow(res3, cmap='gray')
axes[0][2].set_title("Собель", fontsize=12)
axes[0][2].axis('off')

axes[0][3].imshow(res4, cmap='gray')
axes[0][3].set_title("Щарр", fontsize=12)
axes[0][3].axis('off')

axes[1][0].imshow(L1, cmap='gray')
axes[1][0].set_title("Лапласс 8%", fontsize=12)
axes[1][0].axis('off')

axes[1][1].imshow(L2, cmap='gray')
axes[1][1].set_title("Лапласс 12%", fontsize=12)
axes[1][1].axis('off')

axes[1][2].imshow(edges, cmap='gray')
axes[1][2].set_title("Кэнни", fontsize=12)
axes[1][2].axis('off')

axes[1][3].imshow(I, cmap='gray')
axes[1][3].set_title("Исходное", fontsize=12)
axes[1][3].axis('off')

plt.tight_layout()
plt.show()






