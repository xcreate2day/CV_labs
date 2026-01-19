# %%
# import pandas as pd
import matplotlib.pyplot as plt
from kernels import *
from tools import *

WITH_FIELDS = False

# Loading image
working = load_image("C:/Users/xcrea/Documents/YandexPython/pythonProject/img/SPb1.jpg")
I = rgb2gray(working)
I_blurred_5 = blur_image(I, 5)
I_blurred_11 = blur_image(I, 11)

# Roberts, Prewitt, Sobel
results = {method_name: edges_grad(I, method) for method_name, method in gradient_methods.items() if method_name != scharr_kernels}
# Scharr
results.update({"Щарр": edges_grad(I_blurred_11, scharr_kernels)})

# Laplace
LK1 = laplace_kernel_4()
LK2 = laplace_kernel_8()
results.update({"Лапласс 4% 4 соседа": laplacian_edge_detection(I_blurred_5, LK1, 4),
                "Лапласс 4% 8 соседей": laplacian_edge_detection(I_blurred_5, LK2, 4)})

# Canny
edges_canny = cv2.Canny(I, 30, 115)

# Результаты
fig, axes = plt.subplots(2, 4, figsize=(24, 8))
fig.suptitle("Все вместе, итоговые", fontsize=16, fontweight='bold')

axes[0][0].imshow(I, cmap='gray')
axes[0][0].set_title("Исходное", fontsize=12)
axes[0][0].axis('off')

axes[0][1].imshow(results["Робертс"], cmap='gray')
axes[0][1].set_title("Робертс", fontsize=12)
axes[0][1].axis('off')

axes[0][2].imshow(results["Прюитт"], cmap='gray')
axes[0][2].set_title("Прюитт", fontsize=12)
axes[0][2].axis('off')

axes[0][3].imshow(results["Собель"], cmap='gray')
axes[0][3].set_title("Собель", fontsize=12)
axes[0][3].axis('off')

axes[1][0].imshow(results["Щарр"], cmap='gray')
axes[1][0].set_title("Щарр", fontsize=12)
axes[1][0].axis('off')
1
axes[1][1].imshow(results["Лапласс 4% 4 соседа"], cmap='gray')
axes[1][1].set_title("Лапласс 4% 4 соседа", fontsize=12)
axes[1][1].axis('off')

axes[1][2].imshow(results["Лапласс 4% 8 соседей"], cmap='gray')
axes[1][2].set_title("Лапласс 4% 8 соседей", fontsize=12)
axes[1][2].axis('off')

axes[1][3].imshow(edges_canny, cmap='gray')
axes[1][3].set_title("Кэнни", fontsize=12)
axes[1][3].axis('off')

plt.tight_layout()
plt.show()

# img2 = working + edges_canny[..., np.newaxis]
# fig, axes = plt.subplots(1, 2, figsize=(24, 8))
# axes[0].imshow(img2)
# axes[0].axis('off')
# axes[1].imshow(working)
# axes[1].axis('off')
# plt.tight_layout()
# plt.show()
