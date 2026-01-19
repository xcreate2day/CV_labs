# %%
from tools import *


image = load_image("C:/Users/xcrea/Documents/YandexPython/pythonProject/img/Karelia2_text.jpg")
gray_image = rgb2gray(image)
binarized_and_inverted = binarize_and_invert(gray_image)
print(f"Размер: {binarized_and_inverted.shape}")
print(f"Уникальные значения: {np.unique(binarized_and_inverted)}")
print(f"Минимум: {binarized_and_inverted.min()}, Максимум: {binarized_and_inverted.max()}")

show_two_images_together(gray_image, binarized_and_inverted, "Исходное, gray", "binarized_and_inverted")

struct_element_B = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))


# Дилатация
dilated_image = cv2.dilate(binarized_and_inverted, struct_element_B)

# Эрозия
eroded_image = cv2.erode(binarized_and_inverted, struct_element_B)

show_two_images_together(dilated_image, eroded_image, "Дилатация", "Эрозия")


# Замыкание
closed_image = cv2.morphologyEx(binarized_and_inverted, cv2.MORPH_CLOSE, struct_element_B)
# inverted = cv2.bitwise_not(binarized_and_inverted)
# closed_image = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, struct_element_B)
# closed = cv2.bitwise_not(closed_image)

# Размыкание
opened_image = cv2.morphologyEx(binarized_and_inverted, cv2.MORPH_OPEN, struct_element_B)

show_two_images_together(closed_image, opened_image, "Замыкание", "Размыкание")


# Условная дилатация
# struct_element_C = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
struct_element_D = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
struct_element_E = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
# struct_element_F = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

cond_dilate_res1 = conditional_dilate_reconstruction(binarized_and_inverted, struct_element_D)
cond_dilate_res2 = conditional_dilate_reconstruction(binarized_and_inverted, struct_element_E)

show_two_images_together(cond_dilate_res1, cond_dilate_res2, "Условная дилатация v1", "Условная дилатация v2")


# Морфологический скелет
text_contours = cv2.Canny(binarized_and_inverted, 50, 150)
morphological_skeleton = morphological_skeleton(text_contours)

# show_two_images_together(text_contours, morphological_skeleton, "text_contours", "morphological_skeleton")

plt.imshow(morphological_skeleton, cmap='gray', vmin=0, vmax=255)
plt.title("Морфологический скелет")
plt.tight_layout()
plt.show()
