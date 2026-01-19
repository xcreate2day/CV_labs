import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image


def load_image(img_path: str) -> np.array:
    if not (Path(img_path).exists() and Path(img_path).is_file()):
        raise FileNotFoundError(f"Image at {img_path} not found")

    img = np.array(Image.open(img_path), dtype=np.uint8)

    return img


def rgb2gray(img: np.array) -> np.array:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return gray


def binarize_and_invert(img: np.array) -> np.array:
    # фон (большое размытие)
    # background = cv2.medianBlur(img, 51)  # сильное размытие
    # background = cv2.GaussianBlur(img, (0, 0), sigmaX=50, sigmaY=50)
    background = cv2.GaussianBlur(img, (201, 201), 0)  # сильное размытие
    # оригинал минус фон
    foreground = cv2.subtract(background, img)  # текст -> светлее

    # нормализуем контраст
    # foreground = cv2.normalize(foreground, None, 0, 255, cv2.NORM_MINMAX)
    foreground = cv2.normalize(foreground, None, 0, 255, cv2.NORM_MINMAX)

    image = cv2.adaptiveThreshold(
        foreground,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # Текст белый, фон чёрный
        13,  # Оптимальный размер
        3  # Оптимальная константа
    )

    return image


# def create_test_marker(binarized_and_inverted):
#     """маркеры в ключевых местах"""
#     marker = np.zeros_like(binarized_and_inverted)
#     height, width = binarized_and_inverted.shape
#
#     # -> маркеры в центре и по углам
#     positions = [
#         (height // 2, width // 2),  # центр
#         (height // 4, width // 4),  # левый верх
#         (height // 4, 3 * width // 4),  # правый верх
#         (3 * height // 4, width // 4),  # левый низ
#         (3 * height // 4, 3 * width // 4),  # правый низ
#     ]
#
#     # ближайшие белые пиксели к этим позициям
#     for y, x in positions:
#         # поиск в окрестности 50x50
#         y_start = max(0, y - 25)
#         y_end = min(height, y + 25)
#         x_start = max(0, x - 25)
#         x_end = min(width, x + 25)
#
#         region = binarized_and_inverted[y_start:y_end, x_start:x_end]
#         white_pixels = np.argwhere(region == 255)
#
#         if len(white_pixels) > 0:
#             # первый белый пиксель
#             dy, dx = white_pixels[0]
#             marker[y_start + dy, x_start + dx] = 255
#
#     return marker


def conditional_dilate_reconstruction(mask, struct_elem1=None, struct_elem2=None, max_iter=1000):
    """Условная дилатация (реконструкция)"""
    if struct_elem1 is None:
        struct_elem1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    if struct_elem2 is None:
        struct_elem2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    # marker = create_test_marker(mask)
    # if np.sum(marker) == 0:
    #     print("Маркер пустой, возвращаю исходное изображение")
    #     return mask

    current = cv2.erode(mask, struct_elem1)
    # если всё удалила эрозия
    if np.sum(current) == 0:
        current = np.zeros_like(mask)
        h, w = mask.shape
        # несколько точек в центре
        current[h // 2, w // 2] = 255
        current[h // 3, w // 3] = 255
        current[2 * h // 3, 2 * w // 3] = 255

    it = 0
    kernels = [struct_elem1, struct_elem2]

    while True:
        # ядро для текущей итерации
        kernel = kernels[it % 2]

        # Дилатация + ограничение маской
        dilated = cv2.dilate(current, kernel)
        new = cv2.bitwise_and(dilated, mask)
        it += 1

        # новое состояние vs текущее
        if np.array_equal(new, current):
            # Если не изменилось с этим ядром - пробуем другое
            other_kernel = struct_elem2 if kernel is struct_elem1 else struct_elem1
            dilated_other = cv2.dilate(current, other_kernel)
            new_other = cv2.bitwise_and(dilated_other, mask)

            # с другим ядром тоже не изменилось -> ВЫХОД
            if np.array_equal(new_other, current):
                print(f"Реконструкция завершена за {it} итераций")
                break
            else:
                # С другим ядром есть рост - продолжаем с ним
                current = new_other
                continue
        else:
            # Есть рост с текущим ядром -> продолжаем
            current = new

        if it >= max_iter:
            print(f"Достигнут лимит {max_iter} итераций")
            break

    return current


def morphological_skeleton(img, kernel_size=3):
    """Корректное вычисление морфологического скелета"""
    # Гарантируем бинарное изображение
    if img.max() > 1:
        _, img_binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    else:
        img_binary = (img * 255).astype(np.uint8)

    # Структурный элемент
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))

    skeleton = np.zeros(img_binary.shape, dtype=np.uint8)
    eroded = np.zeros(img_binary.shape, dtype=np.uint8)
    temp = np.zeros(img_binary.shape, dtype=np.uint8)

    # Классический алгоритм
    while True:
        # Эрозия
        cv2.erode(img_binary, element, eroded)
        # Дилатация эродированного
        cv2.dilate(eroded, element, temp)
        # Вычитание: img_binary - temp
        cv2.subtract(img_binary, temp, temp)
        # Объединение скелета
        cv2.bitwise_or(skeleton, temp, skeleton)
        # Копируем эродированное для следующей итерации
        img_binary, eroded = eroded, img_binary  # swap

        # Проверяем, не исчезло ли всё
        if cv2.countNonZero(img_binary) == 0:
            break

    return skeleton

# def morphological_skeleton(img, struct_elem=None, max_iter=None, visualize=False):
#     """Вычисление морфологического скелета бинарного изображения"""
#     # к 0/255 uint8
#     bin_img = (img > 0).astype(np.uint8) * 255
#
#     if struct_elem is None:
#         # крест (4-соседство) часто даёт более «скелетообразные» линии
#         struct_elem = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
#
#     # Для кумуляции результата
#     skeleton = np.zeros_like(bin_img, dtype=np.uint8)
#     layers = []  # если нужно визуализировать
#
#     # Текущий эродированный образ
#     eroded = bin_img.copy()
#     iteration = 0
#     if max_iter is None:
#         max_iter = min(bin_img.shape) // 2 + 1
#
#     while True:
#         # 1) эрозия на один шаг
#         eroded_next = cv2.erode(eroded, struct_elem)
#         if np.all(eroded_next == 0):
#             # если следующая эрозия полностью исчезает, от неё нет слоя,
#             # но остаётся возможно последний слой: E_k - O_k где E_k = eroded
#             opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, struct_elem)
#             sk_layer = cv2.subtract(eroded, opened)
#             skeleton = cv2.bitwise_or(skeleton, sk_layer)
#             if visualize:
#                 layers.append(sk_layer.copy())
#             break
#
#         # 2) открытие текущего eroded (то есть E_k -> O_k)
#         opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, struct_elem)
#
#         # 3) слой скелета S_k = E_k - O_k
#         sk_layer = cv2.subtract(eroded, opened)
#         skeleton = cv2.bitwise_or(skeleton, sk_layer)
#         if visualize:
#             layers.append(sk_layer.copy())
#
#         # подготовка к следующей итерации
#         eroded = eroded_next
#         iteration += 1
#         if iteration >= max_iter:
#             print("Достигнут max_iter, прерываю")
#             break
#
#     # skeleton уже 0/255
#     if visualize:
#         return skeleton, layers
#     return skeleton


def show_two_images_together(img1, img2, title1: str = None, title2: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(wspace=2)

    axes[0].imshow(img1, cmap='gray')
    axes[0].set_title(title1, fontsize=12)
    axes[0].set_frame_on(True)
    axes[0].patch.set_edgecolor('green')
    axes[0].patch.set_linewidth(3)

    axes[1].imshow(img2, cmap='gray')
    axes[1].set_title(title2, fontsize=12)
    axes[1].set_frame_on(True)
    axes[1].patch.set_edgecolor('green')
    axes[1].patch.set_linewidth(3)

    plt.tight_layout()
    plt.show()
