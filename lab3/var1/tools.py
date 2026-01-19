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
    # background = cv2.GaussianBlur(img, (0, 0), sigmaX=50, sigmaY=50)
    background = cv2.GaussianBlur(img, (201, 201), 0)  # сильное размытие

    # оригинал минус фон
    foreground = cv2.subtract(background, img)  # текст -> светлее

    # нормализуем контраст
    foreground = cv2.normalize(foreground, None, 0, 255, cv2.NORM_MINMAX)

    _, img = cv2.threshold(foreground, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return img


def create_test_marker(binarized_and_inverted):
    """маркеры в ключевых местах"""
    marker = np.zeros_like(binarized_and_inverted)
    height, width = binarized_and_inverted.shape

    # -> маркеры в центре и по углам
    positions = [
        (height // 2, width // 2),  # центр
        (height // 4, width // 4),  # левый верх
        (height // 4, 3 * width // 4),  # правый верх
        (3 * height // 4, width // 4),  # левый низ
        (3 * height // 4, 3 * width // 4),  # правый низ
    ]

    # ближайшие белые пиксели к этим позициям
    for y, x in positions:
        # поиск в окрестности 50x50
        y_start = max(0, y - 25)
        y_end = min(height, y + 25)
        x_start = max(0, x - 25)
        x_end = min(width, x + 25)

        region = binarized_and_inverted[y_start:y_end, x_start:x_end]
        white_pixels = np.argwhere(region == 255)

        if len(white_pixels) > 0:
            # первый белый пиксель
            dy, dx = white_pixels[0]
            marker[y_start + dy, x_start + dx] = 255

    return marker


def conditional_dilate_reconstruction(mask, struct_elem=None, max_iter=1000):
    """Условная дилатация (реконструкция)"""
    if struct_elem is None:
        struct_elem = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    marker = create_test_marker(mask)
    if np.sum(marker) == 0:
        print("Маркер пустой, возвращаю исходное изображение")
        return mask

    prev = marker.copy()
    it = 0

    while True:
        dil = cv2.dilate(prev, struct_elem)
        curr = cv2.bitwise_and(dil, mask)
        it += 1
        if np.array_equal(curr, prev) or it >= max_iter:
            break
        prev = curr

    print(f"Реконструкция завершена за {it} итераций")
    return curr


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
