import numpy as np
import cv2
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


def blur_image(img: np.array, size: int) -> np.array:
    blurred = cv2.GaussianBlur(img, (size, size), 0)

    return blurred


def grad(I, method):
    """Градиент изображения"""
    kx, ky = method()
    Gx = cv2.filter2D(I, cv2.CV_64F, kx)
    Gy = cv2.filter2D(I, cv2.CV_64F, ky)

    return Gx, Gy


def edges_grad(I, method, threshold=30):
    """Границы по градиенту"""
    Gx, Gy = grad(I, method)
    magnitude = np.sqrt(Gx ** 2 + Gy ** 2)
    # magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    edges = np.zeros_like(magnitude, dtype=np.uint8)
    edges[magnitude > threshold] = 255

    return edges


def laplacian_edge_detection(image, laplacian_kernel, threshold_percent=4):
    """
    Применяет оператор Лапласа для выделения границ с условиями zero-crossing и порогом 4%

    Args:
        image: входное изображение (в оттенках серого)
        threshold_percent: порог в процентах для разности по модулю (по умолчанию 4%)
        laplacian_kernel: ядро Лапласа

    Returns:
        edges: бинарное изображение границ (255 - граница, 0 - фон)
    """

    # оператор Лапласа
    laplacian_result = cv2.filter2D(image.astype(np.float32), -1, laplacian_kernel)

    # маска для границ
    edges = np.zeros_like(image, dtype=np.uint8)

    # пороговое значение, % от максимального значения Лапласиана
    max_val = np.max(np.abs(laplacian_result))
    threshold = (threshold_percent / 100.0) * max_val

    print(f"Максимальное значение Лапласиана: {max_val:.2f}")
    print(f"Порог разности ({threshold_percent}%): {threshold:.2f}")

    # zero-crossing с порогом для разности
    height, width = image.shape

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            switch = 0

            # 4 пары соседей, с разных сторон по направлениям -- | \ /
            neighbors = np.array([
                [i - 1, j, i + 1, j],
                [i, j - 1, i, j + 1],
                [i - 1, j - 1, i + 1, j + 1],
                [i - 1, j + 1, i + 1, j - 1],
            ])
            # перебор пар
            for ni, nj, ki, kj in neighbors:
                has_sign_change = False
                meets_threshold = False
                neighbor_val_1 = laplacian_result[ni, nj]
                neighbor_val_2 = laplacian_result[ki, kj]

                # Условие 1: разные знаки
                if neighbor_val_1 * neighbor_val_2 < 0:
                    has_sign_change = True

                    # Условие 2: абс разницы >= порога
                    if abs(neighbor_val_1 - neighbor_val_2) >= threshold:
                        meets_threshold = True

                        if has_sign_change and meets_threshold:
                            switch += 1

                        continue

            # условие 3: не менее 2-х направлений (пар)
            if switch >= 2:
                edges[i, j] = 255

    return edges
