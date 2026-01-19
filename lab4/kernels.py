import cv2
import numpy as np


def robertsKernels():
    """Возвращает ядра оператора Робертса"""
    kx = np.array(
        [
            [1, 0],
            [0, -1],
        ],
        dtype=np.float32,
    )
    ky = np.array(
        [
            [0, 1],
            [-1, 0],
        ],
        dtype=np.float32,
    )
    return kx, ky


def prewittKernels():
    """Возвращает ядра оператора Прюитт"""
    kx = np.array(
        [
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1],
        ],
        dtype=np.float32,
    )
    ky = np.array(
        [
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1],
        ],
        dtype=np.float32,
    )
    return kx, ky


def sobelKernels():
    """Возвращает ядра оператора Собеля"""
    kx = np.array(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ],
        dtype=np.float32,
    )
    ky = np.array(
        [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1],
        ],
        dtype=np.float32,
    )
    return kx, ky


def scharrKernels():
    """Возвращает ядра оператора Щарра"""
    kx = np.array(
        [
            [-3, 0, 3],
            [-10, 0, 10],
            [-3, 0, 3],
        ],
        dtype=np.float32,
    )
    ky = np.array(
        [
            [-3, -10, -3],
            [0, 0, 0],
            [3, 10, 3],
        ],
        dtype=np.float32,
    )
    return kx, ky


def laplaceKernel():
    """Возвращает ядро Лаплассиана"""
    return np.array(
        [
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1],
        ],
        dtype=np.float32,
    )


def gaussian_kernel(size, sigma):
    """Ядро Гаусса, номализация"""
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma**2))
        * np.exp(
            -((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma**2)
        ),
        (size, size),
        dtype=np.float32,
    )
    return kernel / np.sum(kernel)


def custom_convolution(image, kernel):
    """Кастомная свёртка с произвольным ядром"""
    ih, iw = image.shape
    kh, kw = kernel.shape

    pad = kh // 2
    padded = np.pad(image, pad, mode="constant")

    output = np.zeros_like(image, dtype=np.float32)

    for y in range(ih):
        for x in range(iw):
            output[y, x] = np.sum(padded[y : y + kh, x : x + kw] * kernel)

    return output


def custom_gaussian_blur(image, size, sigma):
    """Применяет размытие Гаусса с использованием самописной свертки"""
    kernel = gaussian_kernel(size, sigma)
    return custom_convolution(image, kernel)


def grad_custom(image, kernel_func):
    """Вычисляет градиент изображения с помощью заданных ядер и самописной свертки"""
    kx, ky = kernel_func()
    Gx = custom_convolution(image, kx)
    Gy = custom_convolution(image, ky)
    return Gx, Gy


def edgesGrad_custom(image, kernel_func, threshold=0.1):
    """Выделяет края на основе градиента с использованием самописной свертки"""
    Gx, Gy = grad_custom(image, kernel_func)
    magnitude = np.sqrt(Gx * Gx + Gy * Gy)

    # Нормализация и применение порога
    max_val = np.max(magnitude)
    if max_val > 0:
        magnitude = magnitude / max_val
    else:
        magnitude = np.zeros_like(magnitude)
    edges = (magnitude > threshold).astype(np.uint8) * 255

    return edges


def zeros_crossing(img, thresh):
    d_size = (img.shape[1], img.shape[0])

    mask = np.array(
        [
            [1, 0, -1],
            [0, 1, 0],
        ],
        dtype=np.float32,
    )
    shift_left = cv2.warpAffine(img, mask, d_size)
    mask = np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=np.float32,
    )
    shift_right = cv2.warpAffine(img, mask, d_size)

    mask = np.array(
        [
            [1, 0, 0],
            [0, 1, -1],
        ],
        dtype=np.float32,
    )
    shift_up = cv2.warpAffine(img, mask, d_size)
    mask = np.array(
        [
            [1, 0, 0],
            [0, 1, 1],
        ],
        dtype=np.float32,
    )
    shift_down = cv2.warpAffine(img, mask, d_size)

    mask = np.array(
        [
            [1, 0, 1],
            [0, 1, 1],
        ],
        dtype=np.float32,
    )
    shift_right_down = cv2.warpAffine(img, mask, d_size)
    mask = np.array(
        [
            [1, 0, -1],
            [0, 1, -1],
        ],
        dtype=np.float32,
    )
    shift_left_up = cv2.warpAffine(img, mask, d_size)

    mask = np.array(
        [
            [1, 0, 1],
            [0, 1, -1],
        ],
        dtype=np.float32,
    )
    shift_right_up = cv2.warpAffine(img, mask, d_size)
    mask = np.array(
        [
            [1, 0, -1],
            [0, 1, 1],
        ],
        dtype=np.float32,
    )
    shift_left_down = cv2.warpAffine(img, mask, d_size)

    shift_left_right_sign = shift_left * shift_right
    shift_up_down_sign = shift_up * shift_down
    shift_rd_lu_sign = shift_right_down * shift_left_up
    shift_ru_ld_sign = shift_right_up * shift_left_down

    shift_left_right_norm = np.abs(shift_left - shift_right)
    shift_up_down_norm = np.abs(shift_up - shift_down)
    shift_rd_lu_norm = np.abs(shift_right_down - shift_left_up)
    shift_ru_ld_norm = np.abs(shift_right_up - shift_left_down)

    zero_crossing = (
        ((shift_left_right_sign < 0) & (shift_left_right_norm > thresh)).astype(
            np.uint8
        )
        + ((shift_up_down_sign < 0) & (shift_up_down_norm > thresh)).astype(np.uint8)
        + ((shift_rd_lu_sign < 0) & (shift_rd_lu_norm > thresh)).astype(np.uint8)
        + ((shift_ru_ld_sign < 0) & (shift_ru_ld_norm > thresh)).astype(np.uint8)
    )

    result = np.zeros(shape=img.shape, dtype=np.uint8)
    result[zero_crossing >= 2] = 255

    return result


def laplacian(img, kernel_size, sigma=1.0, threshold=None, alpha=0.01):
    blur_img = cv2.GaussianBlur(
        img.astype("float32"), (kernel_size, kernel_size), sigmaX=sigma
    )
    laplacian_img = cv2.Laplacian(blur_img, cv2.CV_32F)
    if threshold is None:
        threshold = np.max(np.abs(laplacian_img)) * alpha
    edge_image = zeros_crossing(laplacian_img, threshold)
    return edge_image
