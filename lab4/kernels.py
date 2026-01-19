import numpy as np


def roberts_kernels():
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


def prewitt_kernels():
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


def sobel_kernels():
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


def scharr_kernels():
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


def laplace_kernel_8():
    """Возвращает ядро Лаплассиана (вариант с даигоналями)"""
    return np.array(
        [
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1],
        ],
        dtype=np.float32,
    )


def laplace_kernel_4():
    """Возвращает ядро Лаплассиана"""
    return np.array(
        [
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0],
        ],
        dtype=np.float32,
    )


gradient_methods = {"Робертс": roberts_kernels, "Прюитт": prewitt_kernels, "Собель": sobel_kernels, "Щарр": scharr_kernels}
