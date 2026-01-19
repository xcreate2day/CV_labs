import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image


def load_image(img_path: str) -> np.ndarray:
    if not (Path(img_path).exists() and Path(img_path).is_file()):
        raise FileNotFoundError(f"Image at {img_path} not found")

    img = np.array(Image.open(img_path), dtype=np.uint8)

    return img


def rgb2gray(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return gray


def __fitFigure(shape, maxwidth=800, maxheight=800, title=None):
    """Настройка размера фигуры matplotlib"""
    # В matplotlib размеры задаются в дюймах
    dpi = 100
    width_inches = min(shape[1], maxwidth) / dpi
    height_inches = min(shape[0], maxheight) / dpi

    # Создаем фигуру
    fig, ax = plt.subplots(figsize=(width_inches, height_inches))

    # Добавляем заголовок если указан
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    # Настройка осей для сохранения пропорций
    if shape[0] > shape[1]:
        ax.set_aspect('equal')
    else:
        ax.set_aspect('equal')

    # Убираем отступы
    plt.subplots_adjust(left=0, right=1, top=0.95 if title else 1, bottom=0)

    return fig, ax


def __buildMatplotlibImage(image, ax):
    """Создание изображения для matplotlib"""
    if len(image.shape) > 2:
        # Цветное изображение (RGB/BGR)
        # OpenCV использует BGR, matplotlib - RGB
        if image.shape[2] == 3:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image
        ax.imshow(img_rgb)
    else:
        # Серое изображение
        ax.imshow(image, cmap='gray', vmin=0, vmax=255)

    # Инвертируем ось Y для соответствия координатам изображения
    ax.invert_yaxis()
    ax.axis('off')


def show_image(image, maxwidth=800, maxheight=800, title="Изображение"):
    """Отображение цветного или серого изображения"""
    fig, ax = __fitFigure(image.shape, maxwidth, maxheight, title)
    __buildMatplotlibImage(image, ax)
    plt.show()


# def showGray(image, maxwidth=800, maxheight=800):
#     """Преобразование в серое изображение"""
#     if len(image.shape) > 2:
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     return image


# def showGrayInput(image, maxwidth=800, maxheight=800, title=None):
#     """Отображение серого изображения"""
#     # gray_image = showGray(image)
#     gray_image = rgb2gray(image)
#     fig, ax = __fitFigure(gray_image.shape, maxwidth, maxheight, title)
#     ax.imshow(gray_image, cmap='gray', vmin=0, vmax=255)
#     ax.axis('off')
#     plt.show()


# def showVecField(x, y, u, v, maxwidth=800, maxheight=800, title="Векторное поле"):
#     """Отображение векторного поля"""
#     fig, ax = plt.subplots(figsize=(maxwidth / 100, maxheight / 100))
#
#     # Добавляем общий заголовок
#     if title:
#         fig.suptitle(title, fontsize=14, fontweight='bold')
#
#     # Масштабируем векторы для лучшей визуализации
#     magnitude = np.sqrt(u ** 2 + v ** 2)
#     if magnitude.max() > 0:
#         scale = 0.9 * min(x.max() - x.min(), y.max() - y.min()) / magnitude.max()
#         u_scaled = u * scale
#         v_scaled = v * scale
#     else:
#         u_scaled = u
#         v_scaled = v
#
#     ax.quiver(x, y, u_scaled, v_scaled, angles='xy', scale_units='xy', scale=1)
#     ax.set_xlim(x.min(), x.max())
#     ax.set_ylim(y.max(), y.min())  # Инвертируем ось Y для соответствия координатам изображения
#     ax.set_aspect('equal')
#     ax.set_xlabel("X координата")
#     ax.set_ylabel("Y координата")
#     ax.set_title("Векторное поле градиента", fontsize=12)
#     plt.tight_layout()
#     plt.show()


def quiverImage(shape, x, y, u, v, color1=(0, 0, 255, 255), color2=(0, 255, 0, 255), qMaxLen=0.05):
    """Создание изображения с векторами """
    assert x.shape == y.shape and x.shape == u.shape and x.shape == v.shape
    assert np.min(x) >= 0 and np.min(y) >= 0 and np.max(x) < shape[1] and np.max(y) < shape[0]

    c1 = cv2.cvtColor(np.uint8([[[*color1]]]), cv2.COLOR_RGB2HSV)
    c2 = cv2.cvtColor(np.uint8([[[*color2]]]), cv2.COLOR_RGB2HSV)
    c_delta = c2 - c1

    # Normalize quiver lengths to qMaxLen of min(shape)
    l = min(shape) * qMaxLen
    s = np.sqrt(u * u + v * v)
    ms = np.max(s)
    scale = l / ms

    u = scale * u
    v = scale * v
    s = scale * s

    x = np.floor(x, dtype=np.int32, casting='unsafe')
    y = np.floor(y, dtype=np.int32, casting='unsafe')
    xu = np.floor(x + u, dtype=np.int32, casting='unsafe')
    yv = np.floor(y + v, dtype=np.int32, casting='unsafe')

    xu[xu < 0] = 0
    yv[yv < 0] = 0
    xu[xu >= shape[1]] = shape[1] - 1
    yv[yv >= shape[0]] = shape[0] - 1

    Q = np.zeros((*shape, 4), dtype=np.uint8)
    for i in range(y.shape[0]):
        for j in range(x.shape[1]):
            c = cv2.cvtColor(np.uint8(c1 + s[i, j] / l * c_delta), cv2.COLOR_HSV2RGB)
            Q = cv2.arrowedLine(Q,
                                (x[i, j], y[i, j]),
                                (xu[i, j], yv[i, j]),
                                tuple(map(int, c[0, 0])))

    return Q


# def showGrayGrad(G1, G2, showField=False, step=(1, 1), original=None, title="Анализ градиента изображения"):
#     """
#     Отображение градиента серого изображения:
#         - Первая компонента градиента
#         - Вторая компонента градиента
#         - Величина градиента
#         - (опционально) Векторное поле градиента
#     """
#     assert G1.shape == G2.shape
#
#     # Количество субплотов зависит от параметров
#     if showField:
#         fig, axes = plt.subplots(2, 2, figsize=(15, 10))
#         axes = axes.flatten()
#     else:
#         fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#
#     # Общий заголовок для всей фигуры
#     fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
#
#     # 1. Первая компонента градиента
#     im1 = axes[0].imshow(G1, cmap='gray')
#     axes[0].set_title("Горизонтальная компонента градиента (Gx)", fontsize=12, fontweight='semibold')
#     axes[0].axis('off')
#
#     # 2. Вторая компонента градиента
#     im2 = axes[1].imshow(G2, cmap='gray')
#     axes[1].set_title("Вертикальная компонента градиента (Gy)", fontsize=12, fontweight='semibold')
#     axes[1].axis('off')
#
#     # 3. Величина градиента
#     Magnitude = np.sqrt(G1 * G1 + G2 * G2)
#     im3 = axes[2].imshow(Magnitude, cmap='gray')
#     axes[2].set_title("Величина градиента (√(Gx² + Gy²))", fontsize=12, fontweight='semibold')
#     axes[2].axis('off')
#
#     # 4. Векторное поле (если нужно)
#     if showField:
#         x, y = np.meshgrid(range(0, G1.shape[1], step[1]), range(0, G1.shape[0], step[0]))
#         Q = quiverImage(G1.shape, x, y, G1[::step[0], ::step[1]], G2[::step[0], ::step[1]])
#
#         if original is not None:
#             if len(original.shape) < 3:
#                 original_rgb = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
#             else:
#                 original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
#
#             # Наложение векторного поля на исходное изображение
#             combined = cv2.addWeighted(original_rgb, 0.7, Q[:, :, :3], 0.3, 0)
#             axes[3].imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
#             axes[3].set_title("Векторное поле на исходном изображении", fontsize=12, fontweight='semibold')
#         else:
#             axes[3].imshow(cv2.cvtColor(Q, cv2.COLOR_BGR2RGB))
#             axes[3].set_title("Векторное поле градиента", fontsize=12, fontweight='semibold')
#
#         axes[3].axis('off')
#
#     plt.tight_layout()
#     plt.show()
#
#     return G1, G2, Magnitude


def plotVectorContour(img, start, vc, color=(255, 0, 0), thickness=3):
    """
    Рисование векторного контура на изображении
    """
    y, x = np.int32(start)
    for v in vc:
        y1, x1 = np.int32(y + v.imag), np.int32(x + v.real)
        cv2.line(img, (x, y), (x1, y1), color, thickness)
        y, x = y1, x1


def plotVectorContours(VCs, img=None, shape=None, color=(255, 0, 0), thickness=3):
    """
    Рисование нескольких векторных контуров на изображении
    """
    if img is None:
        if shape is None:
            raise ValueError("Либо 'img', либо 'shape' должен быть задан")
        img = np.zeros((*shape, 3), dtype=np.uint8)
    elif shape is not None:
        raise ValueError("Либо 'img', либо 'shape' должен быть задан, не оба")
    elif len(img.shape) < 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = np.copy(img)

    for start, v in VCs:
        plotVectorContour(img, start, v, color, thickness)

    return img


def grad(I, method, title="Вычисление градиента"):
    """Вычисление градиента изображения с использованием заданного метода"""
    kx, ky = method()
    Gx = cv2.filter2D(I, cv2.CV_64F, kx)
    Gy = cv2.filter2D(I, cv2.CV_64F, ky)

    # # Отображение результатов
    # fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    # fig.suptitle(title, fontsize=16, fontweight='bold')
    #
    # axes[0].imshow(Gx, cmap='gray')
    # axes[0].set_title("Горизонтальный градиент (Gx)", fontsize=12)
    # axes[0].axis('off')
    #
    # axes[1].imshow(Gy, cmap='gray')
    # axes[1].set_title("Вертикальный градиент (Gy)", fontsize=12)
    # axes[1].axis('off')
    #
    # Magnitude = np.sqrt(Gx ** 2 + Gy ** 2)
    # axes[2].imshow(Magnitude, cmap='gray')
    # axes[2].set_title("Величина градиента", fontsize=12)
    # axes[2].axis('off')
    #
    # plt.tight_layout()
    # plt.show()

    return Gx, Gy


def edgesGrad(I, method, threshold=30, title="Выделение границ по градиенту"):
    """Выделение границ по градиенту"""
    Gx, Gy = grad(I, method, title="Вычисление градиента для выделения границ")
    magnitude = np.sqrt(Gx ** 2 + Gy ** 2)
    edges = np.zeros_like(magnitude, dtype=np.uint8)
    edges[magnitude > threshold] = 255

    # # Отображение результатов
    # fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    # fig.suptitle(title, fontsize=16, fontweight='bold')
    #
    # axes[0].imshow(I, cmap='gray')
    # axes[0].set_title("Исходное изображение", fontsize=12)
    # axes[0].axis('off')
    #
    # axes[1].imshow(magnitude, cmap='gray')
    # axes[1].set_title(f"Величина градиента\n(порог = {threshold})", fontsize=12)
    # axes[1].axis('off')
    #
    # axes[2].imshow(edges, cmap='gray')
    # axes[2].set_title("Выделенные границы", fontsize=12)
    # axes[2].axis('off')
    #
    # plt.tight_layout()
    # plt.show()

    return edges


# def show_images(images, titles, figsize=(20, 4), cmap='gray', vmin=0, vmax=255, main_title="Сравнение изображений"):
#     """
#     Отображает несколько изображений в ряд
#
#     Parameters:
#     - images: список изображений
#     - titles: список заголовков
#     - figsize: размер фигуры
#     - cmap: цветовая карта
#     - vmin, vmax: диапазон значений
#     - main_title: общий заголовок
#     """
#     n = len(images)
#     plt.figure(figsize=figsize)
#
#     # Общий заголовок
#     plt.suptitle(main_title, fontsize=16, fontweight='bold', y=0.98)
#
#     for i in range(n):
#         plt.subplot(1, n, i + 1)
#         plt.imshow(images[i], cmap=cmap, vmin=vmin, vmax=vmax)
#         plt.title(titles[i], fontsize=12, fontweight='semibold')
#         plt.axis('off')
#
#     plt.tight_layout()
#     plt.show()


def laplacian_edge_detection(image, threshold_percent=4, laplacian_kernel=None,
                             title="Выделение границ оператором Лапласа"):
    """
    Применяет оператор Лапласа для выделения границ с условиями zero-crossing и порогом 4%

    Args:
        image: входное изображение (в оттенках серого)
        threshold_percent: порог в процентах для разности по модулю (по умолчанию 4%)
        laplacian_kernel: ядро Лапласа
        title: заголовок для отображения

    Returns:
        edges: бинарное изображение границ (255 - граница, 0 - фон)
    """

    # 1. Определяем ядро Лапласа
    if laplacian_kernel is None:
        laplacian_kernel = np.array([[0, -1, 0],
                                     [-1, 4, -1],
                                     [0, -1, 0]])

    # 2. Применяем оператор Лапласа
    laplacian_result = cv2.filter2D(image.astype(np.float32), -1, laplacian_kernel)

    # 3. Создаем маску для границ
    edges = np.zeros_like(image, dtype=np.uint8)

    # 4. Вычисляем пороговое значение (4% от максимального значения Лапласиана)
    max_val = np.max(np.abs(laplacian_result))
    threshold = (threshold_percent / 100.0) * max_val

    print(f"Максимальное значение Лапласиана: {max_val:.2f}")
    print(f"Порог разности ({threshold_percent}%): {threshold:.2f}")

    # 5. Ищем zero-crossing с порогом для разности
    height, width = image.shape

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            center_val = laplacian_result[i, j]
            switch = 0

            # Проверяем 4-связных соседей
            neighbors = [
                (i - 1, j, i + 1, j),
                (i, j - 1, i, j + 1),
                (i - 1, j - 1, i + 1, j + 1),
                (i - 1, j + 1, i + 1, j - 1),
            ]

            for ni, nj, ki, kj in neighbors:
                has_sign_change = False
                meets_threshold = False
                neighbor_val_1 = laplacian_result[ni, nj]
                neighbor_val_2 = laplacian_result[ki, kj]

                # Условие 1: произведение < 0 (разные знаки)
                if neighbor_val_1 * neighbor_val_2 < 0:
                    has_sign_change = True

                    # Условие 2: |center_val - neighbor_val| >= 4% порога
                    difference = abs(neighbor_val_1 - neighbor_val_2)
                    if difference >= threshold:
                        meets_threshold = True
                        if (has_sign_change is True) and (meets_threshold is True):
                            switch += 1
                        continue
            if switch >= 2:
                edges[i, j] = 255

    # # Отображение результатов
    # fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    # fig.suptitle(title, fontsize=16, fontweight='bold')
    #
    # axes[0].imshow(image, cmap='gray')
    # axes[0].set_title("Исходное изображение", fontsize=12)
    # axes[0].axis('off')
    #
    # axes[1].imshow(laplacian_result, cmap='gray')
    # axes[1].set_title("Результат оператора Лапласа", fontsize=12)
    # axes[1].axis('off')
    #
    # axes[2].imshow(np.abs(laplacian_result), cmap='gray')
    # axes[2].set_title("Абсолютное значение Лапласиана", fontsize=12)
    # axes[2].axis('off')
    #
    # axes[3].imshow(edges, cmap='gray')
    # axes[3].set_title(f"Границы (порог {threshold_percent}%)", fontsize=12)
    # axes[3].axis('off')
    #
    # plt.tight_layout()
    # plt.show()

    return edges


# Дополнительные функции с заголовками

# def showHistogram(image, title="Гистограмма яркости изображения"):
#     """Отображение гистограммы изображения"""
#     if len(image.shape) > 2:
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     fig, axes = plt.subplots(1, 2, figsize=(12, 4))
#     fig.suptitle(title, fontsize=16, fontweight='bold')
#
#     # Гистограмма
#     axes[0].hist(image.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
#     axes[0].set_title("Распределение яркости", fontsize=12)
#     axes[0].set_xlabel("Яркость (0-255)", fontsize=10)
#     axes[0].set_ylabel("Частота", fontsize=10)
#     axes[0].grid(True, alpha=0.3)
#
#     # Изображение
#     axes[1].imshow(image, cmap='gray')
#     axes[1].set_title("Исходное изображение", fontsize=12)
#     axes[1].axis('off')
#
#     plt.tight_layout()
#     plt.show()


# def compareMethods(images, method_names, main_title="Сравнение методов обработки изображений"):
#     """Сравнение результатов разных методов"""
#     n = len(images)
#
#     fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
#     fig.suptitle(main_title, fontsize=16, fontweight='bold', y=0.98)
#
#     for i in range(n):
#         axes[i].imshow(images[i], cmap='gray')
#         axes[i].set_title(method_names[i], fontsize=12, fontweight='semibold')
#         axes[i].axis('off')
#
#     plt.tight_layout()
#     plt.show()


# def showWithContours(original, contours_image, title="Изображение с контурами"):
#     """Отображение исходного изображения и изображения с контурами"""
#     fig, axes = plt.subplots(1, 2, figsize=(10, 4))
#     fig.suptitle(title, fontsize=16, fontweight='bold')
#
#     # Исходное изображение
#     if len(original.shape) > 2:
#         original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
#         axes[0].imshow(original_rgb)
#     else:
#         axes[0].imshow(original, cmap='gray')
#     axes[0].set_title("Исходное изображение", fontsize=12)
#     axes[0].axis('off')
#
#     # Изображение с контурами
#     if len(contours_image.shape) > 2:
#         contours_rgb = cv2.cvtColor(contours_image, cv2.COLOR_BGR2RGB)
#         axes[1].imshow(contours_rgb)
#     else:
#         axes[1].imshow(contours_image, cmap='gray')
#     axes[1].set_title("С выделенными контурами", fontsize=12)
#     axes[1].axis('off')
#
#     plt.tight_layout()
#     plt.show()