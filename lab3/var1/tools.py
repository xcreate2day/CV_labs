import numpy as np
import cv2
import matplotlib.pyplot as plt

#
# def __fitFigure(fig, shape, maxwidth=800, maxheight=800):
#     """Вспомогательная функция для настройки размера фигуры matplotlib"""
#     # В matplotlib размеры задаются в дюймах, поэтому переводим пиксели в дюймы (1 дюйм = 100 DPI)
#     dpi = 100
#     width_inches = min(shape[1], maxwidth) / dpi
#     height_inches = min(shape[0], maxheight) / dpi
#
#     # Устанавливаем размер фигуры
#     fig.set_size_inches(width_inches, height_inches)
#
#     # Настройка осей для сохранения пропорций
#     ax = fig.gca()
#     ax.set_aspect('equal')
#
#     # Убираем отступы
#     plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
#
#
# def showImage(image, maxwidth=800, maxheight=800):
#     """Отображение цветного или серого изображения"""
#     fig, ax = plt.subplots()
#
#     if len(image.shape) > 2:
#         # Цветное изображение
#         ax.imshow(image)
#     else:
#         # Серое изображение
#         ax.imshow(image, cmap='gray')
#
#     ax.axis('off')
#     __fitFigure(fig, image.shape, maxwidth, maxheight)
#     plt.show()
#
#
# def showGray(image, maxwidth=800, maxheight=800):
#     """Отображение серого изображения"""
#     if len(image.shape) > 2:
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#
#     fig, ax = plt.subplots()
#     ax.imshow(image, cmap='gray', vmin=0, vmax=255 if image.dtype == np.uint8 else None)
#     ax.axis('off')
#     __fitFigure(fig, image.shape, maxwidth, maxheight)
#     plt.show()


# def showVecField(x, y, u, v, maxwidth=800, maxheight=800):
#     """Отображение векторного поля"""
#     fig, ax = plt.subplots()
#
#     # Масштабируем векторы для лучшей визуализации
#     magnitude = np.sqrt(u**2 + v**2)
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
#
#     __fitFigure(fig, (np.max(x + u), np.max(y + v)), maxwidth, maxheight)
#     plt.show()
#
#
# def quiverImage(shape, x, y, u, v, color1=(0, 0, 255, 255), color2=(0, 255, 0, 255), qMaxLen=0.05):
#     """Создание изображения с векторами"""
#     assert x.shape == y.shape and x.shape == u.shape and x.shape == v.shape
#     assert np.min(x) >= 0 and np.min(y) >= 0 and np.max(x) < shape[1] and np.max(y) < shape[0]
#
#     # Создаем пустое RGB изображение
#     Q = np.zeros((*shape, 3), dtype=np.uint8)
#
#     # Нормализация длин векторов
#     l = min(shape) * qMaxLen
#     s = np.sqrt(u * u + v * v)
#     ms = np.max(s) if np.max(s) > 0 else 1
#     scale = l / ms
#
#     u_scaled = scale * u
#     v_scaled = scale * s
#
#     # Преобразуем координаты в целые числа
#     x = x.astype(np.int32)
#     y = y.astype(np.int32)
#     xu = (x + u_scaled).astype(np.int32)
#     yv = (y + v_scaled).astype(np.int32)
#
#     # Ограничиваем координаты границами изображения
#     xu = np.clip(xu, 0, shape[1] - 1)
#     yv = np.clip(yv, 0, shape[0] - 1)
#
#     # Создаем градиент цветов
#     color1_rgb = np.array(color1[:3], dtype=np.uint8)
#     color2_rgb = np.array(color2[:3], dtype=np.uint8)
#
#     # Рисуем векторы
#     for i in range(y.shape[0]):
#         for j in range(x.shape[1]):
#             # Интерполяция цвета в зависимости от длины вектора
#             t = s[i, j] / l if l > 0 else 0
#             color = color1_rgb * (1 - t) + color2_rgb * t
#             color = color.astype(np.uint8)
#
#             # Рисуем линию (стрелку)
#             cv2.arrowedLine(Q,
#                             (x[i, j], y[i, j]),
#                             (xu[i, j], yv[i, j]),
#                             tuple(map(int, color)),
#                             thickness=1,
#                             tipLength=0.3)
#
#     return Q


# def showGrayGrad(G1, G2, showField=False, step=(1, 1), original=None):
#     """
#     Отображение градиента серого изображения:
#         - Первая компонента градиента
#         - Вторая компонента градиента
#         - Величина градиента
#         - (опционально) Векторное поле градиента
#     """
#     assert G1.shape == G2.shape
#
#     # Создаем общую фигуру для всех графиков
#     if showField and original is not None:
#         fig, axes = plt.subplots(2, 2, figsize=(12, 10))
#         ax_idx = 0
#     else:
#         fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#         axes = axes.flatten()
#         ax_idx = 0
#
#     # Первая компонента градиента
#     im1 = axes[ax_idx].imshow(G1, cmap='gray')
#     axes[ax_idx].set_title("Первая компонента градиента")
#     axes[ax_idx].axis('off')
#     ax_idx += 1
#
#     # Вторая компонента градиента
#     im2 = axes[ax_idx].imshow(G2, cmap='gray')
#     axes[ax_idx].set_title("Вторая компонента градиента")
#     axes[ax_idx].axis('off')
#     ax_idx += 1
#
#     # Величина градиента
#     Magnitude = np.sqrt(G1 * G1 + G2 * G2)
#     im3 = axes[ax_idx].imshow(Magnitude, cmap='gray')
#     axes[ax_idx].set_title("Величина градиента")
#     axes[ax_idx].axis('off')
#     ax_idx += 1
#
#     # Векторное поле (если нужно)
#     if showField:
#         x, y = np.meshgrid(range(0, G1.shape[1], step[1]), range(0, G1.shape[0], step[0]))
#         Q = quiverImage(G1.shape, x, y, G1[::step[0], ::step[1]], G2[::step[0], ::step[1]])
#
#         if original is not None:
#             if len(original.shape) < 3:
#                 original_rgb = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
#             else:
#                 original_rgb = original
#
#             # Наложение векторного поля на исходное изображение
#             combined = cv2.addWeighted(original_rgb, 0.7, Q, 0.3, 0)
#             axes[ax_idx].imshow(combined)
#         else:
#             axes[ax_idx].imshow(Q)
#
#         axes[ax_idx].set_title("Векторное поле градиента")
#         axes[ax_idx].axis('off')
#
#     plt.tight_layout()
#     plt.show()
#
#
# def plotVectorContour(img, start, vc, color=(255, 0, 0), thickness=3):
#     """Рисование векторного контура на изображении"""
#     y, x = np.int32(start)
#     for v in vc:
#         y1, x1 = np.int32(y + v.imag), np.int32(x + v.real)
#         cv2.line(img, (x, y), (x1, y1), color, thickness)
#         y, x = y1, x1
#
#
# def plotVectorContours(VCs, img=None, shape=None, color=(255, 0, 0), thickness=3):
#     """Рисование нескольких векторных контуров на изображении"""
#     if img is None:
#         if shape is None:
#             raise ValueError("Либо 'img', либо 'shape' должен быть задан")
#         img = np.zeros((*shape, 3), dtype=np.uint8)
#     elif shape is not None:
#         raise ValueError("Либо 'img', либо 'shape' должен быть задан, не оба")
#     elif len(img.shape) < 3:
#         img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#     else:
#         img = np.copy(img)
#
#     for start, v in VCs:
#         plotVectorContour(img, start, v, color, thickness)
#
#     return img


def create_test_image_A():
    """Создание тестового изображения A"""
    image = np.array([
        [1,1,1,1,1,1,1,1,1],
        [1,1,0,1,1,1,0,1,1],
        [1,0,0,0,1,1,1,0,1],
        [1,1,1,1,1,1,1,1,1],
        [1,1,1,0,1,1,1,1,1],
        [1,1,1,0,0,1,1,1,1],
        [1,1,1,0,0,0,1,1,1],
        [1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1]
    ], dtype=np.uint8) * 255
   
    return image


def create_test_marker_from_A():
    """Создание тестового маркера из изображения A"""
    m = np.zeros((9,9), dtype=np.uint8)
    # Некоторые координаты — убедись, что они внутри белой области маски
    m[1,2] = 255
    m[2,1] = 255
    m[2,2] = 255
    m[4,3] = 255
    return m


def condDilate_reconstruction(mask, marker, selem=None, max_iter=1000):
    """Условная дилатация (реконструкция)"""
    if selem is None:
        selem = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    curr = mask.copy()
    it = 0

    curr = cv2.erode(curr, marker)
    while True:
        dil = cv2.dilate(curr, selem)
        curr_next = cv2.bitwise_and(dil, mask)
        it += 1
        if np.array_equal(curr_next, curr) or it >= max_iter:
            break
        curr = curr_next

    print(f"Реконструкция завершена за {it} итераций")
    return curr


def conditional_dilation(A=None, B=None, C=None):
    """Функция для демонстрации условной дилатации"""
    if A is None:
        A = create_test_image_A()
    if B is None:
        B = create_test_marker_from_A()
    if C is None:
        C = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    res = condDilate_reconstruction(A, B, C, max_iter=1000)

    # Отображение результатов
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(A, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Маска (A)')
    axes[0].axis('off')
    
    axes[1].imshow(B, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('Маркер (B)')
    axes[1].axis('off')
    
    axes[2].imshow(res, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title('Результат реконструкции')
    axes[2].axis('off')
    
    plt.suptitle('Условная дилатация (реконструкция)')
    plt.tight_layout()
    plt.show()


def morphological_skeleton(img, selem=None, max_iter=None, visualize=False):
    """Вычисление морфологического скелета бинарного изображения"""
    # к 0/255 uint8
    bin_img = (img > 0).astype(np.uint8) * 255

    if selem is None:
        # крест (4-соседство) часто даёт более «скелетообразные» линии
        selem = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    # Для кумуляции результата
    skeleton = np.zeros_like(bin_img, dtype=np.uint8)
    layers = []  # если нужно визуализировать

    # Текущий эродированный образ
    eroded = bin_img.copy()
    iteration = 0
    if max_iter is None:
        max_iter = bin_img.shape[0] * bin_img.shape[1]  # очень большой потолок

    while True:
        # 1) эрозия на один шаг
        eroded_next = cv2.erode(eroded, selem)
        if np.all(eroded_next == 0):
            # если следующая эрозия полностью исчезает, от неё нет слоя,
            # но остаётся возможно последний слой: E_k - O_k где E_k = eroded
            opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, selem)
            sk_layer = cv2.subtract(eroded, opened)
            skeleton = cv2.bitwise_or(skeleton, sk_layer)
            if visualize:
                layers.append(sk_layer.copy())
            break

        # 2) открытие текущего eroded (то есть E_k -> O_k)
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, selem)

        # 3) слой скелета S_k = E_k - O_k
        sk_layer = cv2.subtract(eroded, opened)
        skeleton = cv2.bitwise_or(skeleton, sk_layer)
        if visualize:
            layers.append(sk_layer.copy())

        # подготовка к следующей итерации
        eroded = eroded_next
        iteration += 1
        if iteration >= max_iter:
            print("Достигнут max_iter, прерываю")
            break

    # skeleton уже 0/255
    if visualize:
        return skeleton, layers
    return skeleton


def morphological_skeleton_show():
    """Функция для демонстрации морфологического скелета"""

    rectangle_image = "C:/Users/xcrea/Documents/YandexPython/pythonProject/img/cats3.jpg"
    
    try:
        I = cv2.cvtColor(cv2.imread(rectangle_image), cv2.COLOR_RGB2GRAY)
        _, I = cv2.threshold(I, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        sk = morphological_skeleton(I)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(I, cmap='gray', vmin=0, vmax=255)
        axes[0].set_title('Оригинал')
        axes[0].axis('off')
        
        axes[1].imshow(sk, cmap='gray', vmin=0, vmax=255)
        axes[1].set_title('Скелет')
        axes[1].axis('off')
        
        plt.suptitle('Морфологический скелет')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Ошибка при загрузке изображения: {e}")
        print("Создаем тестовое изображение...")
        
        # Создаем тестовое изображение
        test_img = np.zeros((100, 150), dtype=np.uint8)
        test_img[20:80, 30:120] = 255
        
        sk = morphological_skeleton(test_img)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(test_img, cmap='gray', vmin=0, vmax=255)
        axes[0].set_title('Тестовое изображение')
        axes[0].axis('off')
        
        axes[1].imshow(sk, cmap='gray', vmin=0, vmax=255)
        axes[1].set_title('Скелет')
        axes[1].axis('off')
        
        plt.suptitle('Морфологический скелет (тестовый пример)')
        plt.tight_layout()
        plt.show()