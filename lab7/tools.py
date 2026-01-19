import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
import cv2
import matplotlib.pyplot as plt


def gaussian_kernel(size, sigma):
    """Ядро Гаусса, номализация"""
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2))
                     * np.exp(
            -((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)
        ),
        (size, size),
    )
    return kernel / np.sum(kernel)


def custom_convolution(image, kernel):
    """Кастомная свертка с ядром Гаусса"""
    ih, iw = image.shape
    kh, kw = kernel.shape

    pad = kh // 2
    padded = np.pad(image, pad, mode="constant")

    output = np.zeros_like(image)

    for y in range(ih):
        for x in range(iw):
            output[y, x] = np.sum(padded[y: y + kh, x: x + kw] * kernel)

    return output


def __fitFigure(fig, shape, maxwidth=800, maxheight=800):
    layout_params = {
        'margin': {
            'l': 0,
            'r': 0,
            't': 0,
            'b': 0,
        },
        'height': min(shape[0], maxheight),
        'width': min(shape[1], maxwidth)
    }
    if shape[0] > shape[1]:
        layout_params['xaxis_scaleanchor'] = 'y'
    else:
        layout_params['yaxis_scaleanchor'] = 'x'
    fig.update_layout(**layout_params)
    fig.update_yaxes(autorange='reversed')


def __buildPlotlyImage(image):
    if len(image.shape) > 2:
        img = go.Image(z=image)
    else:
        img = go.Heatmap(z=image, colorscale='gray')
    return img


def showImage(image, maxwidth=800, maxheight=800):
    fig = go.Figure(__buildPlotlyImage(image))
    __fitFigure(fig, image.shape, maxwidth, maxheight)
    # fig.show()


def showGray(image, maxwidth=800, maxheight=800):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    showImage(image, maxwidth, maxheight)


def showGrayInput(image, maxwidth=800, maxheight=800):
    showImage(showGray(image, maxwidth=maxwidth, maxheight=maxheight), maxwidth, maxheight)


def showVecField(x, y, u, v, maxwidth=800, maxheight=800):
    fig = ff.create_quiver(x, y, u, v)
    __fitFigure(fig, (np.max(x + u), np.max(y + v)), maxwidth, maxheight)
    fig.show()


def quiverImage(shape, x, y, u, v, color1=(0, 0, 255, 255), color2=(0, 255, 0, 255), qMaxLen=0.05):
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


def showGrayGrad(G1, G2, showField=False, step=(1, 1), original=None):
    """
    Displays grayscale image gradient in separate images:
        - First direction of gradient
        - Second direction of gradient
        - Magnitude
        - (optional) Gradien field if needed

    Parameters
    -------
        G1, G2 :
            Gradients 1st and 2nd coordinate matricies
        showField : optional
            If True, will also display a gradient vector field
        step : tuple-like, optional
            X and Y steps for the gradient vector field grid
        original : optional
            Original image to draw gradient vector field on
    """
    assert G1.shape == G2.shape

    showGray(G1, maxheight=500)
    showGray(G2, maxheight=500)

    Magnitude = np.sqrt(G1 * G1 + G2 * G2)
    showGray(Magnitude, maxheight=500)

    if showField:
        x, y = np.meshgrid(range(0, G1.shape[1], step[1]), range(0, G1.shape[0], step[0]))
        Q = quiverImage(G1.shape, x, y, G1[::step[0], ::step[1]], G2[::step[0], ::step[1]])

        fig = go.Figure()
        if original is not None:
            if len(original.shape) < 3:
                original = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
            fig.add_image(z=original)
        fig.add_image(z=Q, opacity=0.5 if original is not None else 1)

        __fitFigure(fig, G1.shape)
        fig.show()

    return G1, G2, Magnitude


def plotVectorContour(img, start, vc, color=(255, 0, 0), thickness=3):
    """
    Plots a vector-contour in the desired position of the image

    Parameters
    --------
        img :
            An rgb image to plot the vector-contour on
        start : Tuple-like
            A pair of x, y coordinates representing the vector-contour origin on the img
        vc : A sequence of complex numbers
            The vector-contour to plot
        color : Tuple-like, optional
            A color to plot the vc with
    """
    y, x = np.int32(start)
    for v in vc:
        y1, x1 = np.int32(y + v.imag), np.int32(x + v.real)
        cv2.line(img, (x, y), (x1, y1), color, thickness)
        y, x = y1, x1


def plotVectorContours(VCs, img=None, shape=None, color=(255, 0, 0), thickness=3):
    """
    Plots multiple vector-contours in the image

    Parameters
    --------
        vc : A sequence of pairs (Tuple-like, Sequence of complex numbers)
            The sequence of pairs of vector-contours origins and vector-contours themselves
        img: Mat-like, optional
            An image to plot on its copy.
            If None, will create and return new RGB-image.
            If grayscale, create an return new RGB-image with all 3 channels equal to img
            Either 'img' or 'shape' should be set, not both.
        shape : A pair of integer numbers, optional
            A desired image height and width.
            Either 'img' or 'shape' should be set, not both.
        color : Tuple-like, optional
            A color to plot the vector-contours with
    Returns
    -------
    An RGB-image with vector-contours plotted on it
    """
    if img is None:
        if shape is None:
            raise ValueError("Either 'img' or 'shape' should be set")
        img = np.zeros((*shape, 3), dtype=np.uint8)
    elif shape is not None:
        raise ValueError("Either 'img' or 'shape' should be set, not both")
    elif len(img.shape) < 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = np.copy(img)

    for start, v in VCs:
        plotVectorContour(img, start, v, color, thickness)

    return img


def grad(I, method):
    """Вычисление градиента изображения с использованием заданного метода"""
    kx, ky = method()
    Gx = cv2.filter2D(I, cv2.CV_64F, kx)
    Gy = cv2.filter2D(I, cv2.CV_64F, ky)
    return Gx, Gy


def edgesGrad(I, method, threshold=30):
    Gx, Gy = grad(I, method)
    magnitude = np.sqrt(Gx ** 2 + Gy ** 2)
    edges = np.zeros_like(magnitude)
    edges[magnitude > threshold] = 255
    return edges


def show_images(images, titles, figsize=(20, 4), cmap='gray', vmin=0, vmax=255):
    """
    Отображает несколько изображений в ряд

    Parameters:
    - images: список изображений
    - titles: список заголовков
    - figsize: размер фигуры
    - cmap: цветовая карта
    - vmin, vmax: диапазон значений
    """
    n = len(images)
    plt.figure(figsize=figsize)

    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i], cmap=cmap, vmin=vmin, vmax=vmax)
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def laplacian_edge_detection(image, threshold_percent=4, laplacian_kernel=None):
    """
    Применяет оператор Лапласа для выделения границ с условиями zero-crossing и порогом 4%

    Args:
        image: входное изображение (в оттенках серого)
        threshold_percent: порог в процентах для разности по модулю (по умолчанию 4%)
        laplacian_kernel: ядро Лапласа

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

    return edges
