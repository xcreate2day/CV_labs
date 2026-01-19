import cv2
import numpy as np
import matplotlib.pyplot as plt


def showGray(image, maxwidth=800, maxheight=800):
    plt.figure(figsize=(maxwidth/100, maxheight/100))
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def calculate_histogram(image):
    hist, bins = np.histogram(image.flatten(), bins=256, range=(0, 256))
    return hist, bins


def show_histogram_with_threshold(image, threshold=None, title="Гистограмма яркости"):
    hist, bins = calculate_histogram(image)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(hist)), hist, width=1.0)
    
    if threshold is not None:
        plt.axvline(x=threshold, color='red', linestyle='--', 
                   linewidth=2, label=f'Порог: {threshold}')
        plt.legend()
    
    plt.title(title)
    plt.xlabel("Яркость")
    plt.ylabel("Количество пикселей")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def otsu_threshold_manual(image):
    """
    Ручная реализация алгоритма Оцу для поиска оптимального порога

    Алгоритм работы:
    1) Гистограмма: Строится гистограмма яркости изображения (256 уровней)
    2) Вероятности: Для каждого возможного порога t (0-255) вычисляются:
        - ω₀ = вероятность класса фона (пиксели ≤ t)
        - ω₁ = вероятность класса объектов (пиксели > t)
    3) Средние значения: Вычисляются средние яркости для каждого класса
    4) Межклассовая дисперсия: σ² = ω₀·ω₁·(μ₀ - μ₁)²
    5) Оптимизация: Находится t, который максимизирует σ²
    
    """
    # Вычисляем гистограмму
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel()
    hist = hist.astype(np.float)

    total_pixels = image.size
    
    # Инициализируем переменные для поиска максимума
    current_max = 0
    optimal_threshold = 0
    
    # Перебираем все возможные пороги
    for threshold in range(256):
        # Вероятность класса 0 (фон)
        w0 = np.sum(hist[:threshold]) / total_pixels
        # Вероятность класса 1 (объекты)
        w1 = np.sum(hist[threshold:]) / total_pixels
        
        # Избегаем деления на ноль
        if w0 == 0 or w1 == 0:
            continue
            
        # Средние значения для каждого класса
        mean0 = np.sum(np.arange(threshold) * hist[:threshold]) / w0
        mean1 = np.sum(np.arange(threshold, 256) * hist[threshold:]) / w1
        
        # Межклассовая дисперсия
        between_variance = w0 * w1 * (mean0 - mean1)**2
        
        # Обновляем оптимальный порог
        if between_variance > current_max:
            current_max = between_variance
            optimal_threshold = threshold
    
    return optimal_threshold


def integral_image(image):
    rows, cols = image.shape
    integral = np.zeros((rows + 1, cols + 1), dtype=np.int64)
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            integral[i, j] = (
                    image[i - 1, j - 1]
                    + integral[i - 1, j]
                    + integral[i, j - 1]
                    - integral[i - 1, j - 1]
            )
    return integral


def bradley_binarization(image, window_size=5, threshold_ratio=0.05):
    rows, cols = image.shape
    s = max(rows, cols) * window_size // 100
    if s % 2 == 0:
        s += 1
    integral = integral_image(image)
    binary_image = np.zeros_like(image, dtype=np.uint8)  # Явно указываем тип
    half_s = s // 2

    print(f"Параметры бинаризации Брэдли:")
    print(f"  Размер окна: {s}x{s} пикселей")
    print(f"  Коэффициент порога: {threshold_ratio}")

    for i in range(rows):
        for j in range(cols):
            y1 = max(0, i - half_s)
            y2 = min(rows - 1, i + half_s)
            x1 = max(0, j - half_s)
            x2 = min(cols - 1, j + half_s)
            area = (y2 - y1 + 1) * (x2 - x1 + 1)
            sum_pixels = (
                    integral[y2 + 1, x2 + 1]
                    - integral[y1, x2 + 1]
                    - integral[y2 + 1, x1]
                    + integral[y1, x1]
            )
            mean_value = sum_pixels / area
            threshold = mean_value * (1 - threshold_ratio)
            binary_image[i, j] = 255 if image[i, j] > threshold else 0

    return binary_image, integral


def run_otsu_binarization(image):
    """
    Ручная реализация бинаризации Оцу
    Возвращает: (порог, бинаризованное_изображение)
    """
    # Вычисляем гистограмму
    hist, bins = np.histogram(image.flatten(), bins=256, range=(0, 256))
    
    # Нормализуем гистограмму
    total_pixels = image.shape[0] * image.shape[1]
    hist_prob = hist.astype(float) / total_pixels
    
    # Ищем оптимальный порог
    max_variance = 0
    optimal_threshold = 0
    
    for t in range(1, 256):
        w0 = np.sum(hist_prob[:t])  # вес фона
        w1 = np.sum(hist_prob[t:])  # вес объектов
        
        if w0 == 0 or w1 == 0:
            continue
            
        mean0 = np.sum(np.arange(t) * hist_prob[:t]) / w0
        mean1 = np.sum(np.arange(t, 256) * hist_prob[t:]) / w1
        
        between_class_variance = w0 * w1 * (mean0 - mean1) ** 2
        
        if between_class_variance > max_variance:
            max_variance = between_class_variance
            optimal_threshold = t
    
    # Создаем бинаризованное изображение
    binary_image = np.zeros_like(image)
    binary_image[image > optimal_threshold] = 255
    
    return optimal_threshold, binary_image