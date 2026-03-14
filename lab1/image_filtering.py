# %%
import cv2
import numpy as np
import plotly.graph_objects as go
from matplotlib import pyplot as plt

imfile = "C:/Users/xcrea/Documents/YandexPython/pythonProject/img/Karelia1.jpg"


# %%
# Tools
def _2dfunCentered(f, shape):
    muy = np.double(shape[0] // 2)
    if shape[0] % 2 == 0:
        muy -= 0.5
    mux = np.double(shape[1] // 2)
    if shape[1] % 2 == 0:
        mux -= 0.5

    fun = lambda y, x: f(y - muy, x - mux)

    return np.fromfunction(fun, shape, dtype=np.double)


def gauss2d(shape, sigma):
    return _2dfunCentered(
        lambda y, x: np.exp(-(x * x + y * y) / (2 * sigma * sigma)), shape
    )


def LoG(shape, sigma):
    return _2dfunCentered(
        lambda y, x: (x * x + y * y - 2 * sigma * sigma)
        / sigma**4
        * np.exp(-(x * x + y * y) / (2 * sigma * sigma)),
        shape,
    )


# %%
# Load image and convert it to grayscale
image = cv2.cvtColor(cv2.imread(imfile), cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()

I = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
plt.imshow(I, cmap="gray")
plt.show()

fig = go.Figure(go.Surface(z=I))
fig.show()

# %%
# Fourier
F = np.fft.fft2(I)
aspectrum = np.abs(F)

yfreq = np.fft.fftfreq(F.shape[0])
xfreq = np.fft.fftfreq(F.shape[1])

fig = go.Figure(go.Surface(x=xfreq, y=yfreq, z=aspectrum))
fig.update_layout(title="Fourier Spectrum")
fig.show()

# %%
# Gaussian
G = gauss2d(F.shape, 30)
plt.imshow(G)
fig = go.Figure(go.Surface(z=G))
fig.update_layout(title="Gaussian Filter in Frequency Domain")
fig.show()

# %% Filtering in frequency domain (custom implementation)
shiftedG = np.fft.fftshift(G)
FilteredFourier = F * shiftedG
Filtered = np.fft.ifft2(FilteredFourier).real
Filtered = cv2.normalize(Filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

plt.title("Custom Gaussian Filter (Freq Domain)")
plt.imshow(Filtered, cmap="gray")
plt.show()


# %% Кастомная свертка
def gaussian_kernel(size, sigma):
    """Ядро Гаусса, номализация"""
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma**2))
        * np.exp(
            -((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma**2)
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
            output[y, x] = np.sum(padded[y : y + kh, x : x + kw] * kernel)

    return output


# Параметры ядра Гаусса
kernel_size = 15
sigma = 3.0
gauss_kernel = gaussian_kernel(kernel_size, sigma)

# Отобразить кастомную свертку
custom_blurred = custom_convolution(I, gauss_kernel)
plt.imshow(custom_blurred, cmap="gray")
plt.title("Custom Gaussian Filter (Spatial Domain)")
plt.show()

# %% OpenCV implementation
opencv_blurred = cv2.GaussianBlur(I, (kernel_size, kernel_size), sigma)
plt.imshow(opencv_blurred, cmap="gray")
plt.title("OpenCV Implementation")
plt.show()

# %% Сравнение Оригинала, Кастомной реализации, OpenCV
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.imshow(I, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(132)
plt.imshow(custom_blurred, cmap="gray")
plt.title("Custom Implementation")
plt.axis("off")

plt.subplot(133)
plt.imshow(opencv_blurred, cmap="gray")
plt.title("OpenCV Implementation")
plt.axis("off")

plt.tight_layout()
plt.show()


# %%
# Highpass
HP = 1 - G
fig = go.Figure(go.Surface(z=HP))
fig.show()

# %%
# Highpass filtering
shiftedHP = 1 - shiftedG
HighFourier = F * shiftedHP

High = np.fft.ifft2(HighFourier).real

plt.imshow(High, cmap="gray")
