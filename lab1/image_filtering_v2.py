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
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(image)
plt.title('Original RGB Image')
plt.axis('off')
plt.show()

I = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(I, cmap="gray")
plt.title('Grayscale Image')
plt.axis('off')
plt.show()

fig = go.Figure(go.Surface(z=I, colorscale='Viridis'))
fig.update_layout(title='3D Surface Plot of Grayscale Image',
                  scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Intensity'))
fig.show()


# %%
# Fourier
F = np.fft.fft2(I)
aspectrum = np.abs(F)
aspectrum_log = np.log1p(aspectrum)  # Log scale for better visualization

yfreq = np.fft.fftfreq(F.shape[0])
xfreq = np.fft.fftfreq(F.shape[1])

# Create meshgrid for frequency coordinates
X, Y = np.meshgrid(xfreq, yfreq)

# Plot Fourier spectrum in 3D
fig = go.Figure(go.Surface(x=X, y=Y, z=aspectrum_log, colorscale='Hot'))
fig.update_layout(title='Fourier Spectrum (Log Scale)',
                  scene=dict(xaxis_title='Frequency X', yaxis_title='Frequency Y', zaxis_title='Magnitude (log)'))
fig.show()

# 2D visualization of spectrum
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(np.log1p(np.abs(np.fft.fftshift(F))), cmap='gray')
plt.title('Fourier Spectrum (Centered)')
plt.colorbar(label='log magnitude')
plt.axis('off')

plt.subplot(132)
plt.imshow(np.angle(np.fft.fftshift(F)), cmap='hsv')
plt.title('Phase Spectrum')
plt.colorbar(label='phase (rad)')
plt.axis('off')

plt.subplot(133)
plt.hist(aspectrum.flatten(), bins=100, log=True)
plt.title('Spectrum Histogram')
plt.xlabel('Magnitude')
plt.ylabel('Frequency (log)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %%
# Gaussian
sigma_gauss = 30
G = gauss2d(F.shape, sigma_gauss)
G_shifted = np.fft.fftshift(G)

# 3D visualization of Gaussian filter
fig = go.Figure(go.Surface(z=G, colorscale='Blues'))
fig.update_layout(title=f'Gaussian Filter in Frequency Domain (σ={sigma_gauss})',
                  scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Amplitude'))
fig.show()

# 2D visualization
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(G_shifted, cmap='viridis', extent=[-0.5, 0.5, -0.5, 0.5])
plt.title('Gaussian Filter (Centered)')
plt.colorbar(label='Amplitude')
plt.xlabel('Normalized Frequency')
plt.ylabel('Normalized Frequency')

plt.subplot(132)
# Cross-section through center
center_y = G_shifted.shape[0] // 2
plt.plot(np.linspace(-0.5, 0.5, G_shifted.shape[1]), G_shifted[center_y, :])
plt.title('Cross-section of Gaussian Filter')
plt.xlabel('Normalized Frequency')
plt.ylabel('Amplitude')
plt.grid(True, alpha=0.3)

plt.subplot(133)
# 1D Gaussian for comparison
x = np.linspace(-3*sigma_gauss, 3*sigma_gauss, 100)
gauss_1d = np.exp(-x**2/(2*sigma_gauss**2))
plt.plot(x, gauss_1d)
plt.title('1D Gaussian (for reference)')
plt.xlabel('x')
plt.ylabel('Amplitude')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# %% Filtering in frequency domain (custom implementation)
shiftedG = np.fft.fftshift(G)
FilteredFourier = F * shiftedG
Filtered = np.fft.ifft2(FilteredFourier).real
Filtered = cv2.normalize(Filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Visualize filtering process
plt.figure(figsize=(15, 10))

plt.subplot(231)
plt.imshow(I, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(232)
plt.imshow(G_shifted, cmap='viridis')
plt.title('Gaussian Filter')
plt.axis('off')

plt.subplot(233)
filtered_spectrum = np.log1p(np.abs(np.fft.fftshift(FilteredFourier)))
plt.imshow(filtered_spectrum, cmap='hot')
plt.title('Filtered Spectrum (log)')
plt.axis('off')

plt.subplot(234)
plt.imshow(Filtered, cmap='gray')
plt.title('Filtered Image')
plt.axis('off')

plt.subplot(235)
# Difference from original
difference = cv2.absdiff(I, Filtered)
plt.imshow(difference, cmap='hot')
plt.title('Difference (Original - Filtered)')
plt.axis('off')
plt.colorbar()

plt.subplot(236)
# Histogram comparison
plt.hist(I.flatten(), bins=50, alpha=0.5, label='Original', density=True)
plt.hist(Filtered.flatten(), bins=50, alpha=0.5, label='Filtered', density=True)
plt.title('Histogram Comparison')
plt.xlabel('Intensity')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# %% Кастомная свертка
def gaussian_kernel(size, sigma):
    """Ядро Гаусса, нормализация"""
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
sigma_spatial = 3.0
gauss_kernel = gaussian_kernel(kernel_size, sigma_spatial)

# Visualize the kernel
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.imshow(gauss_kernel, cmap='viridis', extent=[-kernel_size//2, kernel_size//2, -kernel_size//2, kernel_size//2])
plt.title(f'Gaussian Kernel ({kernel_size}x{kernel_size}, σ={sigma_spatial})')
plt.colorbar(label='Weight')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(132)
# Cross-section of kernel
center = kernel_size // 2
plt.plot(range(-center, center+1), gauss_kernel[center, :], 'o-')
plt.title('Kernel Cross-section')
plt.xlabel('Position')
plt.ylabel('Weight')
plt.grid(True, alpha=0.3)

plt.subplot(133)
# 3D visualization of kernel
x = np.arange(-center, center+1)
y = np.arange(-center, center+1)
X, Y = np.meshgrid(x, y)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, gauss_kernel, cmap='viridis')
ax.set_title('3D Kernel Visualization')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Weight')

plt.tight_layout()
plt.show()

# Отобразить кастомную свертку
custom_blurred = custom_convolution(I, gauss_kernel)

# Visualize convolution process
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.imshow(I[:50, :50], cmap='gray')
plt.title('Image Patch (50x50)')
plt.axis('off')

plt.subplot(132)
# Show kernel overlapped on image
patch = I[:kernel_size, :kernel_size].copy()
overlay = np.zeros((kernel_size, kernel_size, 3))
overlay[:,:,0] = patch/255.0  # Red channel - original
overlay[:,:,1] = gauss_kernel/np.max(gauss_kernel)  # Green channel - kernel
plt.imshow(overlay)
plt.title('Kernel Overlay (green) on Image (red)')
plt.axis('off')

plt.subplot(133)
plt.imshow(custom_blurred[:50, :50], cmap='gray')
plt.title('Result Patch (50x50)')
plt.axis('off')

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.imshow(custom_blurred, cmap="gray")
plt.title("Custom Gaussian Filter (Spatial Domain)")
plt.colorbar(label='Intensity')
plt.axis('off')
plt.show()


# %% OpenCV implementation
opencv_blurred = cv2.GaussianBlur(I, (kernel_size, kernel_size), sigma_spatial)

plt.figure(figsize=(10, 5))
plt.imshow(opencv_blurred, cmap="gray")
plt.title("OpenCV Implementation")
plt.colorbar(label='Intensity')
plt.axis('off')
plt.show()


# %% Сравнение Оригинала, Кастомной реализации, OpenCV
plt.figure(figsize=(15, 10))

plt.subplot(231)
plt.imshow(I, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(232)
plt.imshow(custom_blurred, cmap="gray")
plt.title("Custom Implementation")
plt.axis("off")

plt.subplot(233)
plt.imshow(opencv_blurred, cmap="gray")
plt.title("OpenCV Implementation")
plt.axis("off")

# Difference maps
plt.subplot(234)
diff_custom = cv2.absdiff(I, custom_blurred)
plt.imshow(diff_custom, cmap='hot')
plt.title("Difference (Original - Custom)")
plt.axis("off")
plt.colorbar()

plt.subplot(235)
diff_opencv = cv2.absdiff(I, opencv_blurred)
plt.imshow(diff_opencv, cmap='hot')
plt.title("Difference (Original - OpenCV)")
plt.axis("off")
plt.colorbar()

plt.subplot(236)
diff_methods = cv2.absdiff(custom_blurred, opencv_blurred)
plt.imshow(diff_methods, cmap='hot')
plt.title("Difference (Custom - OpenCV)")
plt.axis("off")
plt.colorbar()

plt.tight_layout()
plt.show()

# Quantitative comparison
print("Quantitative Comparison:")
print(f"Original mean: {np.mean(I):.2f}, std: {np.std(I):.2f}")
print(f"Custom mean: {np.mean(custom_blurred):.2f}, std: {np.std(custom_blurred):.2f}")
print(f"OpenCV mean: {np.mean(opencv_blurred):.2f}, std: {np.std(opencv_blurred):.2f}")
print(f"MAE (Custom vs Original): {np.mean(np.abs(I.astype(float) - custom_blurred.astype(float))):.2f}")
print(f"MAE (OpenCV vs Original): {np.mean(np.abs(I.astype(float) - opencv_blurred.astype(float))):.2f}")
print(f"MAE (Custom vs OpenCV): {np.mean(np.abs(custom_blurred.astype(float) - opencv_blurred.astype(float))):.2f}")


# %%
# Highpass
HP = 1 - G
HP_shifted = 1 - np.fft.fftshift(G)

# 3D visualization of high-pass filter
fig = go.Figure(go.Surface(z=HP, colorscale='Reds'))
fig.update_layout(title='High-Pass Filter in Frequency Domain',
                  scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Amplitude'))
fig.show()

# 2D visualization
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.imshow(HP_shifted, cmap='hot', extent=[-0.5, 0.5, -0.5, 0.5])
plt.title('High-Pass Filter (Centered)')
plt.colorbar(label='Amplitude')
plt.xlabel('Normalized Frequency')
plt.ylabel('Normalized Frequency')

plt.subplot(132)
# Cross-section
center_y = HP_shifted.shape[0] // 2
plt.plot(np.linspace(-0.5, 0.5, HP_shifted.shape[1]), HP_shifted[center_y, :])
plt.title('Cross-section of High-Pass Filter')
plt.xlabel('Normalized Frequency')
plt.ylabel('Amplitude')
plt.grid(True, alpha=0.3)

plt.subplot(133)
# Comparison with low-pass
plt.plot(np.linspace(-0.5, 0.5, G_shifted.shape[1]), G_shifted[center_y, :], label='Low-Pass')
plt.plot(np.linspace(-0.5, 0.5, HP_shifted.shape[1]), HP_shifted[center_y, :], label='High-Pass')
plt.title('Filter Comparison')
plt.xlabel('Normalized Frequency')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# %%
# Highpass filtering
shiftedHP = 1 - shiftedG
HighFourier = F * shiftedHP

High = np.fft.ifft2(HighFourier).real
High_normalized = cv2.normalize(High, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Visualize high-pass filtering results
plt.figure(figsize=(15, 10))

plt.subplot(231)
plt.imshow(I, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(232)
plt.imshow(HP_shifted, cmap='hot')
plt.title('High-Pass Filter')
plt.axis('off')

plt.subplot(233)
highpass_spectrum = np.log1p(np.abs(np.fft.fftshift(HighFourier)))
plt.imshow(highpass_spectrum, cmap='hot')
plt.title('High-Pass Filtered Spectrum (log)')
plt.axis('off')

plt.subplot(234)
plt.imshow(High_normalized, cmap='gray')
plt.title('High-Pass Filtered Image')
plt.axis('off')

plt.subplot(235)
# Enhanced edges (absolute value)
edges = np.abs(High)
edges_normalized = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
plt.imshow(edges_normalized, cmap='hot')
plt.title('Edge Magnitude')
plt.axis('off')
plt.colorbar()

plt.subplot(236)
# Overlay edges on original
overlay = I.copy().astype(float)
overlay += edges_normalized * 0.5
overlay = np.clip(overlay, 0, 255).astype(np.uint8)
plt.imshow(overlay, cmap='gray')
plt.title('Original + Edges')
plt.axis('off')

plt.tight_layout()
plt.show()

# 3D visualization of high-pass result
fig = go.Figure(go.Surface(z=High_normalized, colorscale='Gray'))
fig.update_layout(title='High-Pass Filtered Image (3D Surface)',
                  scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Intensity'))
fig.show()