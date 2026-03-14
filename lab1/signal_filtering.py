# %%
from typing import Callable

import numpy as np
import plotly.graph_objects as go


# %%
# Signals and noise functions
def signal(t: np.ndarray[np.double], f: Callable[[np.double], np.double]) -> np.ndarray:
    return np.vectorize(f)(t)


def addWhiteNoise(sig: np.ndarray, mu: np.double, sigma: np.double) -> np.ndarray:
    return sig + np.random.normal(mu, sigma, sig.size)


# %%
# Make test signals data
# Too low discretization step or interval comparing to frequencies
# will distort frequencies after fourier transform.
dt = 0.01
t = np.arange(-10, 10, dt)
f1 = 0.1
f2 = 0.2
s = signal(t, lambda t: np.sin(2 * np.pi * f1 * t))
s1 = signal(t, lambda t: np.sin(2 * np.pi * f2 * t))
sn = addWhiteNoise(s, 0, 0.3)

# %%
fig = go.Figure()
fig.add_scatter(x=t, y=s, mode="lines", name="Сигнал 1")
fig.add_scatter(x=t, y=s1, mode="lines", name="Сигнал 2")
fig.add_scatter(x=t, y=sn, mode="lines", name="Сигнал 1 + шум")
fig.show()

# %%
# Calculate fourier transform of the signals
fourier = np.fft.fft(s)
fourier2 = np.fft.fft(s1)
fourierNoise = np.fft.fft(sn)

aspectrum = np.abs(fourier)
aspectrum2 = np.abs(fourier2)

aspectrumNoise = np.abs(fourierNoise)

freq = np.fft.fftfreq(t.size, dt)  # Extract frequencies

# %%
fig = go.Figure()
fig.update_layout(title="Амплитудный спектр преобразования Фурье")
fig.add_scatter(x=freq, y=aspectrum, mode="lines", name="Сигнал 1")
fig.add_scatter(x=freq, y=aspectrum2, mode="lines", name="Сигнал 2")
fig.show()
# %%
# Gaussian function
sg = 0.5  # sigma
gauss = signal(t, lambda t: np.exp(-t * t / (2 * sg * sg)))
fig = go.Figure()
fig.add_scatter(x=t, y=gauss, name="Функция Гаусса")
fig.show()
# %%
# Rearange Gaussian function according to frequencies
n = len(gauss)
if n % 2 == 0:
    G = np.concatenate((gauss[n // 2 - 1:: -1], gauss[-1: -n // 2 - 1: -1]))
    # G = np.concat((gauss[n // 2 - 1 :: -1], gauss[-1 : -n // 2 - 1 : -1]))
else:
    G = np.concatenate((gauss[(n - 1) // 2 :: -1], gauss[-1 : -(n - 1) // 2 : -1]))
    # G = np.concatenate((gauss[(n - 1) // 2 :: -1], gauss[-1 : -(n - 1) // 2 : -1]))

fig = go.Figure()
fig.update_layout(title="Амплитудный спектр преобразования Фурье")
fig.add_scatter(x=freq, y=aspectrumNoise, mode="lines", name="Сигнал + шум")
fig.add_scatter(x=freq, y=G, name="Функция Гаусса")
fig.show()

# %%
# Filtering
filteredSpectrum = fourierNoise * G
aspectrumFiltered = np.abs(filteredSpectrum)
fig = go.Figure()
fig.update_layout(title="Амплитудный спектр преобразования Фурье")
fig.add_scatter(x=freq, y=aspectrumFiltered, name="После фильтрации")
fig.show()
# %%
restored = np.fft.ifft(filteredSpectrum).real
fig = go.Figure()
fig.update_layout(title="Устранение шума")
fig.add_scatter(x=t, y=s, mode="lines", name="Сигнал 1")
fig.add_scatter(x=t, y=sn, mode="lines", name="Сигнал 1 + шум")
fig.add_scatter(x=t, y=restored, name="После фильтрации")
fig.show()
# %%
