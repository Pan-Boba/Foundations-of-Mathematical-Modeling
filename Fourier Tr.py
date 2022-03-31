import numpy as np
import matplotlib.pyplot as plt
import cmath


# const, t-grid
a_0, a_1, w_0, w_1 = 1, 0.02, 5.1, 25.5
t_interval = 2 * np.pi
n = 300
t_data = np.linspace(0, t_interval, n)


def f(t):
    return a_0 * np.sin(w_0 * t) + a_1 * np.sin(w_1 * t)


def rectangular_window(k):
    return 1


def haan_window(k):
    return (1 - np.cos(2 * np.pi * k/n))/2


def discrete_fourier_transform(window):
    w, dft = [], []
    # anti-aliasing
    for i in range(0, int(n/2)):
        f_j = complex(0, 0)
        for k in range(0, n):
            f_j += f(t_data[k]) * cmath.exp(2 * np.pi * 1j * i * k/n) * window(k)
        # w > 0 for real signal
        w.append(i * 2 * np.pi/t_interval)
        dft.append(abs(f_j)/n)
    return w, dft


spectrum_rect = discrete_fourier_transform(rectangular_window)
spectrum_haan = discrete_fourier_transform(haan_window)
w_data, intensity_rect, intensity_haan = spectrum_rect[0], spectrum_rect[1], spectrum_haan[1]


# graphics
fig, axes = plt.subplots(3, 1, figsize=(8, 12))
axes[0].set_xlim(min(t_data), max(t_data))
axes[0].set_title(r'Signal f(t) = ' f'{a_0} ' + r'$\cdot$ ' + f'sin({w_0} ' + r'$\cdot$ ' + 't)' +
             f' + {a_1} ' + r'$\cdot$ ' + f'sin({w_1} ' + r'$\cdot$ ' + 't)')
axes[0].text(2, 0.8, f'n = {n}', bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5})
axes[0].plot(t_data, f(t_data), c='black')

axes[1].set_xlim(1, max(w_0, w_1) * 3)
axes[1].set_ylim(pow(10, -3), pow(10, 0))
axes[1].set_yscale('log')
axes[1].set_title(r'Spectrum $|f(\omega)|$ with rectangular window')
axes[1].scatter(w_data[1:], intensity_rect[1:], c='darkorange', s=10)
axes[1].plot(w_data[1:], intensity_rect[1:], c='black', linestyle='dashed')

axes[2].set_xlim(1, max(w_0, w_1) * 3)
axes[2].set_ylim(pow(10, -7), pow(10, 0))
axes[2].set_yscale('log')
axes[2].set_title(r'Spectrum $|f(\omega)|$ with Haan window')
axes[2].scatter(w_data[1:], intensity_haan[1:], c='red', s=10)
axes[2].plot(w_data[1:], intensity_haan[1:], c='black', linestyle='dashed')


plt.show()



