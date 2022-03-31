import numpy as np
import matplotlib.pyplot as plt

n = 20


def y_k(x):
    return np.log(x)


def div_difference(x, y):
    x, y_div = np.copy(x), np.copy(y)
    for k in range(1, len(x)):
        y_div[k:] = (y_div[k:] - y_div[k - 1])/(x[k:] - x[k - 1])
    return y_div


def horner_scheme(x, y, x_point):
    coeff = div_difference(x, y)
    p_x = coeff[n]
    for k in range(1, n + 1):
        p_x = coeff[n - k] + (x_point - x[n - k]) * p_x
    return p_x


# x-grid
x_k = [1 + x/n for x in range(0, n+1)]
x_data = np.arange(0.5, 2.5, 0.001)

y_data = []
for i in range(len(x_data)):
    y_data.append(horner_scheme(x_k, y_k(x_k), x_data[i]))

# graphics
fig, axes = plt.subplots(2, 1, figsize=(8, 8))

axes[0].set_title('Interpolation', fontsize=16)
axes[0].set_xlabel('x', labelpad=3, fontsize=15)
axes[0].set_ylabel('y', labelpad=5, fontsize=15)
axes[0].axis([0.99, 2.01, -0.01, 0.8])

axes[1].set_xlabel('x', labelpad=3, fontsize=15)
axes[1].set_ylabel(r'$P_{'f'{n}''}(x)$ - ln x', labelpad=5, fontsize=15)
# axes[1].axis([0.99, 2.01, -0.0005, 0.0005])


axes[0].plot(x_data, y_k(x_data), color='darkorange', label=r'$y = \ln{x}$')
axes[0].scatter(x_k, y_k(x_k), color='black')
axes[0].plot(x_data, y_data, color='black', linestyle='dashed', label=r'$P_{'f'{n}''}(x)$')
axes[0].legend(loc='upper left')

axes[1].plot(x_data, y_k(x_data) - y_data, color='black', label='Error value')
axes[1].legend(loc='upper left')


plt.show()
