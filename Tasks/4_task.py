import numpy as np
import matplotlib.pyplot as plt


N = 100
# h < (x_max-x_min)/N
h = 0.00001
x_data = np.linspace(0, 2 * np.pi, N)
t_data = np.linspace(0, np.pi, N + 1)


def bessel_function(x, t, m):
    return np.cos(m * t - x * np.sin(t))/np.pi


def simpson_method(x, m):
    integral = 0
    t_step = np.pi/(N + 1)
    y = bessel_function(x, t_data, m)

    for i in range(0, N-1, 2):
        integral += t_step * (y[i] + 4 * y[i+1] + y[i+2])/3
    return integral


def diff(x):
    return (simpson_method(x + h, 0) - simpson_method(x - h, 0))/(2 * h)


J_0, J_1, diffJ_0 = [], [], []
for i in range(0, N):
    J_0.append(simpson_method(x_data[i], 0))
    J_1.append(simpson_method(x_data[i], 1))
    diffJ_0.append(diff(x_data[i]))
    print(f'In {x_data[i]} point: {diffJ_0[i] + J_1[i]}')


# graphics
fig, ax = plt.subplots(figsize=(8, 6))
plt.xlabel('x', labelpad=5, fontsize=12)
plt.ylabel('y', labelpad=5, fontsize=12)
ax.axis([0, 2 * np.pi, -0.7, 1])
ax.plot(x_data, J_0, c='r', label=r'$J_{0}(x)$')
ax.plot(x_data, J_1, c='green', label=r'$J_{1}(x)$')
ax.plot(x_data, diffJ_0, color='red', linestyle='dashed', label=r'$J^{\prime}_{0}(x)$')
plt.legend(loc='upper right')


plt.show()
