import matplotlib.pyplot as plt
import numpy as np

# const
alfa = 0.75
a, b, c, d = 0.4, 0.11, 0.6, 0.35
E, F = 0.05, 0.01

# initial conditions: x(t_0) = x_0, y(t_0) = y_0
t_0, x_0, y_0 = 0, 10, 5

# time range
t_min, t_max = 0, 12
n = t_max * 1000


def syst_diff_eq(x, y):
    return a * x - b * x * y + E / x, d * x * y - c * y + F / y


def runge_kutta_2():
    x, y, t = [], [], []
    h = (t_max - t_min)/n
    x.append(x_0)
    y.append(y_0)
    t.append(t_0)
    for i in range(n + 1):
        f_i, g_i = syst_diff_eq(x[i], y[i])[0], syst_diff_eq(x[i], y[i])[1]
        x.append(x[i] + h * ((1 - alfa) * f_i + alfa * syst_diff_eq(x[i] + h * f_i/(2 * alfa), y[i] + h * g_i/(2 * alfa))[0]))
        y.append(y[i] + h * ((1 - alfa) * g_i + alfa * syst_diff_eq(x[i] + h * f_i/(2 * alfa), y[i] + h * g_i/(2 * alfa))[1]))
        t.append(t_0 + h * i)
    return x, y, t


rung = runge_kutta_2()
x_rung, y_rung, t_rung = rung[0], rung[1], rung[2]

# graphics
fig, axes = plt.subplots(2, 1, figsize=(8, 8))

axes[0].set_title('Phase trajectory y(x)')
axes[0].plot(x_rung, y_rung, c='red')

axes[1].set_title('Time dependence')
axes[1].plot(t_rung, x_rung, c='darkorange', label='x(t)')
axes[1].plot(t_rung, y_rung, c='black', label='y(t)')
plt.legend(loc='upper left')
print(x_rung[-1], t_rung[-1])

plt.show()
