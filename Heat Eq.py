import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# const, x&t-grid
L, t_max = 1, 1
n_t, n = 1000, 100
x_step = L/n
t_step = t_max/n_t
x_data,  t_data = np.linspace(0, L, int(1/x_step)), np.linspace(0, t_max, int(1/t_step))

# Dirichlet problem
a = [0] + [- t_step/(2 * pow(x_step, 2)) for _ in range(1, n)]
b = [1 + t_step/pow(x_step, 2) for _ in range(0, n)]
c = [- t_step/(2 * pow(x_step, 2)) for _ in range(0, n-1)] + [0]


def init_u_distribution(x):
    # u(x, 0)
    return x * pow((1 - x/L), 2)


def tridiagonal_matrix_algorithm(v):
    d, s, t = v.copy(), c.copy(), v.copy()

    s[0], t[0] = -s[0]/b[0], d[0]/b[0]
    for i in range(1, n - 1):
        d[i] += t_step * (v[i+1] - 2 * v[i] + v[i-1])/(2 * pow(x_step, 2))
        s[i] = -c[i]/(b[i] + a[i] * s[i-1])
        t[i] = -(a[i] * t[i-1] - d[i])/(b[i] + a[i] * s[i-1])

    y = t.copy()
    for i in range(n - 2, -1, -1):
        y[i] = s[i] * y[i+1] + y[i]
    return y


def k_n_algorithm():
    temperature, temp_max, i = [], [], 0
    temperature.append([])
    temperature[0] = init_u_distribution(x_data)
    temperature[0][0], temperature[0][-1] = 0, 0
    for _ in t_data:
        i += 1
        temperature.append([])
        temperature[i] = tridiagonal_matrix_algorithm(temperature[i-1])
        temp_max.append(round(max(temperature[i]), 5))
    return temperature, temp_max


v = k_n_algorithm()
v_data, v_max = v[0], v[1]


# graphics
fig, axes = plt.subplots(2, 1, figsize=(8, 8))

axes[0].set_title(r'Heat equation $\frac{\partial u}{\partial t} = \frac{\partial^2 u}{\partial x^2}$', fontsize=14, pad=15)
axes[0].set_xlim(-0.001, L)
axes[0].set_ylim(-0.001, 0.15)

axes[1].set_title(r'$T_{max}(t)$', fontsize=14, pad=10)
axes[1].set_xlim(-0.001, t_max)
axes[1].set_ylim(-0.001, 0.15)

# declaring lines for animation
line1, = axes[0].plot(x_data, v_data[0], color='darkorange')
time_text = axes[0].text(0.7, 0.11, r'$u(x, 0) = x \cdot \left(1 - \frac{x}{L}\right)^2$', bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5})
line2, = axes[1].plot(t_data[0], v_max[0], color='black')


def update(num, line1, line2):
    line1.set_data(x_data, v_data[num])
    line2.set_data(t_data[:num], v_max[:num])
    time_text.set_text(r'$u(x, 0) = x \cdot \left(1 - \frac{x}{L}\right)^2$' + f'\n t = {round(t_data[num], 3)} \n' + r'$T_{max} = $' + f' {v_max[num]}')
    return line1, line2, time_text,


animation.FuncAnimation(fig, update, len(v_data), fargs=[line1, line2], interval=20, blit=True)
plt.show()





