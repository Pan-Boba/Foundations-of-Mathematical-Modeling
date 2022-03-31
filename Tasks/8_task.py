import matplotlib.pyplot as plt
import numpy as np


# initial conditions: u(x_0) = u_0, v(x_0) = v_0
x_0, u_0, v_0 = 0, 1, 1

# x range
x_min, x_max = 0, 5

# h < 2 / max(abs(eigenvalues)) = 2 / 1000
n_expl = 1000 * x_max
h_start_ex, h_end_ex = (10/n_expl - x_min)/100, (x_max - 10/n_expl)/n_expl


n_impl = 100
h_start_im, h_end_im = (0.04 - x_min)/100, (x_max - 0.04)/n_impl


def syst_diff_eq(u, v):
    # eigenvalues = -1000, -1
    return 998 * u + 1998 * v, -999 * u - 1999 * v


def syst_diff_impl(u, v, h):
    # was derived manually due to linearity of f(u, v)
    return ((1 + 1999 * h) * u + 1998 * h * v)/(999 * 1998 * pow(h, 2) - (998 * h - 1) * (1999 * h + 1)), \
           ((1 - 998 * h) * v - 999 * h * u)/(999 * 1998 * pow(h, 2) - (998 * h - 1) * (1999 * h + 1))


def solution(x):
    # for x_0, u_0, v_0 = 0, 1, 1
    return 4.0 * np.exp(-x) - 3.0 * np.exp(-1000 * x), -2.0 * np.exp(-x) + 3.0 * np.exp(-1000 * x)


def euler_explicit():
    x, u, v = [], [], []
    x.append(x_0)
    u.append(u_0)
    v.append(v_0)
    for i in range(0, 101):
        y_i = syst_diff_eq(u[i], v[i])
        u.append(u[i] + h_start_ex * y_i[0])
        v.append(v[i] + h_start_ex * y_i[1])
        x.append(x_0 + i * h_start_ex)
    for i in range(101, n_expl + 1):
        y_i = syst_diff_eq(u[i], v[i])
        u.append(u[i] + h_end_ex * y_i[0])
        v.append(v[i] + h_end_ex * y_i[1])
        x.append(x[i] + h_end_ex)
    return u, v, x


def euler_implicit():
    x, u, v = [], [], []
    x.append(x_0)
    u.append(u_0)
    v.append(v_0)
    for i in range(0, 101):
        y_i = syst_diff_impl(u[i], v[i], h_start_im)
        u.append(y_i[0])
        v.append(y_i[1])
        x.append(x_0 + i * h_start_im)
    for i in range(101, n_impl + 1):
        y_i = syst_diff_impl(u[i], v[i], h_end_im)
        u.append(y_i[0])
        v.append(y_i[1])
        x.append(x_0 + i * h_end_im)
    return u, v, x


eul_expl = euler_explicit()
eul_impl = euler_implicit()

u_e, v_e, x_e = eul_expl[0], eul_expl[1], eul_expl[2]
u_i, v_i, x_i = eul_expl[0], eul_expl[1], eul_expl[2]

x_data = np.arange(x_min, x_max, h_start_ex)
y = solution(x_data)
u_data, v_data = y[0], y[1]


# graphics
fig, axes = plt.subplots(2, 1, figsize=(8, 8))
axes[0].set_title('Explicit Euler scheme')
axes[0].plot(x_data, u_data, c='darkorange', label='Solution u(x)', linestyle='dashed')
axes[0].plot(x_e[1:], u_e[1:], c='darkorange', label='u(x)')
axes[0].plot(x_data, v_data, c='grey', label=' Solution v(x)', linestyle='dashed')
axes[0].plot(x_e[1:], v_e[1:], c='grey', label='v(x)')
axes[0].text(2, 3.2, f'h_start = {h_start_ex} \nh_end = {h_end_ex}', bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5})
axes[0].legend(loc='upper right')

axes[1].set_title('Implicit Euler scheme')
axes[1].plot(x_data, u_data, c='red', label='Solution u(x)', linestyle='dashed')
axes[1].plot(x_i, u_i, c='red', label='u(x)')
axes[1].plot(x_data, v_data, c='black', label='Solution v(x)', linestyle='dashed')
axes[1].plot(x_i, v_i, c='black', label='v(x)')
axes[1].text(2, 3.2, f'h_start = {h_start_im} \nh_end = {h_end_im}', bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5})
axes[1].legend(loc='upper right')



plt.show()
