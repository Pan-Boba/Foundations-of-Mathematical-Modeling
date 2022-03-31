import matplotlib.pyplot as plt
import numpy as np

# const
n = 100
alfa = 0.75


def f(x):
    # dx/dt = f(x) = -x, 0 = a < t < b =3, x(t = 0) = 1
    return [-x, 0, 3, [0, 1]]


t_min, t_max, x_0, t_0 = f(0)[1], f(0)[2], f(0)[3][1], f(0)[3][0]
h = (t_max - t_min)/n


def solution(t):
    x = 1/np.exp(t)
    return x


def euler():
    x, t = [], []
    t.append(t_0)
    x.append(x_0)
    for i in range(n + 1):
        x.append(x[i] + h * f(x[i])[0])
        t.append(t_0 + h * i)
    return t, x


def runge_kutta_2():
    x, t = [], []
    t.append(t_0)
    x.append(x_0)
    for i in range(n + 1):
        f_i = f(x[i])[0]
        x.append(x[i] + h * ((1 - alfa) * f_i + alfa * f(x[i] + h * f_i/(2 * alfa))[0]))
        t.append(t_0 + h * i)
    return t, x


def runge_kutta_4():
    x, t = [], []
    t.append(t_0)
    x.append(x_0)
    for i in range(n + 1):
        k_1 = f(x[i])[0]
        k_2 = f(x[i] + h * k_1/2)[0]
        k_3 = f(x[i] + h * k_2/2)[0]
        k_4 = f(x[i] + h * k_3)[0]

        x.append(x[i] + h * (k_1 + 2 * k_2 + 2 * k_3 + k_4)/6)
        t.append(t_0 + h * i)
    return t, x


t_sol = np.arange(-1, 4, 0.01)
x_sol, eul, rung2, rung4 = solution(t_sol), euler(), runge_kutta_2(), runge_kutta_4()
t_eul, x_eul = eul[0], eul[1]
t_rung2, x_rung2 = rung2[0], rung2[1]
t_rung4, x_rung4 = rung4[0], rung4[1]

# graphics
fig, axes = plt.subplots(2, 1, figsize=(8, 8))

axes[0].set_title('Solution x(t)')
axes[0].axis([-0.01, 3.01, 1.1, 0])
axes[0].plot(t_sol, x_sol, c='black', linestyle='dashed', label='Real solution')
axes[0].plot(t_eul[1:], x_eul[1:], c='darkorange', label='Euler 1')
axes[0].plot(t_rung2[1:], x_rung2[1:], c='red', label='R-K 2')
axes[0].plot(t_rung4[1:], x_rung4[1:], c='green', label='R-K 4')
axes[0].legend(loc='upper left')

x_data = solution(t_eul[1:])
axes[1].set_title('Error value(t)')
axes[1].plot(t_eul[1:], x_data - x_eul[1:], c='darkorange', label='Euler 1 accuracy')
axes[1].plot(t_eul[1:], x_data - x_rung2[1:], c='red', label='R-K 2 accuracy')
axes[1].plot(t_eul[1:], x_data - x_rung4[1:], c='green', label='R-K 4 accuracy')
axes[1].legend(loc='upper right')

plt.show()




