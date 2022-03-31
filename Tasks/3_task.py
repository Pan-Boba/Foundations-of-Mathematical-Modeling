import matplotlib.pyplot as plt
import numpy as np

# using Wolfram Alpha
integral_f, integral_g = np.pi/2, 1.29587400873170808779041249
N_max = 32
fig, axes = plt.subplots(int(np.log2(N_max)), 2, figsize=(12, 10))


# first integral
def f(x):
    return [1 / (1 + pow(x, 2)), -1, 1, 'f']


# second integral
def g(x):
    return [pow(x, 1/3) * np.exp(np.sin(x)), 0, 1, 'g']


def inv_2_pow(x):
    return 0.3/pow(x, 2)


def inv_4_pow(x):
    return 1/pow(x, 4)


def graphics(function, x, N, number, S):
    x_min, x_max = function(0)[1], function(0)[2]
    x_data = np.linspace(x_min, x_max, 100)
    y_data = function(x_data)[0]
    axes[0][0].set_title('Fragmentation of  integration segment for ' + str(function(0)[3]) + '\n using Trapezoid method')
    axes[0][1].set_title('Fragmentation of  integration segment for ' + str(function(0)[3]) + '\n using Simpsons method')

    axes[int(np.log2(N) - 1)][number].plot(x_data, y_data, color='darkorange')
    axes[int(np.log2(N) - 1)][number].text(x_min, 0.75 * max(y_data), f'N = {N} \nS = {S}', bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5})

    for i in range(N):
        xs = [x[i], x[i], x[i + 1], x[i + 1]]
        ys = [0, function(x[i])[0], function(x[i + 1])[0], 0]
        axes[int(np.log2(N) - 1)][number].fill(xs, ys, 'b', edgecolor='black', alpha=0.2)


def trapezoid_method(function, N):
    s_area = 0
    x_min, x_max = function(0)[1], function(0)[2]
    x = np.linspace(x_min, x_max, N + 1)
    h = x[1] - x[0]
    y = function(x)[0]

    for i in range(len(x) - 1):
        s_area += h * (y[i] + y[i+1])/2
    if function(0)[3] == 'f':
        error = abs(s_area - integral_f)
    else:
        error = abs(s_area - integral_g)
    graphics(function, x, N, 0, round(s_area, 4))
    return s_area, error


def simpson_method(function, N):
    s_area = 0
    x_min, x_max = function(0)[1], function(0)[2]
    x = np.linspace(x_min, x_max, N + 1)
    h = x[1] - x[0]
    y = function(x)[0]

    for i in range(0, len(x) - 2, 2):
        s_area += h * (y[i] + 4 * y[i+1] + y[i+2])/3
    if function(0)[3] == 'f':
        error = abs(s_area - integral_f)
    else:
        error = abs(s_area - integral_g)
    graphics(function, x,  N, 1, round(s_area, 4))
    return s_area, error


err_trap, err_simp = [], []
for i in range(1, int(np.log2(N_max) + 1)):
    err_trap.append(trapezoid_method(f, pow(2, i))[1])
    err_simp.append(simpson_method(f, pow(2, i))[1])
    print(f'Trapezoid error for {pow(2, i)}: {err_trap [i-1]}' + f'\t Simpson error for {pow(2, i)}: {err_simp[i-1]}')


fig, ax = plt.subplots(figsize=(8, 6))
plt.xlabel('N', labelpad=10, fontsize=15)
plt.ylabel('Error value', labelpad=10, fontsize=15)
ax.axis([0, N_max + 1, -0.01, 0.1])
ax.plot(np.arange(1, N_max + 1, 0.1), inv_2_pow(np.arange(1, N_max + 1, 0.1)), c='black')
ax.plot(np.arange(1, N_max + 1, 0.1), inv_4_pow(np.arange(1, N_max + 1, 0.1)), c='darkorange')
ax.scatter([pow(2, x) for x in range(1, int(np.log2(N_max))+1)], err_simp, c='darkorange', label='Simpson method')
ax.scatter([pow(2, x) for x in range(1, int(np.log2(N_max))+1)], err_trap, c='black', label='Trapezoid method')
plt.legend(loc='upper right')


plt.show()
