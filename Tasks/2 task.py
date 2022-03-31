import numpy as np
import matplotlib.pyplot as plt

# 1 MeV = 1,78 * 10^(-30) kg; 1 eV = 1,6 * 10^(-6) J; 1 Angstrom = 10^(-10) m
const = 0.2848
# precision of determining 0
precision = 0.0000001
# precision of dichotomy method
dichotomy_pre = 0.00001
# precision of simple iteration method
iteration_pre = 0.00001
lambd = 0.01
# precision of Newton method
newton_pre = 0.00001

# getting particle mass, width and depth values of rectangular potential
# m = float(input("Input m[MeV] = "))
# L = float(input("Input L[Angstroms] = "))
# U_0 = float(input("Input U_0[eV] = "))

m, L, U_0 = 1, 10, 1
K = 2 * m * pow(L, 2) * U_0 * const
x_data = np.arange(0 + precision, 1, 0.00001)


def f(x):
    return np.tan(np.sqrt(K * (1 - x))) * np.sqrt(1/x - 1) - 1


def der_f(x):
    return - np.tan(np.sqrt(K * (1 - x)))/(2 * pow(x, 2) * np.sqrt(1/x - 1)) - np.sqrt(K)/(2 * np.sqrt(x) * pow(np.cos(np.sqrt(K * (1 - x))), 2))


def phi(x):
    return x - lambd * np.sign(der_f(x)) * f(x)


def phi_newton(x):
    return x - f(x)/der_f(x)


def ground_state(function):
    return max(list(map(lambda x, y: x if y > x else 0, x_data, function(x_data))))


def graphics(result, length, function):
    y_data = function(x_data)

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.xlabel(r'$\xi$', labelpad=10, fontsize=15)
    ax.axis([0, 1, -2, 2])

    ax.plot(x_data, y_data, c='darkorange')
    # arrows step-by-step
    dist_x, dist_y = [], []
    for i in range(length - 1):
        dist_x.append(result[0][i+1] - result[0][i])
        dist_y.append(result[1][i+1] - result[1][i])
        ax.annotate(s=f'{i}', xy=(result[0][i+1], result[1][i+1]), xytext=(result[0][i], result[1][i]), arrowprops=dict(arrowstyle='-|>', color='black', lw=0))
    ax.scatter(result[0][0:length-1], result[1][0:length-1], c='r')
    ax.plot(result[0][length-1], result[1][length-1], '*', color='yellow', markersize=20)

    return ax


def dichotomy():
    def dichotomy_method(start):
        sol_x, sol_y = [], []
        a, b = start, 1 - precision
        while b - a >= dichotomy_pre:
            c = (a + b)/2
            f_a, f_b, f_c = f(a), f(b), f(c)
            if f_a == 0:
                return a
            elif f_b == 0:
                return b
            elif f_c == 0:
                return c
            elif f_c * f_b < 0:
                a = c
            else:
                b = c
            sol_x.append(c)
            sol_y.append(f_c)
        return sol_x, sol_y

    dichotomy_res = dichotomy_method(ground_state(f))
    dichotomy_len = len(dichotomy_res[0])
    # output
    ax = graphics(dichotomy_res, dichotomy_len, f)
    plt.ylabel(r'$f(\xi)$', labelpad=10, fontsize=15)
    ax.plot([-5, 5], [0, 0], c='k', label='y = 0', linestyle='dashed')
    plt.legend(loc='upper left')
    plt.title('Dichotomy', fontsize=15, pad=20)
    print('Dichotomy:' + '\n \t Ground state energy level = ' + str(- U_0 * dichotomy_res[0][dichotomy_len - 1]) + ' eV'
          + '\n \t Number of steps = ' + str(dichotomy_len) + ' with precision ' + str(dichotomy_pre)
          + '\n \t Total error = ' + str(abs(dichotomy_res[1][dichotomy_len - 1])))


def iteration():
    def iteration_method(x_0, max_steps=100):
        sol_x, sol_y, i = [], [], 0
        for _ in range(max_steps):
            x_1 = x_0
            x_0 = phi(x_0)
            sol_x.append(x_1)
            sol_y.append(x_0)
            if x_0 >= 1 or x_0 <= 0:
                raise Exception('ERROR: Choose another initial point')
            if abs(x_0 - x_1) < iteration_pre:
                return sol_x, sol_y
        return sol_x, sol_y

    iteration_res = iteration_method(float(input(f'Input initial point (from {ground_state(phi)} to 1) = ')))
    iteration_len = len(iteration_res[0])
    # output
    ax = graphics(iteration_res, iteration_len, phi)
    ax.plot([-5, 5], [-5, 5], c='k', label='y = x', linestyle='dashed')
    plt.ylabel(r'$x + \lambda \cdot f(\xi)$', labelpad=10, fontsize=15)
    plt.legend(loc='upper left')
    plt.title('Iteration', fontsize=15, pad=20)
    print('Iteration:' + '\n \t Ground state energy level = ' + str(- U_0 * iteration_res[0][iteration_len - 1]) + ' eV'
          + '\n \t Number of steps = ' + str(iteration_len) + ' with precision ' + str(iteration_pre)
          + '\n \t Total error = ' + str(abs(f(iteration_res[0][iteration_len - 1]))))


def newton():
    def newton_method(x_0, max_steps=100):
        sol_x, sol_y, i = [], [], 0
        for _ in range(max_steps):
            x_1 = x_0
            x_0 = phi_newton(x_0)
            sol_x.append(x_1)
            sol_y.append(x_0)
            if x_0 >= 1 or x_0 <= 0:
                raise Exception('ERROR: Choose another initial point')
            if abs(x_0 - x_1) < newton_pre:
                return sol_x, sol_y
        return sol_x, sol_y

    newton_res = newton_method(float(input(f'Input initial point (from {ground_state(phi_newton)} to 1) = ')))
    newton_len = len(newton_res[0])
    # output
    ax = graphics(newton_res, newton_len, phi_newton)
    ax.plot([-5, 5], [-5, 5], c='k', label='y = x', linestyle='dashed')
    plt.ylabel(r'$x + \frac{f(\xi)}{f^{\prime}(\xi)}$', labelpad=5, fontsize=15)
    plt.legend(loc='upper left')
    plt.title('Newton', fontsize=15, pad=20)
    print('Newton:' + '\n \t Ground state energy level = ' + str(- U_0 * newton_res[0][newton_len - 1]) + ' eV'
          + '\n \t Number of steps = ' + str(newton_len) + ' with precision ' + str(iteration_pre)
          + '\n \t Total error = ' + str(abs(f(newton_res[0][newton_len - 1]))))


dichotomy()
iteration()
newton()
plt.show()
